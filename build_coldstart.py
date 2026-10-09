"""Build controlled cold-start evaluation sets for DEP (SOP Sec. III-B, III-C).

For every test user we keep ONLY the K most recent reviews (K = 1, 2, 5) as the
available history, and build three variants of the model input:

  orig          DEP exactly as released, but with only K reviews available
  enrich_text   + compact preference profile (text) appended to the prompt
  enrich_blend  + the same text AND the profile embedding blended into the
                user representation before inter-user difference modeling

The same users, target items, peers and reference reviews are used in every
condition, so differences are caused by the history size / the enrichment only.

Run from the root of the official DEP repo (after `python embedding.py`):

    python -m coldstart.build_coldstart --ks 1 2 5 \
        --variants orig enrich_text enrich_blend --alpha 0.3

Output (per category / K / variant):  <out_dir>/<cat>_K<k>_<variant>/
    data.jsonl  - inp_str, out_str, user_id, metadata
    emb.pt      - float16 tensor (N, 16, 1024): rows 0-7 = review embeddings,
                  rows 8-15 = difference embeddings (the layout the DEP model
                  and the DEP vLLM fork expect). Unused rows are zero.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from coldstart.enrich import PreferenceEnricher

CATEGORIES = ["Books", "Movies_and_TV", "CDs_and_Vinyl"]
VARIANTS = ["orig", "enrich_text", "enrich_blend"]
N_SLOTS = 8          # DEP has 8 review slots and 8 difference slots
EMB_DIM = 1024       # bge-m3

NEW_TOKENS = ([f"[HIS_TOKEN_{i}]" for i in range(N_SLOTS)]
              + [f"[DIFF_TOKEN_{i}]" for i in range(N_SLOTS)]
              + ["<his_token_start>", "<his_token_end>",
                 "<diff_token_start>", "<diff_token_end>"])

# Copied verbatim from data/personal_dataset.py of the official repo.
# It is deliberately NOT changed for the enrichment variants, so that the only
# difference between variants is the extra [User Preference Profile] block.
SYSTEM_PROMPT = (
    f"Given the title and description of an item, "
    f"along with the user's past reviews (including item title, item description, review rating, review title, review text, review embedding, review difference embedding), "
    f"and the output review rating and review title, "
    f"generate a personalized item review for the user.\n"
    f"Note: [Review Embedding] denotes a soft prompt of the review text and [Review Difference Embedding] denotes a soft prompt showing the difference between the review text and other reviews on the same item. "
    f"[Review Embedding] and [Review Difference Embedding] should serve as hints for personalized review text generation.\n"
)


def qwen_prompt(user_message: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    # identical to utils/templates.py::Qwen2PromptTemplate.build_prompt
    return (f"<|im_start|>system\n{system_prompt}<|im_end|>"
            f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n")


# ----------------------------------------------------------------------------
# global peer maps (same logic as create-dataset.py)
# ----------------------------------------------------------------------------
class PeerMaps:
    """user_his_emb_map / user_prof_mean_emb_map / asin_reviewers_map.

    Built from the TEST split exactly like the official create-dataset.py:
    for each user the last two profile reviews are dropped, the rest is
    embedded, and every (user, index) is registered under the item's ASIN.
    """

    def __init__(self):
        self.his_emb: Dict[str, torch.Tensor] = {}
        self.prof_mean: Dict[str, torch.Tensor] = {}
        self.asin_reviewers = defaultdict(set)

    def add_user(self, user_id: str, category: str, profile: list,
                 emb_ascending: torch.Tensor):
        profile = profile[:-2]
        his = emb_ascending[:-2]
        key = f"{user_id}_{category}"
        self.his_emb[key] = his
        self.prof_mean[key] = torch.mean(his, dim=0)
        for i, p in enumerate(profile):
            self.asin_reviewers[p["asin"]].add((user_id, i))


def aggregate_difference(user_id: str, category: str, asin: str,
                         user_rep: torch.Tensor, user_mean: torch.Tensor,
                         maps: PeerMaps) -> Optional[torch.Tensor]:
    """Difference-aware embedding of one review vs. peers (as in DEP).

    Same arithmetic as data/personal_dataset.py. Returns None when the item
    has no other reviewer. NOTE: `user_mean` is passed in explicitly (computed
    from the K available reviews only) instead of being read from the global
    map, which would leak the user's full history into the cold-start setting.
    """
    diffs = []
    for r_uid, p_idx in sorted(maps.asin_reviewers.get(asin, ())):
        if r_uid == user_id:
            continue
        peer_emb = maps.his_emb[f"{r_uid}_{category}"][p_idx]
        peer_mean = maps.prof_mean[f"{r_uid}_{category}"]
        diffs.append((user_mean - peer_mean, user_rep - peer_emb))
    if not diffs:
        return None
    w = torch.stack([d[0] for d in diffs]).norm(dim=1)
    s = w.sum()
    w = w / s if s > 0 else torch.full_like(w, 1.0 / len(w))
    w = torch.softmax(w, dim=0)   # kept from the official code
    return torch.stack([w[i] * d[1] for i, d in enumerate(diffs)]).sum(0)


# ----------------------------------------------------------------------------
# optional: embed the profile text with the same encoder DEP uses (bge-m3)
# ----------------------------------------------------------------------------
class ProfileEmbedder:
    def __init__(self, name: str = "BAAI/bge-m3", device: Optional[str] = None):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, text: str) -> torch.Tensor:
        batch = self.tok([text], truncation=True, padding=True,
                         return_tensors="pt").to(self.device)
        out = self.model(**batch)[0][:, 0]            # CLS, as in embedding.py
        return F.normalize(out, p=2, dim=1)[0].float().cpu()


# ----------------------------------------------------------------------------
# one sample
# ----------------------------------------------------------------------------
def build_sample(sample: dict, category: str, K: int, variant: str,
                 emb_loader: Callable[[str, str], torch.Tensor],
                 maps: PeerMaps, meta: Dict[str, Tuple[str, str]],
                 count_tokens: Callable[[str], int],
                 enricher: Optional[PreferenceEnricher] = None,
                 profile_embedder: Optional[Callable[[str], torch.Tensor]] = None,
                 alpha: float = 0.3, max_length: int = 3072) -> dict:
    assert variant in VARIANTS and 1 <= K <= N_SLOTS
    user_id = sample["user_id"]
    data = sample["data"]

    # most-recent-first profile, aligned with its embeddings
    profile = sorted(sample["profile"], key=lambda x: x["timestamp"], reverse=True)
    emb = emb_loader(category, user_id)[: len(profile)]       # ascending in time
    emb = torch.flip(emb, dims=[0])                            # -> most recent first
    k = min(K, len(profile))
    profile, emb = profile[:k], emb[:k]                        # <- the cold-start cut

    for p in profile:
        p["item_title"], p["item_desc"] = meta.get(p["asin"], ("", ""))
    item_title, item_desc = meta.get(data["asin"], ("", ""))

    # -- optional enrichment ------------------------------------------------
    profile_text, reps = "", emb
    if variant != "orig":
        assert enricher is not None
        profile_text = enricher.enrich(profile).to_text()
        if variant == "enrich_blend":
            assert profile_embedder is not None and alpha > 0
            pe = profile_embedder(profile_text)
            reps = F.normalize(emb + alpha * pe.unsqueeze(0), p=2, dim=1)

    # -- DEP representation from the K available reviews only ---------------
    user_mean = reps.mean(dim=0)                               # no full-history leak
    his_diff = torch.zeros(2 * N_SLOTS, EMB_DIM)
    has_diff: List[bool] = []
    for i in range(k):
        his_diff[i] = reps[i]
        d = aggregate_difference(user_id, category, profile[i]["asin"],
                                 reps[i], user_mean, maps)
        if d is not None:
            his_diff[N_SLOTS + i] = d
        has_diff.append(d is not None)

    # -- prompt (same format/trimming rule as the official dataset code) -----
    def review_block(i: int) -> str:
        p = profile[i]
        s = (f"- [Review {i+1}]:\n"
             f"  - [Item Title]: {p['item_title']}\n"
             f"  - [Item Description]: {p['item_desc']}\n"
             f"  - [Review Rating]: {p['rating']}\n"
             f"  - [Review Title]: {p['title']}\n"
             f"  - [Review Text]: {p['text']}\n"
             f"  - [Review Embedding]: <his_token_start>{NEW_TOKENS[i]}<his_token_end>\n")
        if has_diff[i]:   # no peers -> no (meaningless) difference token
            s += (f"  - [Review Difference Embedding]: "
                  f"<diff_token_start>{NEW_TOKENS[i + N_SLOTS]}<diff_token_end>\n")
        return s

    tail = (f"[Output Review Rating]: {data['rating']}\n"
            f"[Output Review Title]: {data['title']}\n")
    head = f"[Item Title]: {item_title}\n[Item Description]: {item_desc}\n"
    fixed = qwen_prompt(head + profile_text + tail)
    avail = max_length - count_tokens(fixed)

    n_used, past = k, ""
    for n_used in range(k, 0, -1):
        past = "[User's Past Reviews]:\n" + "".join(review_block(i) for i in range(n_used))
        if count_tokens(past) <= avail:
            break
    inp_str = qwen_prompt(head + past + profile_text + tail)

    return dict(
        inp_str=inp_str, out_str=data["text"], user_id=user_id,
        category=category, K=K, k_available=k, k_in_prompt=n_used,
        n_with_peers=int(sum(has_diff)), profile_text=profile_text,
        emb=his_diff,
    )


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def save_split(rows: List[dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    embs = torch.stack([r.pop("emb") for r in rows]).to(torch.float16)
    torch.save(embs, os.path.join(out_dir, "emb.pt"))
    with open(os.path.join(out_dir, "data.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=CATEGORIES, choices=CATEGORIES)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 5])
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="weight of the profile embedding in enrich_blend")
    ap.add_argument("--emb_dir", default="embeddings")
    ap.add_argument("--out_dir", default="data/coldstart")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N test users per category (same users for every condition)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_length", type=int, default=3072)
    args = ap.parse_args()

    from datasets import concatenate_datasets, load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    tok.add_special_tokens({"additional_special_tokens": NEW_TOKENS})   # as official
    count_tokens = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])

    # peer maps are built on ALL test users, like the official script
    per_cat = {c: load_dataset("SnowCharmQ/DPL-main", c, split=args.split)
               for c in CATEGORIES}
    meta_ds = concatenate_datasets([load_dataset("SnowCharmQ/DPL-meta", c, split="full")
                                    for c in CATEGORIES])
    meta = dict(zip(meta_ds["asin"], zip(meta_ds["title"], meta_ds["description"])))

    emb_loader = lambda cat, uid: torch.load(
        f"{args.emb_dir}/{cat}/{uid}.emb", weights_only=True)
    maps = PeerMaps()
    for cat, ds in per_cat.items():
        for s in ds:
            prof = s["profile"]
            ts = [p["timestamp"] for p in prof]
            if ts != sorted(ts):   # DEP assumes ascending raw order (profile[:-2] <-> emb[:-2])
                print(f"[warn] profile of {s['user_id']} is not time-ordered")
            maps.add_user(s["user_id"], cat, prof, emb_loader(cat, s["user_id"]))

    embedder = None
    if "enrich_blend" in args.variants:
        embedder = ProfileEmbedder()

    for cat in args.categories:
        ds = per_cat[cat]
        n = len(ds) if args.limit is None else min(args.limit, len(ds))
        enricher = PreferenceEnricher(cat)
        for K in args.ks:
            for variant in args.variants:
                rows = [build_sample(ds[i], cat, K, variant, emb_loader, maps, meta,
                                     count_tokens, enricher, embedder, args.alpha,
                                     args.max_length) for i in range(n)]
                no_peer = sum(r["n_with_peers"] == 0 for r in rows)
                print(f"{cat} K={K} {variant}: {len(rows)} users, "
                      f"{no_peer} with no peer-difference at all")
                save_split(rows, f"{args.out_dir}/{cat}_K{K}_{variant}")


if __name__ == "__main__":
    main()
