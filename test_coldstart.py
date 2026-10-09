"""CPU-only sanity tests on synthetic data (no GPU / HF access needed).

    PYTHONPATH=. python tests/test_coldstart.py
"""
import copy
import random

import torch
import torch.nn.functional as F

from coldstart.build_coldstart import (EMB_DIM, N_SLOTS, PeerMaps, build_sample)
from coldstart.enrich import PreferenceEnricher

CAT = "Books"
torch.manual_seed(0)
random.seed(0)


def make_world(n_users=6, prof_len=12, n_items=8):
    """Users review overlapping items so that peers exist."""
    samples, embs = [], {}
    for u in range(n_users):
        profile = []
        for t in range(prof_len):
            asin = f"A{(u + t) % n_items}"
            profile.append(dict(asin=asin, rating=random.choice([2, 4, 5]),
                                title=f"t{u}-{t}", text=f"I loved the story number {t}.",
                                timestamp=1000 + t))        # ascending, like DPL
        target = dict(asin=f"A{u % n_items}", rating=5, title="target",
                      text="reference review text", timestamp=2000)
        samples.append(dict(user_id=f"u{u}", profile=profile, data=target))
        # embeddings: profile (ascending) + target text, unit norm
        embs[f"u{u}"] = F.normalize(torch.randn(prof_len + 1, EMB_DIM), dim=1)
    return samples, embs


def make_maps(samples, embs):
    maps = PeerMaps()
    for s in samples:
        maps.add_user(s["user_id"], CAT, s["profile"], embs[s["user_id"]])
    return maps


META = {f"A{i}": (f"Book {i}", f"A science fiction story {i}") for i in range(8)}
count = lambda s: len(s.split())          # whitespace "tokenizer" stand-in


def build(sample, embs, maps, K, variant, **kw):
    return build_sample(sample, CAT, K, variant, lambda c, u: embs[u], maps, META,
                        count, PreferenceEnricher(CAT), **kw)


def test_shapes_and_layout():
    samples, embs = make_world()
    maps = make_maps(samples, embs)
    for K in (1, 2, 5):
        r = build(copy.deepcopy(samples[0]), embs, maps, K, "orig")
        e = r["emb"]
        assert e.shape == (2 * N_SLOTS, EMB_DIM)
        assert e[K:N_SLOTS].abs().sum() == 0, "unused review rows must be zero"
        assert e[N_SLOTS + K:].abs().sum() == 0, "unused diff rows must be zero"
        for i in range(K):                         # row i = i-th MOST RECENT review
            want = embs["u0"][:-1].flip(0)[i]
            assert torch.allclose(e[i], want)
        # prompt contains exactly K history tokens, no extra
        for i in range(N_SLOTS):
            assert (f"[HIS_TOKEN_{i}]" in r["inp_str"]) == (i < K)
        assert "Preference Profile" not in r["inp_str"]
    print("ok  shapes / layout / token<->row alignment")


def test_no_full_history_leak():
    """Changing anything OLDER than the K available reviews must not change the input."""
    samples, embs = make_world()
    maps = make_maps(samples, embs)
    K = 2
    base = build(copy.deepcopy(samples[0]), embs, maps, K, "orig")

    s2 = copy.deepcopy(samples[0])
    for p in s2["profile"][:-K]:                   # raw order is ascending -> oldest first
        p["text"] = "COMPLETELY DIFFERENT OLD REVIEW"
    embs2 = dict(embs)
    e = embs["u0"].clone()
    e[: -1 - K] = F.normalize(torch.randn(e.shape[0] - 1 - K, EMB_DIM), dim=1)
    embs2["u0"] = e
    # NOTE: the global map of the *target* user is also rebuilt from changed embs
    maps2 = make_maps([s2] + samples[1:], embs2)
    other = build(s2, embs2, maps2, K, "orig")
    assert other["inp_str"] == base["inp_str"]
    assert torch.allclose(other["emb"], base["emb"], atol=1e-6)
    print("ok  no leakage from reviews outside the K most recent")


def test_variants():
    samples, embs = make_world()
    maps = make_maps(samples, embs)
    o = build(copy.deepcopy(samples[1]), embs, maps, 2, "orig")
    t = build(copy.deepcopy(samples[1]), embs, maps, 2, "enrich_text")
    assert torch.allclose(o["emb"], t["emb"]), "text-only enrichment must not touch embeddings"
    assert "[User Preference Profile]" in t["inp_str"] and "science fiction" in t["inp_str"]
    assert o["inp_str"] != t["inp_str"]

    fake_embedder = lambda text: F.normalize(torch.randn(EMB_DIM), dim=0)
    b = build(copy.deepcopy(samples[1]), embs, maps, 2, "enrich_blend",
              profile_embedder=fake_embedder, alpha=0.3)
    assert not torch.allclose(o["emb"][:2], b["emb"][:2])
    assert torch.allclose(b["emb"][:2].norm(dim=1), torch.ones(2), atol=1e-5)
    print("ok  variants differ only where intended")


def test_prompt_budget():
    samples, embs = make_world()
    maps = make_maps(samples, embs)
    r = build_sample(copy.deepcopy(samples[0]), CAT, 5, "enrich_text",
                     lambda c, u: embs[u], maps, META, count,
                     PreferenceEnricher(CAT), max_length=150)
    assert r["k_in_prompt"] < 5 and count(r["inp_str"]) <= 150 + 60
    print("ok  history is trimmed to fit the budget (k_in_prompt =", r["k_in_prompt"], ")")


def test_enricher_on_sop_example():
    revs = [dict(title="", text="I loved the science-fiction story.", rating=5,
                 item_title="", item_desc=""),
            dict(title="", text="The complex storyline was excellent.", rating=4,
                 item_title="", item_desc="")]
    p = PreferenceEnricher("Books").enrich(revs)
    assert "science fiction" in p.genres
    assert any("storyline" in a for a in p.liked), p.liked
    assert p.disliked == []
    neg = PreferenceEnricher("Movies_and_TV").enrich(
        [dict(title="Meh", text="The acting was not good and the plot was boring.",
              rating=2, item_title="", item_desc="")])
    assert "acting" in neg.disliked and any("plot" in a for a in neg.disliked), neg.disliked
    print("ok  enricher: SOP example + negation")


if __name__ == "__main__":
    test_shapes_and_layout()
    test_no_full_history_leak()
    test_variants()
    test_prompt_budget()
    test_enricher_on_sop_example()
    print("ALL TESTS PASSED")
