"""Generate + score reviews for one (category, K, variant) condition.

Mirrors the official model-eval.py (same model, prompts->vLLM fork, metrics),
with these additions for a controlled comparison:
  * reads the cold-start sets written by build_coldstart.py;
  * `--k full` reads the official data/dataset_test_<category> instead;
  * `--seed` gives every request its own fixed sampling seed;
  * `--limit` evaluates only the first N users (same N for every condition);
  * results are appended to results/coldstart_results.csv.

    python -m coldstart.eval_coldstart --category Books --k 2 --variant orig --mode infer
    python -m coldstart.eval_coldstart --category Books --k 2 --variant orig --mode eval
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import torch

SEP = "\n---------------------------------\n"


def load_condition(category: str, k: str, variant: str, out_dir: str, limit):
    if k == "full":     # official DEP test set (full history)
        from datasets import load_from_disk
        ds = load_from_disk(f"data/dataset_test_{category}")
        prompts, refs = list(ds["inp_str"]), list(ds["out_str"])
        embs = [torch.tensor(e) for e in ds["his_diff_emb"]]
    else:
        d = f"{out_dir}/{category}_K{k}_{variant}"
        rows = [json.loads(l) for l in open(f"{d}/data.jsonl", encoding="utf-8")]
        prompts, refs = [r["inp_str"] for r in rows], [r["out_str"] for r in rows]
        embs = [e.float() for e in torch.load(f"{d}/emb.pt")]
    if limit:
        prompts, refs, embs = prompts[:limit], refs[:limit], embs[:limit]
    return prompts, refs, embs


def pred_path(category, k, variant):
    return f"output/coldstart/{category}_K{k}_{variant}.txt"


def infer(args):
    from vllm import LLM, SamplingParams
    prompts, _, embs = load_condition(args.category, args.k, args.variant,
                                      args.out_dir, args.limit)
    params = SamplingParams(max_tokens=args.max_tokens, skip_special_tokens=True,
                            temperature=args.temperature, top_p=0.95, seed=args.seed)
    llm = LLM("SnowCharmQ/DEP-model", dtype=args.dtype,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              enforce_eager=True)
    outs = llm.generate(prompts, his_diff_embs=embs, sampling_params=params)
    preds = [o.outputs[0].text.strip() for o in outs]
    os.makedirs("output/coldstart", exist_ok=True)
    with open(pred_path(args.category, args.k, args.variant), "w", encoding="utf-8") as f:
        for p in preds:
            f.write(p + SEP)


def score(args):
    import numpy as np
    import evaluate
    _, refs, _ = load_condition(args.category, args.k, args.variant,
                                args.out_dir, args.limit)
    with open(pred_path(args.category, args.k, args.variant), encoding="utf-8") as f:
        preds = f.read().split(SEP)[:-1]
    preds = [p.strip() for p in preds]
    assert len(preds) == len(refs), (len(preds), len(refs))

    bleu = evaluate.load("sacrebleu").compute(predictions=preds, references=refs)
    rouge = evaluate.load("rouge").compute(predictions=preds, references=refs)
    meteor = evaluate.load("meteor").compute(predictions=preds, references=refs)
    res = dict(category=args.category, K=args.k, variant=args.variant, n=len(preds),
               rouge1=float(rouge["rouge1"]), rougeL=float(rouge["rougeL"]),
               meteor=float(meteor["meteor"]), bleu=float(bleu["score"]))
    if not args.no_bertscore:
        from bert_score import score as bert_score
        _, _, F1 = bert_score(preds, refs, model_type="allenai/led-base-16384",
                              lang="en", verbose=False)
        res["bertscore"] = float(F1.mean())
    print(res)

    os.makedirs("results", exist_ok=True)
    path = "results/coldstart_results.csv"
    new = not os.path.exists(path)
    fields = ["category", "K", "variant", "n", "rouge1", "rougeL",
              "meteor", "bleu", "bertscore"]
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: res.get(k, "") for k in fields})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True,
                    choices=["Books", "Movies_and_TV", "CDs_and_Vinyl"])
    ap.add_argument("--k", required=True, help="1 | 2 | 5 | full")
    ap.add_argument("--variant", default="orig",
                    choices=["orig", "enrich_text", "enrich_blend"])
    ap.add_argument("--mode", required=True, choices=["infer", "eval"])
    ap.add_argument("--out_dir", default="data/coldstart")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu_mem", type=float, default=0.9)
    ap.add_argument("--max_model_len", type=int, default=6144)
    ap.add_argument("--no_bertscore", action="store_true")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    (infer if args.mode == "infer" else score)(args)


if __name__ == "__main__":
    main()
