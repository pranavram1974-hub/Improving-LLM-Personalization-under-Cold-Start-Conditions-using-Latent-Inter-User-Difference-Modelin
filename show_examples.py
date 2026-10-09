"""Print qualitative side-by-side examples for the same users across variants.

    python -m coldstart.show_examples --category Books --k 2 --n 5
"""
import argparse
import json

from coldstart.eval_coldstart import SEP, pred_path

ap = argparse.ArgumentParser()
ap.add_argument("--category", required=True)
ap.add_argument("--k", required=True)
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--out_dir", default="data/coldstart")
ap.add_argument("--variants", nargs="+", default=["orig", "enrich_text", "enrich_blend"])
a = ap.parse_args()

preds, rows = {}, None
for v in a.variants:
    try:
        preds[v] = open(pred_path(a.category, a.k, v), encoding="utf-8").read().split(SEP)[:-1]
        if rows is None:
            rows = [json.loads(l) for l in open(
                f"{a.out_dir}/{a.category}_K{a.k}_{v}/data.jsonl", encoding="utf-8")]
    except FileNotFoundError:
        print(f"(no predictions for {v}, skipping)")

profiles = []
if "enrich_text" in preds:
    profiles = [json.loads(l)["profile_text"] for l in open(
        f"{a.out_dir}/{a.category}_K{a.k}_enrich_text/data.jsonl", encoding="utf-8")]

for i in range(min(a.n, len(rows or []))):
    r = rows[i]
    print("=" * 80)
    print(f"user {r['user_id']}  |  reviews available: {r['k_available']}  |  "
          f"with peer difference: {r['n_with_peers']}")
    if profiles:
        print(profiles[i])
    print("REFERENCE :", r["out_str"][:400].replace("\n", " "))
    for v, p in preds.items():
        print(f"{v:13s}:", p[i][:400].replace("\n", " "))
