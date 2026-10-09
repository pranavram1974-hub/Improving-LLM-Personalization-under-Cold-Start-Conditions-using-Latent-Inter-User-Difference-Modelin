"""Turn results/coldstart_results.csv into the SOP's Table I (+ relative change).

    python -m coldstart.make_table [--csv results/coldstart_results.csv]

Rows are filled ONLY with measured numbers; a missing condition prints "-".
"""
import argparse
import csv
from collections import defaultdict

ORDER = [("full", "orig"), ("5", "orig"), ("5", "enrich_text"), ("5", "enrich_blend"),
         ("2", "orig"), ("2", "enrich_text"), ("2", "enrich_blend"),
         ("1", "orig"), ("1", "enrich_text"), ("1", "enrich_blend")]
NAMES = {"orig": "Original DEP", "enrich_text": "Modified DEP (text)",
         "enrich_blend": "Modified DEP (text+blend)"}
METRICS = ["meteor", "bleu", "rouge1", "rougeL", "bertscore"]


def load(path):
    cells = defaultdict(lambda: defaultdict(list))   # (K,variant) -> metric -> [per-category]
    for r in csv.DictReader(open(path, encoding="utf-8")):
        for m in METRICS:
            if r.get(m) not in (None, ""):
                cells[(r["K"], r["variant"])][m].append(float(r[m]))
    return {k: {m: sum(v) / len(v) for m, v in d.items()} for k, d in cells.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/coldstart_results.csv")
    a = ap.parse_args()
    t = load(a.csv)
    print("Macro-average over the categories present in the CSV\n")
    print(f"{'Method':28s}{'History':>10s}" + "".join(f"{m:>11s}" for m in METRICS)
          + f"{'dMETEOR vs orig':>18s}")
    for K, v in ORDER:
        row = t.get((K, v))
        hist = "Full" if K == "full" else f"{K} review" + ("s" if K != "1" else "")
        cells = "".join(f"{row[m]:11.4f}" if row and m in row else f"{'-':>11s}"
                        for m in METRICS)
        rel = "-"
        base = t.get((K, "orig"))
        if v != "orig" and row and base and "meteor" in row and "meteor" in base:
            rel = f"{100 * (row['meteor'] - base['meteor']) / base['meteor']:+.2f}%"
        print(f"{NAMES[v]:28s}{hist:>10s}{cells}{rel:>18s}")


if __name__ == "__main__":
    main()
