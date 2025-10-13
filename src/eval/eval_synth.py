# src/eval/eval_synth.py
from __future__ import annotations
import json, argparse
from collections import defaultdict
from src.ocr.normalizers import sanitize_time, sanitize_amount, sanitize_date

FT = {
    "in_am":"time","out_am":"time","in_pm":"time","out_pm":"time","lunch":"time",
    "total_hours":"amount","date":"date","employee_name":"text","signature":"signature"
}

def normalize(col: str, v: str) -> str:
    if v is None: return ""
    v = (v or "").strip()
    t = FT.get(col, "text")
    if t == "time":   return sanitize_time(v)   or v
    if t == "amount": return sanitize_amount(v) or v
    if t == "date":   return sanitize_date(v)   or v
    return v

def load_jsonl(p: str):
    out = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            out[j["row"]] = j
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_rows_jsonl", required=True)
    ap.add_argument("--truth_rows_jsonl", required=True)
    args = ap.parse_args()

    pred = load_jsonl(args.pred_rows_jsonl)
    truth = load_jsonl(args.truth_rows_jsonl)

    cols = sorted({k for r in truth.values() for k in r.keys() if k not in ("row")})
    correct = defaultdict(int); total = defaultdict(int)

    for r in truth:
        trow, prow = truth[r], pred.get(r, {})
        for c in cols:
            if c == "row": continue
            if c not in trow: continue
            tv = normalize(c, trow.get(c, ""))
            pv = normalize(c, prow.get(c, ""))
            total[c] += 1
            if tv == pv:
                correct[c] += 1

    print("\nPer-column exact-match (after sanitize):")
    for c in cols:
        if total[c]:
            acc = 100.0 * correct[c] / total[c]
            print(f"  {c:15s}: {acc:5.1f}%  ({correct[c]}/{total[c]})")

    all_ok = sum(correct.values()); all_tot = sum(total.values())
    if all_tot:
        print(f"\nOverall: {100.0*all_ok/all_tot:5.1f}%  ({all_ok}/{all_tot})")

if __name__ == "__main__":
    main()
