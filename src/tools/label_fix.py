# src/tools/label_fix.py
from __future__ import annotations
import csv, time, os, argparse
def main(csv_path, crop_path, label, field_type="time", reviewer="sam"):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    row = {"crop_path": crop_path, "field_type": field_type,
           "pred_raw": "", "label": label, "reviewer": reviewer, "ts": int(time.time())}
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys()); w.writerow(row)
    print("Appended:", row)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="exports/labels_time.csv")
    ap.add_argument("--crop_path", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--field_type", default="time")
    ap.add_argument("--reviewer", default="sam")
    args = ap.parse_args()
    main(args.csv, args.crop_path, args.label, args.field_type, args.reviewer)
