# ARCHIVED on 2025-10-13 — superseded by synth + YOLO flow.
# Keep for reference only. Not used in the current pipeline.

# src/tools/labels_compact.py
import csv, sys
inp = sys.argv[1] if len(sys.argv)>1 else "exports/labels_time.csv"
out = sys.argv[2] if len(sys.argv)>2 else "exports/labels_time_compacted.csv"
latest={}
with open(inp, newline="", encoding="utf-8") as f:
    r=csv.DictReader(f)
    for row in r: latest[row["crop_path"]]=row
with open(out, "w", newline="", encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=["crop_path","field_type","pred_raw","label","reviewer","ts"])
    w.writeheader(); [w.writerow(latest[k]) for k in sorted(latest)]
print(f"Wrote {len(latest)} rows to {out}")
