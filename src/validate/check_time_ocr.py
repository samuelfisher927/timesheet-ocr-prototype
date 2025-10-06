# src/validate/check_time_ocr.py
from __future__ import annotations
import os, json, argparse
from collections import defaultdict, Counter
from src.validate.validators import normalize_hhmm

def load_labels(labels_dir: str):
    labels = {}
    for fn in os.listdir(labels_dir):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(labels_dir, fn), "r", encoding="utf-8") as f:
            js = json.load(f)
        image_id = js["image_id"]
        labels[image_id] = {k: v.get("text","") for k, v in js["fields"].items()}
    return labels

def parse_crop_path(p: str):
    # e.g. data/clockin_synth/crops/clock_0000/time/row1_clock_in_am.png
    parts = p.replace("\\","/").split("/")
    image_id = parts[-3]  # clock_0000
    fname = os.path.splitext(parts[-1])[0]  # row1_clock_in_am
    return image_id, fname

def main(labels_dir: str, ocr_jsonl: str):
    # load gt labels
    gt = load_labels(labels_dir)

    # read OCR outputs
    preds = defaultdict(dict)  # preds[image_id][field_name] = text
    with open(ocr_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            img_id, fname = parse_crop_path(row["crop_path"])
            preds[img_id][fname] = row["pred_text"]

    # evaluate only time fields present in predictions
    total = 0
    exact = 0
    bad_pred = 0
    mismatches = []
    norm_dist = Counter()

    for img_id, fields in preds.items():
        for fname, pred in fields.items():
            if not any(k in fname for k in ["clock_in", "clock_out"]):
                continue
            total += 1
            gt_text = gt[img_id].get(fname, "")

            n_pred = normalize_hhmm(pred)
            n_gt   = normalize_hhmm(gt_text)

            if n_pred is None:
                bad_pred += 1
                mismatches.append((img_id, fname, gt_text, pred, "INVALID_PRED"))
                continue

            if n_gt is None:
                mismatches.append((img_id, fname, gt_text, pred, "INVALID_GT"))
                continue

            if n_pred == n_gt:
                exact += 1
            else:
                mismatches.append((img_id, fname, gt_text, pred, "MISMATCH"))
                # simple stats on how far off (string-wise)
                norm_dist[(n_gt, n_pred)] += 1

    acc = exact / total if total else 0.0
    print(f"[report] time fields evaluated: {total}")
    print(f"[report] exact matches:        {exact} ({acc:.2%})")
    print(f"[report] invalid predictions:  {bad_pred}")
    if mismatches:
        print("\n[examples] first 10 mismatches:")
        for m in mismatches[:10]:
            img, field, gt_t, pr_t, tag = m
            print(f"  {tag}: {img}:{field}  GT={gt_t!r}  PRED={pr_t!r}")

    if norm_dist:
        print("\n[diagnostics] common (GT -> PRED) confusions (top 10):")
        for (gtv, pv), c in norm_dist.most_common(10):
            print(f"  {gtv} -> {pv}: {c}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--ocr_jsonl", required=True)
    args = ap.parse_args()
    main(args.labels_dir, args.ocr_jsonl)
