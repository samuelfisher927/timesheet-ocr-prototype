# ARCHIVED on 2025-10-13 — superseded by synth + YOLO flow.
# Keep for reference only. Not used in the current pipeline.

from __future__ import annotations
import os, csv, json, hashlib, shutil, random
from typing import Dict, List
from src.ocr.normalizers import sanitize_time

OUT_ROOT = "exports/datasets/time_gold/v1"

def hash_name(path: str) -> str:
    h = hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:16]
    base = os.path.basename(path).replace("\\","/").split("/")[-1]
    stem, ext = os.path.splitext(base)
    return f"{stem}__{h}{ext or '.png'}"

def load_labels(csv_path: str) -> Dict[str, str]:
    """Return latest label per crop_path (CSV is append-only)."""
    latest: Dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            crop = row["crop_path"]
            label = (row.get("label") or "").strip()
            if not crop or not label:
                continue
            latest[crop] = label  # keep last occurrence
    return latest

def main(labels_csv: str):
    labels = load_labels(labels_csv)
    if not labels:
        print("[warn] no labels found")
        return

    # Normalize labels to canonical HH:MM for training targets
    items: List[Dict] = []
    for crop, lab in labels.items():
        clean = sanitize_time(lab) or lab.strip()
        items.append({"crop_path": crop, "text": clean})

    # Shuffle + split
    random.seed(13)
    random.shuffle(items)
    n = len(items)
    n_val = max(1, int(0.1 * n))
    train, val = items[n_val:], items[:n_val]

    img_dir = os.path.join(OUT_ROOT, "images")
    os.makedirs(img_dir, exist_ok=True)

    def copy_and_manifest(split_items: List[Dict], out_jsonl: str):
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for ex in split_items:
                src = ex["crop_path"]
                if not os.path.exists(src):
                    # try relative to repo root if saved with different separators
                    cand = src.replace("\\", "/")
                    if not os.path.exists(cand):
                        print(f"[skip] missing: {src}")
                        continue
                    src = cand
                dst_name = hash_name(src)
                dst_rel = f"images/{dst_name}"
                dst_abs = os.path.join(OUT_ROOT, dst_rel)
                os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
                shutil.copy2(src, dst_abs)
                f.write(json.dumps({"image": dst_rel, "text": ex["text"]}) + "\n")

    os.makedirs(OUT_ROOT, exist_ok=True)
    copy_and_manifest(train, os.path.join(OUT_ROOT, "train.jsonl"))
    copy_and_manifest(val,   os.path.join(OUT_ROOT, "val.jsonl"))

    print(f"[done] wrote {len(train)} train and {len(val)} val examples to {OUT_ROOT}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", default="exports/labels_time.csv")
    args = ap.parse_args()
    main(args.labels_csv)
