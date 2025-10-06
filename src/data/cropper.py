# src/data/cropper.py
from __future__ import annotations
import os, json
from typing import Dict, Tuple
from PIL import Image

from src.detect.stub_detector import StubDetector

def pad_box(b, pad, W, H):
    x1,y1,x2,y2 = b
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(W, x2 + pad),
        min(H, y2 + pad),
    )

def crop_image_fields(images_dir: str, labels_dir: str, out_dir: str, pad: int = 6) -> None:
    os.makedirs(out_dir, exist_ok=True)
    det = StubDetector(labels_dir)
    image_ids = det.list_image_ids(labels_dir)

    manifest = []  # rows: {image_id, field_name, bucket, crop_path}

    # very simple bucketing rule (extend later)
    def bucket_for(field_name: str) -> str:
        if any(k in field_name for k in ["clock_in", "clock_out"]):
            return "time"
        if "lunch" in field_name:
            return "lunch"
        if "total_hours" in field_name:
            return "total_hours"
        if "employee_name" in field_name:
            return "name"
        if "signature" in field_name:
            return "signature"
        return "other"

    for image_id in image_ids:
        img_path = os.path.join(images_dir, f"{image_id}.png")
        if not os.path.exists(img_path):
            print(f"[warn] missing image: {img_path}")
            continue
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        fields = det.detect(image_id)
        out_dir_id = os.path.join(out_dir, image_id)
        os.makedirs(out_dir_id, exist_ok=True)

        for fname, meta in fields.items():
            bbox = meta["bbox"]
            x1,y1,x2,y2 = pad_box(bbox, pad, W, H)
            crop = img.crop((x1,y1,x2,y2))

            bkt = bucket_for(fname)
            # keep folders by bucket for convenience
            out_bucket_dir = os.path.join(out_dir_id, bkt)
            os.makedirs(out_bucket_dir, exist_ok=True)
            crop_path = os.path.join(out_bucket_dir, f"{fname}.png")
            crop.save(crop_path)

            manifest.append({
                "image_id": image_id,
                "field_name": fname,
                "bucket": bkt,
                "crop_path": crop_path.replace("\\", "/"),
                "bbox": [int(x) for x in (x1,y1,x2,y2)],
            })

    # write a simple manifest jsonl
    man_path = os.path.join(out_dir, "manifest.jsonl")
    with open(man_path, "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    print(f"[done] crops saved under: {out_dir}\n[done] manifest: {man_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pad", type=int, default=6)
    args = ap.parse_args()
    crop_image_fields(args.images_dir, args.labels_dir, args.out_dir, pad=args.pad)
