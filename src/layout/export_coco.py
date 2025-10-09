# src/layout/export_coco.py
# Build COCO detection annotations from your synth JSONL (train/val).
# Each JSONL row: {"page_image": ".../pages/xxx.jpg", "bbox_xywh": [x,y,w,h], "field_type": "...", "field_name": "..."}
# Run:
#   python -m src.layout.export_coco --jsonl exports/datasets/daily_timesheet_synth/v1/train.jsonl --out exports/coco/train.json

import os, json, argparse, hashlib
from collections import defaultdict

CATEGORIES = [
    {"id": 1, "name": "employee_name"},
    {"id": 2, "name": "in_am"},
    {"id": 3, "name": "out_am"},
    {"id": 4, "name": "lunch"},
    {"id": 5, "name": "in_pm"},
    {"id": 6, "name": "out_pm"},
    {"id": 7, "name": "total_hours"},
    {"id": 8, "name": "signature"},
]
NAME_TO_ID = {c["name"]: c["id"] for c in CATEGORIES}

def class_from_field_name(field_name: str) -> str:
    # field_name like "in_am_r4" -> "in_am"
    return field_name.split("_r")[0] if "_r" in field_name else field_name

def stable_image_id(path: str) -> int:
    h = hashlib.md5(path.encode("utf-8")).hexdigest()
    return int(h[:12], 16)

def main(args):
    rows = [json.loads(l) for l in open(args.jsonl, "r", encoding="utf-8")]
    # group by page_image
    by_img = defaultdict(list)
    for r in rows:
        # skip header crops if any (they aren't part of the table classes)
        fname = r.get("field_name","")
        cls = class_from_field_name(fname)
        if cls not in NAME_TO_ID:
            continue
        by_img[r["page_image"]].append(r)

    images, annotations = [], []
    ann_id = 1
    for page_path, items in by_img.items():
        img_id = stable_image_id(page_path)
        # If you want actual width/height, you can read it; else set None/0
        images.append({"id": img_id, "file_name": page_path, "width": 0, "height": 0})
        for r in items:
            cls = class_from_field_name(r["field_name"])
            cat_id = NAME_TO_ID[cls]
            x,y,w,h = r["bbox_xywh"]
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x, y, w, h],
                "area": float(w*h),
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(coco, open(args.out, "w"), indent=2)
    print(f"[done] COCO -> {args.out}  images={len(images)} anns={len(annotations)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args)
