# src/layout/export_yolo.py
# Convert your synth JSONL (train/val) into YOLO TXT labels (page-level).
# One .txt per page image; bboxes are classed by the table column name.
#
# Usage:
#   python -m src.layout.export_yolo \
#     --jsonl exports/datasets/daily_timesheet_synth/v1/train.jsonl \
#     --out   exports/yolo/v1/train
#   python -m src.layout.export_yolo \
#     --jsonl exports/datasets/daily_timesheet_synth/v1/val.jsonl \
#     --out   exports/yolo/v1/val

import os, json, argparse, pathlib
from collections import defaultdict
from PIL import Image

# Class order MUST match your pipeline ordering
NAMES = [
    "employee_name", "in_am", "out_am", "lunch",
    "in_pm", "out_pm", "total_hours", "signature"
]
NAME_TO_ID = {n:i for i,n in enumerate(NAMES)}

def class_from_field_name(field_name: str) -> str:
    # "in_am_r4" -> "in_am"
    return field_name.split("_r")[0] if "_r" in field_name else field_name

def main(args):
    rows = [json.loads(l) for l in open(args.jsonl, "r", encoding="utf-8")]

    # group by page
    by_img = defaultdict(list)
    for r in rows:
        cls = class_from_field_name(r.get("field_name",""))
        if cls not in NAME_TO_ID:
            continue  # skip header/company/supervisor/date crops
        by_img[r["page_image"]].append(r)

    out_root = args.out
    imgs_out = os.path.join(out_root, "images")
    lbls_out = os.path.join(out_root, "labels")
    pathlib.Path(imgs_out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(lbls_out).mkdir(parents=True, exist_ok=True)

    # copy/relatively link images? Ultralytics only needs paths; we can just reference original paths.
    # For portability, we’ll symlink if possible, else skip copying.
    def link_or_skip(src, dst):
        if os.path.exists(dst): return
        try:
            os.symlink(os.path.abspath(src), dst)
        except Exception:
            # fallback: don’t copy to keep repo small; trainer can still read absolute paths
            pass

    for page_path, items in by_img.items():
        # image dims
        with Image.open(page_path) as im:
            W, H = im.size

        # write label file
        base = os.path.splitext(os.path.basename(page_path))[0]
        lbl_path = os.path.join(lbls_out, base + ".txt")
        with open(lbl_path, "w", encoding="utf-8") as f:
            for r in items:
                cls = class_from_field_name(r["field_name"])
                cid = NAME_TO_ID[cls]
                x,y,w,h = r["bbox_xywh"]
                # convert to YOLO (xc,yc,w,h) normalized
                xc = (x + w/2) / W
                yc = (y + h/2) / H
                wn = w / W
                hn = h / H
                f.write(f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

        # (optional) symlink image into images/ so data.yaml can point to a single folder
        link_or_skip(page_path, os.path.join(imgs_out, os.path.basename(page_path)))

    print(f"[done] YOLO TXT labels -> {lbls_out}\n[info] Images folder (symlinks if possible): {imgs_out}\n[info] classes: {NAMES}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args)
