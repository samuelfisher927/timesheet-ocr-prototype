# src/layout/export_yolo_merged.py
# Merge one or more synth JSONL files (train/val/anything) into full YOLO TXT labels per page.

import os, json, argparse, pathlib, platform, shutil
from collections import defaultdict
from PIL import Image

NAMES = ["employee_name","in_am","out_am","lunch","in_pm","out_pm","total_hours","signature","date"]
NAME_TO_ID = {n:i for i,n in enumerate(NAMES)}

def class_from_field_name(name:str)->str:
    # "in_am_r4" -> "in_am"; keep header names like "date" as-is
    return name.split("_r")[0] if "_r" in name else name

def main(args):
    pages = defaultdict(list)  # page_image -> list[record]
    for js in args.jsonl:
        with open(js,"r",encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                cls = class_from_field_name(r.get("field_name",""))
                if cls in NAME_TO_ID:
                    pages[r["page_image"]].append(r)

    out_root = args.out
    imgs_out = os.path.join(out_root,"images")
    lbls_out = os.path.join(out_root,"labels")
    pathlib.Path(imgs_out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(lbls_out).mkdir(parents=True, exist_ok=True)

    def place(src,dst):
        if os.path.exists(dst): return
        if args.materialize=="copy" or (args.materialize=="auto" and platform.system().lower().startswith("win")):
            shutil.copy2(src,dst)
        elif args.materialize=="symlink":
            os.symlink(os.path.abspath(src), dst)
        else:
            try: os.symlink(os.path.abspath(src), dst)
            except Exception: shutil.copy2(src,dst)

    n_pages=0; n_anns=0; missing=0
    for page_path, items in pages.items():
        if not os.path.exists(page_path):
            missing += 1
            continue
        with Image.open(page_path) as im:
            W,H = im.size
        base = os.path.splitext(os.path.basename(page_path))[0]
        lbl_path = os.path.join(lbls_out, base + ".txt")
        # write all bboxes for this page (merged across jsonls)
        with open(lbl_path,"w",encoding="utf-8") as f:
            for r in items:
                cls = class_from_field_name(r["field_name"])
                cid = NAME_TO_ID[cls]
                x,y,w,h = r["bbox_xywh"]
                xc = (x + w/2)/W; yc = (y + h/2)/H; wn = w/W; hn = h/H
                f.write(f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
                n_anns += 1
        place(page_path, os.path.join(imgs_out, os.path.basename(page_path)))
        n_pages += 1

    print(f"[done] pages: {n_pages}, annotations: {n_anns}, missing images: {missing}")
    print(f"[out] images: {imgs_out}")
    print(f"[out] labels: {lbls_out}")
    print(f"[classes] {NAMES}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", required=True, help="One or more synth JSONLs to merge (train.jsonl, val.jsonl, etc.)")
    ap.add_argument("--out", required=True, help="Output root with images/ and labels/")
    ap.add_argument("--materialize", choices=["auto","copy","symlink"], default="auto")
    args = ap.parse_args()
    main(args)

