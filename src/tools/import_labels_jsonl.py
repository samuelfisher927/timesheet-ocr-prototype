# src/tools/import_labels_jsonl.py
from __future__ import annotations
import os, json, csv, random, argparse
from typing import Dict, Any, Iterable, Tuple, List, Optional
from PIL import Image
import numpy as np

from src.ocr.normalizers import sanitize_time, sanitize_amount

# Optional: no-op preprocess if cv2 not present
try:
    from src.ocr.preprocess import preprocess_image
except Exception:
    def preprocess_image(arr, return_debug: bool=False):
        return arr

SENTINELS = {"", "<EMPTY>", "<ILLEGIBLE>", "<SIGNATURE>", "NA", "N/A"}

# Keys we want (tight, from your probe)
import re
TIME_PATTERNS   = (re.compile(r'(?:^|_)clock_(?:in|out)_(?:am|pm)$'),)
AMOUNT_PATTERNS = (re.compile(r'(?:^|_)total_hours$'),)

def guess_type(key: str) -> Optional[str]:
    k = key.lower()
    if any(p.search(k) for p in TIME_PATTERNS):   return "time"
    if any(p.search(k) for p in AMOUNT_PATTERNS): return "amount"
    return None

def _xywh_to_xyxy(b): x,y,w,h=b; return int(x),int(y),int(x+w),int(y+h)
def _xyxy_to_xyxy(b): x1,y1,x2,y2=b; return int(x1),int(y1),int(x2),int(y2)

def parse_bbox(bbox: Any) -> Optional[Tuple[int,int,int,int]]:
    """Accept dict/list: (x,y,w,h) or (x1,y1,x2,y2) or polygon points."""
    if bbox is None: return None
    if isinstance(bbox, dict):
        if all(k in bbox for k in ("x","y","w","h")):  return _xywh_to_xyxy((bbox["x"],bbox["y"],bbox["w"],bbox["h"]))
        if all(k in bbox for k in ("x1","y1","x2","y2")): return _xyxy_to_xyxy((bbox["x1"],bbox["y1"],bbox["x2"],bbox["y2"]))
        if all(k in bbox for k in ("left","top","right","bottom")): return _xyxy_to_xyxy((bbox["left"],bbox["top"],bbox["right"],bbox["bottom"]))
        pts = bbox.get("points") or bbox.get("polygon")
        if isinstance(pts,(list,tuple)) and len(pts)>=4:
            flat=[]
            for p in pts:
                if isinstance(p,(list,tuple)) and len(p)==2: flat+=p
            if len(flat)>=8:
                xs=flat[0::2]; ys=flat[1::2]
                return int(min(xs)),int(min(ys)),int(max(xs)),int(max(ys))
    if isinstance(bbox,(list,tuple)):
        b=list(bbox)
        if len(b)==4:
            # Heuristic: xyxy if 3rd>1st and 4th>2nd else xywh
            return _xyxy_to_xyxy(b) if b[2]>b[0] and b[3]>b[1] else _xywh_to_xyxy((b[0],b[1],b[2],b[3]))
        if len(b)>=8:
            xs=b[0::2]; ys=b[1::2]
            return int(min(xs)),int(min(ys)),int(max(xs)),int(max(ys))
    return None

def clamp_box(x1,y1,x2,y2,W,H,pad=2):
    x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(W,x2+pad); y2=min(H,y2+pad)
    if x2<=x1 or y2<=y1: return 0,0,0,0
    return x1,y1,x2,y2

def norm_text(txt: str, ftype: str) -> Optional[str]:
    t=(txt or "").strip()
    if t in SENTINELS: return None
    if ftype=="time":   return sanitize_time(t)
    if ftype=="amount": return sanitize_amount(t)
    return t or None

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_per_image_labels(labels_dir: str, image_id: str) -> Dict[str,Any]:
    """Load bboxes (and possibly texts) for one sheet."""
    p = os.path.join(labels_dir, f"{image_id}.json")
    if not os.path.exists(p): return {}
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # Expect a dict with 'fields' dict inside; fall back to flat
    fields = obj.get("fields") or obj
    out={}
    if isinstance(fields, dict):
        for k,v in fields.items():
            if isinstance(v, dict):
                out[k] = {"bbox": v.get("bbox") or v.get("box") or v.get("rect") or v.get("bounds") or v.get("poly") or v.get("points"),
                          "text": v.get("text")}
    return out

def import_labels(labels_jsonl: str, labels_dir: str, images_root: str, out_root: str,
                  include_types: List[str], val_ratio: float=0.1, debug: bool=False):
    images_dir = os.path.join(out_root, "images"); ensure_dir(images_dir)
    items: List[Dict[str,str]] = []
    c = {k:0 for k in ["lines","img_ok","fields_total","bbox_found","typed","text_ok","crops_saved"]}

    with open(labels_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            c["lines"]+=1
            obj=json.loads(line)

            image_id = obj.get("image_id") or obj.get("id")
            if not image_id: 
                if debug: print("[skip] no image_id")
                continue
            img_path = os.path.join(images_root, f"{image_id}.png")
            if not os.path.exists(img_path):
                if debug: print("[skip] missing image:", img_path)
                continue
            c["img_ok"]+=1

            # texts may be in this line; bboxes in per-image json
            line_fields = obj.get("fields") or {}
            per_image = load_per_image_labels(labels_dir, image_id)

            # load sheet once
            sheet = Image.open(img_path).convert("RGB")
            W,H = sheet.size

            for key, maybe in line_fields.items():
                c["fields_total"]+=1
                ftype = guess_type(key)
                if ftype is None or ftype not in include_types:
                    continue
                text = ""
                if isinstance(maybe, dict):
                    text = (maybe.get("text") or "").strip()
                else:
                    text = str(maybe).strip()

                # get bbox from per-image labels
                bbox_info = per_image.get(key, {})
                bbox = parse_bbox(bbox_info.get("bbox"))
                if not bbox:
                    if debug: print("[skip] no bbox for", image_id, key)
                    continue
                c["bbox_found"]+=1

                clean = norm_text(text, ftype)
                if clean is None:
                    # skip empties/invalids
                    if debug: print("[skip] text rejected:", key, repr(text))
                    continue
                c["text_ok"]+=1

                x1,y1,x2,y2 = clamp_box(*bbox, W,H, pad=2)
                if x2<=x1 or y2<=y1:
                    if debug: print("[skip] degenerate box for", key, bbox)
                    continue

                crop = np.array(sheet)[y1:y2, x1:x2, :]
                try:
                    crop = preprocess_image(crop, return_debug=False)
                except Exception:
                    pass

                safe_key = key.lower().replace(" ", "_")
                dst_name = f"{image_id}__{safe_key}__x{x1}_y{y1}_x{x2}_y{y2}.png"
                dst_abs  = os.path.join(images_dir, dst_name)
                Image.fromarray(crop).save(dst_abs)

                items.append({"image": f"images/{dst_name}", "text": clean, "ftype": ftype})
                c["crops_saved"]+=1

    if not items:
        print("[warn] no items imported.")
        if debug: print("counters:", c)
        return

    random.seed(13); random.shuffle(items)
    n_val = max(1, int(len(items)*val_ratio))
    val, train = items[:n_val], items[n_val:]

    ensure_dir(out_root)
    with open(os.path.join(out_root, "train.jsonl"), "w", encoding="utf-8") as f:
        for it in train: f.write(json.dumps({"image": it["image"], "text": it["text"]}) + "\n")
    with open(os.path.join(out_root, "val.jsonl"), "w", encoding="utf-8") as f:
        for it in val:   f.write(json.dumps({"image": it["image"], "text": it["text"]}) + "\n")

    n_time = sum(1 for it in items if it["ftype"]=="time")
    n_amt  = sum(1 for it in items if it["ftype"]=="amount")
    print(f"[done] imported {len(items)} items → {len(train)} train / {len(val)} val at {out_root}")
    print(f"        time={n_time}, amount={n_amt}")
    if debug: print("counters:", c)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_jsonl", required=True)
    ap.add_argument("--labels_dir", required=True, help="Folder with per-image JSONs that include bboxes")
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--include_types", nargs="+", default=["time","amount"])
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    import_labels(args.labels_jsonl, args.labels_dir, args.images_root, args.out_root,
                  args.include_types, args.val_ratio, debug=args.debug)
    