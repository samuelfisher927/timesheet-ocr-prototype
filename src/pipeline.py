# src/pipeline.py
from __future__ import annotations
import os, json, argparse, pathlib, cv2
from typing import List, Tuple

# --- import your existing modules ---
from src.layout.detect import run_yolo            # you already have this wrapper
from src.layout.grid_snap import snap_to_grid     # aligns dets to (row, col)
# from src.layout.register import register_and_crop  # optional Phase-1
from src.ocr.normalizers import sanitize_time, sanitize_amount
# TODO: replace with your actual TrOCR call
# from src.ocr.trocr_time_infer import infer_batch

# Geometry must match your synth generator
PAGE_W, PAGE_H = 1700, 2200
MARGIN_L, MARGIN_T = 80, 180
HEADER_GAP = 90
HEADER_ROW_H = 120
ROWS_DATA = 12
COLS = [
    ("employee_name","text",   260),
    ("in_am","time",           170),
    ("out_am","time",          170),
    ("lunch","time",           140),
    ("in_pm","time",           170),
    ("out_pm","time",          170),
    ("total_hours","amount",   170),
    ("signature","text",       210),
]
COL_ORDER = [c[0] for c in COLS]

def run_pipeline(image_path: str, yolo_weights: str, out_dir: str,
                 conf: float = 0.25, imgsz: int = 1280):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    page = cv2.imread(image_path, cv2.IMREAD_COLOR)
    assert page is not None, f"Failed to read image: {image_path}"

    # 1) Detect
    dets = run_yolo(yolo_weights, image_path, conf=conf, imgsz=imgsz)
    # dets: [{'cls':'in_am','score':0.99,'bbox':[x,y,w,h]}, ...]

    # 2) Grid-snap (map to (row, col))
    grid = snap_to_grid(
    detections=dets,
    table_top=MARGIN_T + HEADER_GAP,   # top of first header band
    row_h=HEADER_ROW_H,
    n_rows_total=ROWS_DATA,
    x0=MARGIN_L,
    col_widths=[w for _,_,w in COLS],
    class_names=[c[0] for c in COLS],
    header_rows=2,                     # ← skip typed label rows
    )

    # 3) Crop + OCR route (stubbed here; integrate your TrOCR call)
    results = []
    for r in range(ROWS_DATA):
        for c, (key, ftype, _) in enumerate(COLS):
            det = grid.get((r, c))
            if not det:
                results.append({"row":r+1,"col":c+1,"field":key,"text":"","normalized":"","conf":0.0,"reason":"no_det"})
                continue
            x,y,w,h = map(int, det["bbox"])
            crop = page[y:y+h, x:x+w]

            # TODO: batch your OCR for speed; here we keep a stub
            text = ""  # replace with TrOCR output for this crop
            confd = float(det.get("score", 0.0))

            # normalize by field type
            if ftype == "time":
                norm = sanitize_time(text) or ""
            elif ftype == "amount":
                norm = sanitize_amount(text) or ""
            else:
                norm = text

            results.append({
                "row": r+1, "col": c+1, "field": key,
                "raw": text, "normalized": norm, "conf_det": confd,
                "bbox_xywh": [x,y,w,h]
            })

    out_json = os.path.join(out_dir, pathlib.Path(image_path).stem + ".json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"image": image_path, "results": results}, f, indent=2)
    print("[done]", out_json)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default="models/yolo_timesheet_best.pt")
    ap.add_argument("--out", default="outputs/demo")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()
    run_pipeline(args.image, args.weights, args.out, args.conf, args.imgsz)
