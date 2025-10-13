# src/layout/debug_snap.py
from __future__ import annotations
import argparse, pathlib, json
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

from src.layout.grid_snap import snap_to_grid

# --- Canonical layout (match daily_timesheet_synth) ---
PAGE_W, PAGE_H = 1700, 2200
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 80, 80, 180, 120
HEADER_GAP    = 90              # space under title/headers before the table band
HEADER_ROW_H  = 120             # height of each table row
ROWS_DATA     = 12              # number of DATA rows only (no typed header rows)

COLS = [
    ("employee_name", "text",   260),
    ("in_am",         "time",   170),
    ("out_am",        "time",   170),
    ("lunch",         "time",   140),
    ("in_pm",         "time",   170),
    ("out_pm",        "time",   170),
    ("total_hours",   "amount", 170),
    ("signature",     "text",   210),
]
COL_ORDER  = [c[0] for c in COLS]
COL_WIDTHS = [c[2] for c in COLS]

# --------------------------------------------------------------------

def _run_yolo(weights: str, image_path: str, conf: float = 0.25, imgsz: int = 1280):
    """Returns [{'cls': <name>, 'score': float, 'bbox': [x,y,w,h]}, ...]."""
    try:
        # Prefer your wrapper if present
        from src.layout.detect import run_yolo
        return run_yolo(weights, image_path, conf=conf, imgsz=imgsz)
    except Exception:
        from ultralytics import YOLO
        model = YOLO(weights)
        preds = model.predict(source=image_path, imgsz=imgsz, conf=conf, verbose=False)
        out = []
        for p in preds:
            names = p.names
            for b in p.boxes:
                cls_id = int(b.cls.item())
                confd  = float(b.conf.item())
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                out.append({
                    "cls": names.get(cls_id, str(cls_id)),
                    "score": confd,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # xywh
                })
        return out

# Simple color map (BGR)
_PALETTE = [
    (60,180,255),(80,180,60),(200,160,60),(60,60,220),
    (180,90,200),(100,100,100),(30,210,210),(0,160,255),(40,40,40)
]
def _color_for(name: str) -> tuple:
    i = (COL_ORDER + ["date"]).index(name) if name in (COL_ORDER + ["date"]) else 0
    return _PALETTE[i % len(_PALETTE)]

def _put_text(img, txt, org, color=(0,0,0), scale=0.6, thickness=1):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness+2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def main(args):
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    assert img is not None, f"cannot read {args.image}"
    H, W = img.shape[:2]

    # 1) Detect
    dets = _run_yolo(args.weights, args.image, conf=args.conf, imgsz=args.imgsz)

    # 2) Visualize raw detections
    vis = img.copy()
    for d in dets:
        x, y, w, h = map(int, d["bbox"])
        cls = str(d["cls"])
        sc  = float(d.get("score", 0.0))
        col = _color_for(cls)
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
        _put_text(vis, f"{cls} {sc:.2f}", (x, max(14, y - 6)), col, 0.6, 1)

    # 3) Draw expected grid anchors (DATA rows only)
    table_left = MARGIN_L
    table_top  = MARGIN_T + HEADER_GAP                   # top of the FIRST header band
    data_top   = table_top + args.header_rows * HEADER_ROW_H  # top of FIRST DATA row

    col_edges = [table_left]
    for w in COL_WIDTHS:
        col_edges.append(col_edges[-1] + w)
    col_right = col_edges[-1]

    # vertical center guides
    col_centers = [int((col_edges[i] + col_edges[i+1]) / 2) for i in range(len(COL_WIDTHS))]
    for cx in col_centers:
        cv2.line(vis, (cx, data_top), (cx, data_top + ROWS_DATA * HEADER_ROW_H), (180,180,180), 1, cv2.LINE_AA)

    # horizontal data row lines
    for r in range(ROWS_DATA + 1):
        y = data_top + r * HEADER_ROW_H
        cv2.line(vis, (table_left, y), (col_right, y), (180,180,180), 1, cv2.LINE_AA)
        if r < ROWS_DATA:
            _put_text(vis, f"r{r+1}", (table_left - 40, y + 16), (0,0,0), 0.5, 1)

    # 4) Snap
    grid = snap_to_grid(
        detections=dets,
        table_top=table_top,                 # FIRST header band top (snapper will skip header_rows)
        row_h=HEADER_ROW_H,
        n_rows_total=ROWS_DATA,              # DATA rows count
        x0=table_left,
        col_widths=COL_WIDTHS,
        class_names=COL_ORDER,
        header_rows=args.header_rows,
    )

    # 5) Row histogram for quick sanity
    rows = {r: [] for r in range(ROWS_DATA)}
    for (r, c), det in grid.items():
        if det:
            rows[r].append(det)
    print("\nRow histogram (count / yc min..max):")
    for r in range(ROWS_DATA):
        ycs = []
        for d in rows[r]:
            x, y, w, h = d["bbox"]
            ycs.append(float(y) + float(h) / 2.0)
        if ycs:
            print(f"  row {r+1:02d}: {len(rows[r]):2d}   yc {min(ycs):.1f}..{max(ycs):.1f}")
        else:
            print(f"  row {r+1:02d}:  0   (no_dets)")

    # 6) Draw snapped cells (green boxes) and (r,c) tags; overlay matched dets in red
    for (r, c), det in grid.items():
        y = data_top + r * HEADER_ROW_H
        x = col_edges[c]
        w = COL_WIDTHS[c]
        h = HEADER_ROW_H
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0,200,0), 1, cv2.LINE_AA)
        _put_text(vis, f"{COL_ORDER[c]} @({r+1},{c+1})", (x + 6, y + 18), (0,100,0), 0.5, 1)
        if det:
            dx, dy, dw, dh = map(int, det["bbox"])
            cv2.rectangle(vis, (dx, dy), (dx + dw, dy + dh), (0,0,255), 1, cv2.LINE_AA)

    # 7) Write outputs
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out_img  = out_dir / (pathlib.Path(args.image).stem + "_debug.jpg")
    out_json = out_dir / (pathlib.Path(args.image).stem + "_dets.json")

    cv2.imwrite(str(out_img), vis)
    print(f"\n[debug] wrote overlay → {out_img}")

    pack = {
        "image": args.image,
        "table_top": table_top,
        "header_rows": args.header_rows,
        "row_h": HEADER_ROW_H,
        "rows_data": ROWS_DATA,
        "col_widths": COL_WIDTHS,
        "detections": dets,
        "grid": {f"{r},{c}": det for (r, c), det in grid.items()},
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(pack, f, indent=2)
    print(f"[debug] wrote det/grid json → {out_json}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="outputs/debug")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--header_rows", type=int, default=2,
                    help="How many header bands to skip before data rows.")
    args = ap.parse_args()
    main(args)
