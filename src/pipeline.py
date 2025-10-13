# src/pipeline.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
from src.ocr.infer_router import run_router
from src.layout.detect import run_yolo
from src.layout.grid_snap import snap_to_grid
# from src.layout.register import register_to_template  # enable if you use Phase 1

# --- layout constants (match your synth/debug) ---
PAGE_W, PAGE_H = 1700, 2200
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 80, 80, 180, 120
HEADER_GAP   = 90
HEADER_ROW_H = 120
ROWS_DATA    = 12
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

def run_pipeline(
    image_path: str,
    yolo_weights: str,
    out_dir: str = "outputs/demo",
    imgsz: int = 1280,
    conf: float = 0.25,
    header_rows: int = 2,
):
    """
    image -> (optional) register -> YOLO detect -> grid snap -> (downstream OCR)
    """
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    assert img_bgr is not None, f"Cannot read image: {image_path}"
    # img_bgr = register_to_template(img_bgr)  # if Phase 1 enabled

    dets = run_yolo(yolo_weights, image_path, conf=conf, imgsz=imgsz)

    table_top_data = MARGIN_T + HEADER_GAP + header_rows * HEADER_ROW_H
    col_widths = [w for _,_,w in COLS]

    grid = snap_to_grid(
        detections=dets,
        table_top=table_top_data,
        row_h=HEADER_ROW_H,
        n_rows_total=ROWS_DATA,
        x0=MARGIN_L,
        col_widths=col_widths,
        class_names=[c[0] for c in COLS],
        ignore_classes=("date",),
    )

    # overlay (debug)
    vis = img_bgr.copy()
    x = MARGIN_L
    for w in col_widths:
        cv2.line(vis, (x, table_top_data), (x, table_top_data + ROWS_DATA*HEADER_ROW_H), (0,255,0), 1, cv2.LINE_AA)
        x += w
    cv2.line(vis, (x, table_top_data), (x, table_top_data + ROWS_DATA*HEADER_ROW_H), (0,255,0), 1, cv2.LINE_AA)
    for r in range(ROWS_DATA+1):
        y = table_top_data + r*HEADER_ROW_H
        cv2.line(vis, (MARGIN_L, y), (MARGIN_L + sum(col_widths), y), (0,255,0), 1, cv2.LINE_AA)

    for d in dets:
        x,y,w,h = map(int, d["bbox"])
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,0,255), 1, cv2.LINE_AA)

    overlay_path = str(outp / (Path(image_path).stem + "_overlay.jpg"))
    cv2.imwrite(overlay_path, vis)

    json_path = str(outp / (Path(image_path).stem + "_results.json"))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "image": image_path,
            "table_top_data": table_top_data,
            "row_h": HEADER_ROW_H,
            "rows_data": ROWS_DATA,
            "col_widths": col_widths,
            "detections": dets,
            "grid": {f"{r},{c}": v for (r,c), v in grid.items()},
        }, f, indent=2)

    CLASS_MAP = {
        "employee_name": "text",
        "date": "date",
        "in_am": "time", "out_am": "time",
        "lunch": "time",
        "in_pm": "time", "out_pm": "time",
        "total_hours": "amount",
        "signature": "signature",  # presence-only (no OCR text)
    }
    rows_jsonl = str(outp / (Path(image_path).stem + "_rows_pred.jsonl"))
    run_router(
        page_image_path=image_path,
        grid_json_path=json_path,
        heads_yaml="src/ocr/heads.yaml",
        out_rows_jsonl=rows_jsonl,
        class_map=CLASS_MAP,
    )

    return {"json_path": json_path, "overlay_path": overlay_path, "rows_jsonl": rows_jsonl}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default="models/yolo_timesheet_best.pt")
    ap.add_argument("--out", default="outputs/demo")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--header_rows", type=int, default=2)
    args = ap.parse_args()

    run_pipeline(
        image_path=args.image,
        yolo_weights=args.weights,
        out_dir=args.out,
        imgsz=args.imgsz,
        conf=args.conf,
        header_rows=args.header_rows,
    )
