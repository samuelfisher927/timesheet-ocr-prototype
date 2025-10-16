# src/pipeline.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
import glob

import cv2
import yaml

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

DEFAULT_CLASS_MAP = {
    "employee_name": "text",
    "date": "date",
    "in_am": "time", "out_am": "time",
    "lunch": "time",               # keep as "time" for now; can swap to "amount" later if you decide
    "in_pm": "time", "out_pm": "time",
    "total_hours": "amount",
    "signature": "signature",      # presence-only (no OCR text)
}

def _load_yaml(fp: str | None) -> dict:
    if not fp:
        return {}
    p = Path(fp)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {fp}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _coalesce(val, default):
    return default if val is None else val

def run_pipeline(
    image_path: str,
    yolo_weights: str,
    heads_yaml: str,
    out_dir: str = "outputs/demo",
    imgsz: int = 1280,
    conf: float = 0.25,
    header_rows: int = 2,
    crop_version: str = "v1",
    save_debug: bool = True,
    class_map: dict | None = None,
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
    overlay_path = None
    if save_debug:
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

    # persist JSON for router + auditability
    json_path = str(outp / (Path(image_path).stem + "_results.json"))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "image": image_path,
            "meta": {
                "crop_version": crop_version,
                "header_rows": header_rows,
            },
            "table_top_data": table_top_data,
            "row_h": HEADER_ROW_H,
            "rows_data": ROLES_DATA if (ROLES_DATA := ROWS_DATA) else ROWS_DATA,  # keep name in file stable
            "col_widths": col_widths,
            "detections": dets,
            "grid": {f"{r},{c}": v for (r,c), v in grid.items()},
        }, f, indent=2)

    rows_jsonl = str(outp / (Path(image_path).stem + "_rows_pred.jsonl"))
    run_router(
        page_image_path=image_path,
        grid_json_path=json_path,
        heads_yaml=heads_yaml,
        out_rows_jsonl=rows_jsonl,
        class_map=(class_map or DEFAULT_CLASS_MAP),
    )

    return {"json_path": json_path, "overlay_path": overlay_path, "rows_jsonl": rows_jsonl}


def _expand_inputs(inp: str) -> list[str]:
    # supports single file, directory, or glob pattern
    p = Path(inp)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        imgs = []
        for ext in ("*.jpg","*.jpeg","*.png","*.tif","*.tiff","*.bmp","*.webp"):
            imgs.extend(glob.glob(str(p / ext)))
        return sorted(imgs)
    # glob pattern
    paths = glob.glob(inp)
    return sorted([s for s in paths if Path(s).is_file()])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="file, folder, or glob (e.g. data/input/*.jpg)")
    ap.add_argument("--weights", default="models/yolo_timesheet_best.pt")
    ap.add_argument("--heads", default="src/ocr/heads.yaml")
    ap.add_argument("--out", default="outputs/demo")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--header_rows", type=int, default=2)
    ap.add_argument("--crop_version", default="v1")
    ap.add_argument("--save-debug", action="store_true", default=False)
    ap.add_argument("--config", default="", help="optional pipeline yaml to override flags")
    args = ap.parse_args()

    # load optional pipeline yaml and override defaults
    cfg = _load_yaml(args.config)
    weights      = _coalesce(cfg.get("yolo", {}).get("weights") if cfg else None, args.weights)
    conf         = _coalesce(cfg.get("yolo", {}).get("conf_thres") if cfg else None, args.conf)
    imgsz        = _coalesce(cfg.get("yolo", {}).get("imgsz") if cfg else None, args.imgsz)
    crop_version = _coalesce(cfg.get("crop", {}).get("version") if cfg else None, args.crop_version)
    save_debug   = _coalesce(cfg.get("export", {}).get("write_debug") if cfg else None, args.save_debug)

    heads_yaml   = _coalesce(cfg.get("trocr", {}).get("heads") if cfg else None, args.heads)
    header_rows  = _coalesce(cfg.get("table", {}).get("header_rows") if cfg else None, args.header_rows)

    out_dir = args.out
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    images = _expand_inputs(args.input)
    if not images:
        raise SystemExit(f"No images found for --input={args.input}")

    for i, img in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img}")
        run_pipeline(
            image_path=img,
            yolo_weights=weights,
            heads_yaml=heads_yaml,
            out_dir=out_dir,
            imgsz=imgsz,
            conf=conf,
            header_rows=header_rows,
            crop_version=crop_version,
            save_debug=save_debug,
        )
