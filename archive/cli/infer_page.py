# src/cli/infer_page.py
from __future__ import annotations
import argparse
from pathlib import Path

from src.pipeline import run_pipeline  # uses your register→detect→snap→OCR flow

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", required=True, help="YOLO .pt")
    ap.add_argument("--out", default="outputs/demo")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--header_rows", type=int, default=2)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    result = run_pipeline(
        image_path=args.image,
        yolo_weights=args.weights,
        out_dir=args.out,
        imgsz=args.imgsz,
        conf=args.conf,
        header_rows=args.header_rows,
    )
    print("[ok] wrote:", result["json_path"], "and", result.get("overlay_path","(no overlay)"))

if __name__ == "__main__":
    main()
