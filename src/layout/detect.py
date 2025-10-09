# src/layout/detect.py
# YOLO inference wrapper (Ultralytics). pip install ultralytics
# Usage:
#   dets = run_yolo("best.pt", "scan.jpg", conf=0.25, imgsz=1280)

from typing import List, Dict
import numpy as np

def run_yolo(weights: str, image_path: str, conf: float=0.25, imgsz: int=1280) -> List[Dict]:
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise ImportError("Ultralytics not installed. pip install ultralytics") from e

    model = YOLO(weights)
    results = model.predict(image_path, conf=conf, imgsz=imgsz, verbose=False)
    out = []
    for r in results:
        names = r.names
        if r.boxes is None: continue
        for b in r.boxes:
            xywh = b.xywh.cpu().numpy().astype(float)[0]
            x, y, w, h = float(xywh[0]-xywh[2]/2), float(xywh[1]-xywh[3]/2), float(xywh[2]), float(xywh[3])
            cls_id = int(b.cls.cpu().numpy()[0])
            score = float(b.conf.cpu().numpy()[0])
            out.append({"cls": names[cls_id], "bbox": [x,y,w,h], "score": score})
    return out
