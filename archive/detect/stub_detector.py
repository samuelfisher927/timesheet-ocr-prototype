# ARCHIVED on 2025-10-13 — superseded by synth + YOLO flow.
# Keep for reference only. Not used in the current pipeline.

# src/detect/stub_detector.py
from __future__ import annotations
import json
import os
from typing import Dict, Tuple, Any

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

class StubDetector:
    """
    Minimal 'detector' that loads bboxes from ground-truth label JSONs.
    Use this while we build the pipeline; later we swap in a real detector.
    """

    def __init__(self, labels_dir: str):
        self.labels_dir = labels_dir
        if not os.path.isdir(self.labels_dir):
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

    def _label_path(self, image_id: str) -> str:
        # image_id matches the JSON filename (e.g., clock_0003 -> clock_0003.json)
        return os.path.join(self.labels_dir, f"{image_id}.json")

    def load_for_image(self, image_id: str) -> Dict[str, Any]:
        """
        Return the parsed JSON for an image_id.
        """
        path = self._label_path(image_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Label JSON not found for image_id={image_id}: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def detect(self, image_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Mimic a detector API:
        Returns {field_name: {'bbox': (x1,y1,x2,y2), 'score': 0.999, 'class': 'field'}}
        """
        js = self.load_for_image(image_id)
        fields = js.get("fields", {})
        result: Dict[str, Dict[str, Any]] = {}
        for fname, meta in fields.items():
            bbox = meta.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            result[fname] = {
                "bbox": tuple(int(v) for v in bbox),
                "score": 0.999,      # placeholder confidence
                "class": "field",    # single class in this stub
            }
        return result

    @staticmethod
    def list_image_ids(labels_dir: str) -> list[str]:
        ids = []
        for fn in os.listdir(labels_dir):
            if fn.endswith(".json"):
                ids.append(os.path.splitext(fn)[0])
        ids.sort()
        return ids


if __name__ == "__main__":
    # quick smoke test:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", required=True, help="Path to labels directory (e.g., data/clockin_synth/labels)")
    ap.add_argument("--image_id", default=None, help="If provided, print bboxes for this image_id; otherwise list ids")
    args = ap.parse_args()

    if args.image_id is None:
        ids = StubDetector.list_image_ids(args.labels_dir)
        print(f"Found {len(ids)} label files. First 5: {ids[:5]}")
    else:
        det = StubDetector(args.labels_dir)
        out = det.detect(args.image_id)
        print(f"{args.image_id}: {len(out)} fields")
        # print a few sample keys
        for k in list(out.keys())[:8]:
            print(k, out[k])