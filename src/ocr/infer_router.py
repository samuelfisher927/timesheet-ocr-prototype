# src/ocr/infer_router.py
from __future__ import annotations
import os, io, json
from typing import Dict, List, Tuple
from PIL import Image
import yaml
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from src.ocr.beam_filter import rescore_candidates
from src.ocr.normalizers import sanitize_time, sanitize_amount, sanitize_date
from src.ocr.signature_presence import signature_present
import numpy as np
import cv2

FieldType = str  # "time" | "amount" | "text" | "date"
Device = "cuda" if torch.cuda.is_available() else "cpu"

class OCRHead:
    def __init__(self, model_id: str, ckpt_path: str | None):
        self.processor = TrOCRProcessor.from_pretrained(ckpt_path or model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(ckpt_path or model_id).to(Device)
        self.model.eval()

    @torch.no_grad()
    def infer(self, img: Image.Image, field_type: FieldType,
              num_beams=5, nbest=5, max_new_tokens=16):
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.to(Device)
        out = self.model.generate(
            pixel_values,
            num_beams=num_beams,
            num_return_sequences=nbest,
            early_stopping=True,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )
        texts = self.processor.batch_decode(out.sequences, skip_special_tokens=True)
        # quick-and-dirty per-seq score: sum of max logits chosen (same approach you already used)
        scores = []
        for i in range(out.sequences.size(0)):
            scores.append(0.0)
        # (we'll just use beam_filter to choose best; raw scores are a weak ranking anyway)
        best, _debug = rescore_candidates(texts, scores, field_type)
        # final sanitize per type
        if field_type == "time":
            final = sanitize_time(best) or best.strip()
        elif field_type == "amount":
            final = sanitize_amount(best) or best.strip()
        elif field_type == "date":
            final = sanitize_date(best) or best.strip()
        else:
            final = best.strip()
        return final, texts

def load_heads(cfg_yaml: str) -> Dict[FieldType, OCRHead]:
    with open(cfg_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    heads: Dict[FieldType, OCRHead] = {}
    for ft, spec in cfg.items():
        heads[ft] = OCRHead(spec["model_id"], spec.get("ckpt_path"))
    return heads

def crop_from_xyxy(page_img: Image.Image, box: Tuple[int,int,int,int], pad=2) -> Image.Image:
    x1,y1,x2,y2 = box
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(page_img.width,  x2 + pad); y2 = min(page_img.height, y2 + pad)
    return page_img.crop((x1,y1,x2,y2))

def run_router(
    page_image_path: str,
    grid_json_path: str,
    heads_yaml: str,
    out_rows_jsonl: str,
    class_map: Dict[str, FieldType],
):
    """
    grid_json must contain grid mapping like {"grid": {"r,c": {"bbox":[x1,y1,x2,y2], "cls":"in_am", ...}}, ...}
    class_map maps YOLO class names -> field types ("time","amount","text","date","signature")
    """
    heads = load_heads(heads_yaml)
    page = Image.open(page_image_path).convert("RGB")

    with open(grid_json_path, "r", encoding="utf-8") as f:
        G = json.load(f)

    rows: Dict[int, Dict[str, str]] = {}
    for key, det in G["grid"].items():
        r, c = map(int, key.split(","))
        cls = det["cls"]
        ft = class_map.get(cls, "text")
        box = det["bbox"]
        crop = crop_from_xyxy(page, tuple(box), pad=2)

        if ft == "signature":  # presence-only
            val = "present" if signature_present(cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)) else "absent"
        elif ft in heads:
            head = heads[ft]
            val, _cand = head.infer(crop, ft, num_beams=5, nbest=5, max_new_tokens=16)
        else:
            val = ""  # unknown/ignored

        rows.setdefault(r, {})[cls] = val

    with open(out_rows_jsonl, "w", encoding="utf-8") as f:
        for r in sorted(rows):
            f.write(json.dumps({"row": r, **rows[r]}) + "\n")
    print(f"[done] wrote {out_rows_jsonl}")
