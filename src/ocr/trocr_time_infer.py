# src/ocr/trocr_time_infer.py
from __future__ import annotations
import os, json
from typing import Dict, List
from PIL import Image
from tqdm import tqdm
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def load_model(device: str = None):
    """
    Load pretrained TrOCR base model for handwritten text recognition.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Loading TrOCR model on {device} ...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)
    model.eval()
    return processor, model, device

def infer_time_fields(crops_dir: str, out_json: str):
    """
    Run OCR on all 'time' bucket crops under each timesheet folder.
    Saves predictions to a JSONL manifest.
    """
    processor, model, device = load_model()
    results: List[Dict] = []

    # gather all time crops
    time_crops = []
    for root, _, files in os.walk(crops_dir):
        if os.path.basename(root) == "time":
            for fn in files:
                if fn.lower().endswith(".png"):
                    time_crops.append(os.path.join(root, fn))

    print(f"[info] Found {len(time_crops)} time crops.")
    for path in tqdm(time_crops, desc="OCR"):
        img = Image.open(path).convert("RGB")
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # basic cleanup
        text = text.strip().replace(" ", "")
        results.append({"crop_path": path, "pred_text": text})

    with open(out_json, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[done] OCR results saved to {out_json}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops_dir", required=True, help="Path to data/clockin_synth/crops")
    ap.add_argument("--out_json", default="ocr_time_results.jsonl")
    args = ap.parse_args()
    infer_time_fields(args.crops_dir, args.out_json)
