# src/ocr/smoke_decode.py
from __future__ import annotations
import argparse
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to merged model dir (or HF id)")
    ap.add_argument("--image", required=True, help="Path to a single date/text crop image")
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cuda" if args.device == "cuda" else "cpu")
    )

    processor = TrOCRProcessor.from_pretrained(args.ckpt)
    model = VisionEncoderDecoderModel.from_pretrained(args.ckpt).to(device)
    model.eval()
    model.config.max_length = args.max_len
    model.config.use_cache = False

    img = Image.open(args.image).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        gen_ids = model.generate(pixel_values, max_length=args.max_len)
    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    print(f"PRED: {text}")

if __name__ == "__main__":
    main()
