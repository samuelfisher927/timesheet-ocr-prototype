# src/ocr/merge_lora.py
from __future__ import annotations
import os
import argparse
import shutil
from pathlib import Path

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import PeftModel, PeftConfig

def load_processor(preferred_dir: str | None, fallback_model_id: str):
    """
    Try to load processor from the adapters dir (if present), otherwise from base model id.
    """
    if preferred_dir and (Path(preferred_dir) / "preprocessor_config.json").exists():
        print(f"[merge] Loading processor from adapters dir: {preferred_dir}")
        return TrOCRProcessor.from_pretrained(preferred_dir)
    print(f"[merge] Loading processor from base model: {fallback_model_id}")
    return TrOCRProcessor.from_pretrained(fallback_model_id)

def main():
    ap = argparse.ArgumentParser(description="Merge LoRA adapters into a full TrOCR model.")
    ap.add_argument("--base_model", default="microsoft/trocr-base-handwritten",
                    help="HF model id or path for the base model")
    ap.add_argument("--lora_dir", required=True,
                    help="Folder that contains adapter_model.bin (your finetuned LoRA)")
    ap.add_argument("--out_dir", required=True,
                    help="Output folder to save the merged full model")
    ap.add_argument("--copy_readme", action="store_true",
                    help="If set, copy README/README.md from lora_dir into out_dir")
    args = ap.parse_args()

    base_model_id = args.base_model
    lora_dir = args.lora_dir
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    # 1) Sanity check adapters
    if not (Path(lora_dir) / "adapter_model.bin").exists() and not (Path(lora_dir) / "adapter_model.safetensors").exists():
        raise FileNotFoundError(
            f"No LoRA weights found in {lora_dir}. Expected adapter_model.bin or adapter_model.safetensors."
        )
    try:
        _ = PeftConfig.from_pretrained(lora_dir)
    except Exception as e:
        print("[merge] Warning: Could not read PEFT config from adapters. "
              "If your adapter was saved as raw state_dict, merge will still try to proceed.")
        # Not fatal for common saves that only have adapter_model.bin

    # 2) Load base model
    print(f"[merge] Loading base model: {base_model_id}")
    base = VisionEncoderDecoderModel.from_pretrained(base_model_id)

    # 3) Load PEFT adapters on top of base
    print(f"[merge] Attaching LoRA adapters from: {lora_dir}")
    peft_model = PeftModel.from_pretrained(base, lora_dir)

    # 4) Merge & unload (produces a standard VisionEncoderDecoderModel)
    print("[merge] Merging adapters into base weights...")
    merged = peft_model.merge_and_unload()

    # 5) Save merged full model
    print(f"[merge] Saving merged model to: {out_dir}")
    merged.save_pretrained(out_dir)

    # 6) Save processor (prefer adapter’s processor if present, else base)
    processor = load_processor(preferred_dir=lora_dir, fallback_model_id=base_model_id)
    processor.save_pretrained(out_dir)

    # 7) Optional: copy readme for traceability
    if args.copy_readme:
        for name in ("README.md", "README", "readme.md", "readme"):
            src = Path(lora_dir) / name
            if src.exists():
                dst = Path(out_dir) / "README.md"
                try:
                    shutil.copy2(src, dst)
                    print(f"[merge] Copied {src.name} -> {dst}")
                except Exception:
                    pass
                break

    print("[merge] Done. You can now load with:")
    print(f"  from transformers import VisionEncoderDecoderModel, TrOCRProcessor")
    print(f"  model = VisionEncoderDecoderModel.from_pretrained(r\"{out_dir}\")")
    print(f"  processor = TrOCRProcessor.from_pretrained(r\"{out_dir}\")")

if __name__ == "__main__":
    main()
