# src/ocr/train_trocr_head_min.py
from __future__ import annotations
import os, json, math, argparse, warnings
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig, get_peft_model_state_dict

# Optional CER
try:
    import evaluate
    CER = evaluate.load("cer")
except Exception:
    CER = None
    warnings.warn("evaluate not installed -> CER disabled (pip install evaluate)")

Device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class Rec:
    image: str
    text: str

def load_jsonl(fp: str) -> List[Rec]:
    out = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            out.append(Rec(j["image"], str(j["text"])))
    return out

class CropDataset(Dataset):
    def __init__(self, recs: List[Rec], processor: TrOCRProcessor, max_len=32):
        self.recs, self.proc, self.max_len = recs, processor, max_len

    def __len__(self): return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = Image.open(r.image).convert("RGB")
        pixel_values = self.proc(images=img, return_tensors="pt").pixel_values.squeeze(0)

        labels = self.proc.tokenizer(
            r.text, max_length=self.max_len, truncation=True, padding=False, return_tensors="pt"
        ).input_ids.squeeze(0)

        return {"pixel_values": pixel_values, "labels": labels}

def collate_fn(examples: List[Dict], pad_id: int):
    pv = torch.stack([e["pixel_values"] for e in examples])
    max_len = max(e["labels"].shape[0] for e in examples)
    labels = []
    for e in examples:
        lab = e["labels"]
        if lab.shape[0] < max_len:
            pad = torch.full((max_len - lab.shape[0],), pad_id, dtype=lab.dtype)
            lab = torch.cat([lab, pad], dim=0)
        labels.append(lab)
    labels = torch.stack(labels)
    labels = labels.masked_fill(labels == pad_id, -100)
    return {"pixel_values": pv, "labels": labels}

@torch.no_grad()
def evaluate_epoch(model, processor, loader):
    model.eval()
    preds, refs = [], []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(Device, non_blocking=True)

        # Encode once, then generate from encoder_outputs (works with PEFT)
        enc_out = model.get_encoder()(pixel_values=pixel_values)
        pred_ids = model.generate(encoder_outputs=enc_out, max_length=32)
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

        # Recover references (swap -100 back to pad_id before decoding)
        label_ids = batch["labels"].cpu().numpy().copy()
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is not None:
            label_ids[label_ids == -100] = pad_id
        ref_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        preds.extend([p.strip() for p in pred_str])
        refs.extend([r.strip() for r in ref_str])

    exact = sum(p == r for p, r in zip(preds, refs)) / max(1, len(preds))
    out = {"exact_match": exact}
    if CER is not None:
        out["cer"] = CER.compute(predictions=preds, references=refs)
    return out

def save_lora(model: PeftModel, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(get_peft_model_state_dict(model), os.path.join(save_dir, "adapter_model.bin"))
    model.peft_config["default"].save_pretrained(save_dir)
    print(f"[save] LoRA adapters -> {save_dir}")

def merge_and_save(base_model_id: str, lora_dir: str, merged_dir: str):
    os.makedirs(merged_dir, exist_ok=True)
    base = VisionEncoderDecoderModel.from_pretrained(base_model_id)
    _ = PeftConfig.from_pretrained(lora_dir)  # validate presence
    peft_model = PeftModel.from_pretrained(base, lora_dir)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    print(f"[save] merged model -> {merged_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="microsoft/trocr-base-handwritten")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl",   required=True)
    ap.add_argument("--out_dir",     required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--save_lora_dir", default="")
    ap.add_argument("--save_merged_dir", default="")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[info] device={Device}")

    processor = TrOCRProcessor.from_pretrained(args.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model)

    model.config.use_cache = False

    # decoder/pad IDs
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = (
            processor.tokenizer.cls_token_id
            if processor.tokenizer.cls_token_id is not None
            else processor.tokenizer.bos_token_id
        )
    if model.config.pad_token_id is None and processor.tokenizer.pad_token_id is not None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.max_length = args.max_len

    # LoRA on q/v projections
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj","v_proj"], bias="none", task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_cfg).to(Device)
    model.print_trainable_parameters()

    # Data
    train_recs = load_jsonl(args.train_jsonl)
    val_recs   = load_jsonl(args.val_jsonl)
    pad_id = processor.tokenizer.pad_token_id

    train_ds = CropDataset(train_recs, processor, max_len=args.max_len)
    val_ds   = CropDataset(val_recs,   processor, max_len=args.max_len)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_id)
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_id)
    )

    # Optimizer + AMP
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(Device=="cuda"))

    best_exact = -1.0
    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_dl, 1):
            pixel_values = batch["pixel_values"].to(Device, non_blocking=True)
            labels = batch["labels"].to(Device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(Device=="cuda")):
                enc_out = model.get_encoder()(pixel_values=pixel_values)
                out = model(encoder_outputs=enc_out, labels=labels)
                loss = out.loss

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            if step % 2000 == 0:
                print(f"[epoch {epoch}] step {step}/{len(train_dl)}  loss={running/step:.4f}")

        # epoch metrics
        metrics = evaluate_epoch(model, processor, val_dl)
        print(f"[epoch {epoch}] val: exact={metrics['exact_match']:.4f}" +
              (f" cer={metrics['cer']:.4f}" if 'cer' in metrics else ""))

        # save per-epoch lightweight LoRA ckpt
        ckpt_dir = os.path.join(args.out_dir, f"epoch_{epoch:02d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        # save adapters (LoRA weights)
        torch.save(get_peft_model_state_dict(model), os.path.join(ckpt_dir, "adapter_model.bin"))
        model.peft_config["default"].save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)

        if metrics["exact_match"] > best_exact:
            best_exact = metrics["exact_match"]
            torch.save(get_peft_model_state_dict(model), os.path.join(args.out_dir, "best_adapter_model.bin"))
            print(f"[epoch {epoch}] 🔥 new best exact={best_exact:.4f} -> saved")

    # optional final exports
    if args.save_lora_dir:
        save_lora(model, args.save_lora_dir)
    if args.save_merged_dir:
        merge_and_save(args.base_model, args.save_lora_dir or args.out_dir, args.save_merged_dir)

    print("[done]")
    processor.save_pretrained(args.out_dir)

if __name__ == "__main__":
    main()
