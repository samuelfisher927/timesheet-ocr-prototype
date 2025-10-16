# src/ocr/train_trocr_head_min.py
from __future__ import annotations
import os, json, argparse, warnings
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from peft import (
    LoraConfig, get_peft_model, PeftModel, PeftConfig,
    get_peft_model_state_dict, set_peft_model_state_dict
)

# put near top with imports
import re
from transformers import LogitsProcessor

class DigitsOnlyProcessor(LogitsProcessor):
    """Mask all non [0-9 . , <eos> <pad>] tokens during generation."""
    def __init__(self, tokenizer):
        self.tok = tokenizer
        allow = set(list("0123456789.,"))
        self.allowed_ids = {self.tok.convert_tokens_to_ids(ch) for ch in allow if ch in self.tok.get_vocab()}
        # also allow special tokens needed for stopping/padding
        for tid in [self.tok.eos_token_id, self.tok.pad_token_id, self.tok.bos_token_id]:
            if tid is not None: self.allowed_ids.add(tid)

    def __call__(self, input_ids, scores):
        # scores: [batch, vocab]
        mask = torch.full_like(scores, float("-inf"))
        mask[:, list(self.allowed_ids)] = 0.0
        return scores + mask


def _canon_amount(s: str) -> str:
    s = str(s).strip().replace(",", ".")
    s = re.sub(r"[^0-9.]", "", s)
    # keep at most one dot
    if s.count(".") > 1:
        parts = s.split(".")
        s = parts[0] + "." + "".join(parts[1:])
    if s == "" or s == ".": 
        return ""
    if "." not in s:
        s = s + ".00"
    a, b = s.split(".", 1)
    if a == "": a = "0"
    b = (b + "00")[:2]
    return f"{int(a)}.{b}"


# Optional CER metric (nice to have)
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
def evaluate_epoch(model, processor, loader, task_type: str = "generic"):
    model.eval()
    preds, refs = [], []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(Device, non_blocking=True)
        enc_out = model.get_encoder()(pixel_values=pixel_values)
        logits_proc = DigitsOnlyProcessor(processor.tokenizer)
        pred_ids = model.generate(
            encoder_outputs=enc_out,
            max_length=32,
            num_beams=5,
            length_penalty=0.0,
            logits_processor=[logits_proc],
        )
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

        label_ids = batch["labels"].cpu().numpy().copy()
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is not None:
            label_ids[label_ids == -100] = pad_id
        ref_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        for p, r in zip(pred_str, ref_str):
            p = p.strip(); r = r.strip()
            if task_type == "amount":
                p = _canon_amount(p); r = _canon_amount(r)
            preds.append(p); refs.append(r)

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
    ap.add_argument("--weight_decay", type=float, default=0.0)       # NEW
    ap.add_argument("--scheduler", choices=["none","linear","cosine"], default="none")  # NEW
    ap.add_argument("--warmup_ratio", type=float, default=0.0)        # NEW (fraction of total steps)
    ap.add_argument("--grad_clip", type=float, default=0.0)           # NEW (0 disables clipping)
    ap.add_argument("--task_type", default="generic", choices=["generic","time","amount","date","text"])
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--save_lora_dir", default="")
    ap.add_argument("--save_merged_dir", default="")
    ap.add_argument("--resume_dir", default="", help="path to epoch_xx folder with adapter_model.bin")
    ap.add_argument("--start_epoch", type=int, default=1, help="epoch number to start from (e.g., 2 when resuming after epoch_01)")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[info] device={Device}  task_type={args.task_type}")

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

    # ----- optional resume -----
    start_epoch = args.start_epoch
    if args.resume_dir:
        adapter_path = os.path.join(args.resume_dir, "adapter_model.bin")
        if os.path.isfile(adapter_path):
            sd = torch.load(adapter_path, map_location="cpu")
            set_peft_model_state_dict(model, sd)
            print(f"[resume] Loaded LoRA adapters from {adapter_path}")
            # if the tokenizer/preprocessor was saved, reload to be extra safe
            try:
                processor = TrOCRProcessor.from_pretrained(args.resume_dir)
                print(f"[resume] Reloaded processor from {args.resume_dir}")
            except Exception:
                pass
        else:
            print(f"[resume] WARNING: {adapter_path} not found; starting from base LoRA init")

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

    # Optimizer + Scheduler + AMP
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    # total training steps remaining (after any resume)
    steps_per_epoch = max(1, len(train_dl))
    total_epochs_left = max(0, args.epochs - (start_epoch - 1))
    num_training_steps = steps_per_epoch * total_epochs_left
    warmup_steps = int(args.warmup_ratio * num_training_steps)

    if args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
    elif args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=(Device=="cuda"))

    best_exact = -1.0
    for epoch in range(start_epoch, args.epochs+1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_dl, 1):
            pixel_values = batch["pixel_values"].to(Device, non_blocking=True)
            labels = batch["labels"].to(Device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(Device=="cuda")):
                enc_out = model.get_encoder()(pixel_values=pixel_values)
                out = model(encoder_outputs=enc_out, labels=labels)
                logits = out.logits.float()
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                # ignore pads
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            scaler.scale(loss).backward()

            # Gradient clipping (if enabled)
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            # Scheduler step (if any)
            if scheduler is not None:
                scheduler.step()

            running += loss.item()
            if step % 2000 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"[epoch {epoch}] step {step}/{len(train_dl)}  loss={running/step:.4f}  lr={lr_now:.6g}")

        # epoch metrics
        metrics = evaluate_epoch(model, processor, val_dl, task_type=args.task_type)
        print(f"[epoch {epoch}] val: exact={metrics['exact_match']:.4f}" +
              (f" cer={metrics['cer']:.4f}" if 'cer' in metrics else ""))

        # save per-epoch lightweight LoRA ckpt
        ckpt_dir = os.path.join(args.out_dir, f"epoch_{epoch:02d}")
        os.makedirs(ckpt_dir, exist_ok=True)
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
