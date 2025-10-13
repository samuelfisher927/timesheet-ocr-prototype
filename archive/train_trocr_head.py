# src/ocr/train_trocr_head.py
from __future__ import annotations
import os, json, warnings
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import shift_tokens_right

# PEFT (LoRA)
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig, get_peft_model_state_dict

# Metrics (optional, if 'evaluate' installed)
try:
    import evaluate
    _HAS_EVAL = True
except Exception:
    _HAS_EVAL = False
    warnings.warn("evaluate not installed -> metrics disabled (pip install evaluate)")

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
    """
    Produces:
      - pixel_values: FloatTensor [3,H,W]
      - labels:       LongTensor  [T]
    """
    def __init__(self, recs: List[Rec], processor: TrOCRProcessor, max_len=32):
        self.recs, self.proc, self.max_len = recs, processor, max_len

    def __len__(self): return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = Image.open(r.image).convert("RGB")

        # vision
        pixel_values = self.proc(images=img, return_tensors="pt").pixel_values.squeeze(0)

        # text labels
        tok = self.proc.tokenizer(
            r.text,
            max_length=self.max_len,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        labels = tok.input_ids.squeeze(0)
        return {"pixel_values": pixel_values, "labels": labels}

def make_data_collator(processor: TrOCRProcessor):
    pad_id = processor.tokenizer.pad_token_id

    def collate(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])

        # pad labels -> -100
        max_len = max(len(b["labels"]) for b in batch)
        labels = []
        for b in batch:
            lab = b["labels"]
            if lab.size(0) < max_len:
                pad = torch.full((max_len - lab.size(0),), pad_id, dtype=lab.dtype)
                lab = torch.cat([lab, pad], dim=0)
            labels.append(lab)
        labels = torch.stack(labels)
        labels = labels.masked_fill(labels == pad_id, -100)

        return {"pixel_values": pixel_values, "labels": labels}

    return collate

def build_metrics(processor: TrOCRProcessor):
    cer_metric = None
    if _HAS_EVAL:
        import evaluate as _ev
        cer_metric = _ev.load("cer")

    def compute_metrics(pred):
        import numpy as np
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        pad_id = processor.tokenizer.pad_token_id
        label_ids = np.where(label_ids != -100, label_ids, pad_id)

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        out = {}
        # always compute exact-match
        out["exact_match"] = sum(p.strip() == l.strip() for p, l in zip(pred_str, label_str)) / max(1, len(pred_str))
        # add CER if evaluate is available
        if cer_metric is not None:
            out["cer"] = cer_metric.compute(predictions=pred_str, references=label_str)
        return out

    return cer_metric, compute_metrics

class DelayedEarlyStopping(EarlyStoppingCallback):
    """Don’t trigger early stopping until after start_epoch."""
    def __init__(self, start_epoch=10, patience=3, threshold=0.0):
        super().__init__(early_stopping_patience=patience, early_stopping_threshold=threshold)
        self.start_epoch = start_epoch
    def on_evaluate(self, args, state, control, **kwargs):
        if state.epoch is not None and state.epoch < self.start_epoch:
            return control
        return super().on_evaluate(args, state, control, **kwargs)

def save_lora_adapters(model: PeftModel, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    state_dict = get_peft_model_state_dict(model)
    torch.save(state_dict, os.path.join(save_dir, "adapter_model.bin"))
    # save the LoRA config
    model.peft_config["default"].save_pretrained(save_dir)

def merge_and_save(base_model_id: str, lora_dir: str, merged_out_dir: str):
    os.makedirs(merged_out_dir, exist_ok=True)
    base = VisionEncoderDecoderModel.from_pretrained(base_model_id)
    _ = PeftConfig.from_pretrained(lora_dir)  # validates presence
    peft_model = PeftModel.from_pretrained(base, lora_dir)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(merged_out_dir)
    print(f"[save] merged model -> {merged_out_dir}")

@torch.no_grad()
def sanity_probe(model, processor, dataset, device=Device):
    """
    Forward a single example with explicit args to avoid any kwarg routing issues.
    """
    assert len(dataset) > 0, "Dataset is empty"
    model.eval().to(device)

    # pull one record
    rec = dataset.recs[0] if hasattr(dataset, "recs") else None
    if rec is None:
        rec = dataset[0]
        img_path = rec["image"]; text = rec["text"]
    else:
        img_path = rec.image; text = rec.text

    # vision
    from PIL import Image
    from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import shift_tokens_right
    img = Image.open(img_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)

    # text labels
    tok = processor.tokenizer(text, return_tensors="pt", padding=False, truncation=True)
    labels = tok.input_ids.to(device)
    pad_id = processor.tokenizer.pad_token_id
    labels = labels.masked_fill(labels == pad_id, -100)

    # ensure IDs
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = (
            processor.tokenizer.cls_token_id
            if processor.tokenizer.cls_token_id is not None
            else processor.tokenizer.bos_token_id
        )
    if model.config.pad_token_id is None and processor.tokenizer.pad_token_id is not None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id

    # explicit decoder inputs
    decoder_input_ids = shift_tokens_right(
        labels,
        pad_token_id=model.config.pad_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    out = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids, labels=labels)
    print(f"[sanity] device={device} loss={float(out.loss):.4f}")
    model.train()

class VisionSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        pixel_values = inputs.get("pixel_values", None)
        labels = inputs.get("labels", None)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="microsoft/trocr-base-handwritten")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--out_dir", required=True, help="where to save checkpoints")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--eval_steps", type=int, default=250)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--early_start_epoch", type=int, default=25)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--save_merged_dir", default="", help="if set, save merged full model here too")
    ap.add_argument("--save_lora_dir", default="", help="if set, also save LoRA adapters here")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[info] device={Device}")
    processor = TrOCRProcessor.from_pretrained(args.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model)

    # decoder start + pad ids
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = (
            processor.tokenizer.cls_token_id
            if processor.tokenizer.cls_token_id is not None
            else processor.tokenizer.bos_token_id
        )
    if model.config.pad_token_id is None and processor.tokenizer.pad_token_id is not None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.max_length = args.max_len

    # Older versions compatibility
    if not hasattr(type(model.config), "vocab_size"):
        setattr(type(model.config), "vocab_size", property(lambda self: self.decoder.vocab_size))

    # LoRA config (q,v projections)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Datasets
    train_ds = CropDataset(load_jsonl(args.train_jsonl), processor, max_len=args.max_len)
    val_ds   = CropDataset(load_jsonl(args.val_jsonl),   processor, max_len=args.max_len)

    # Collator + metrics
    data_collator = make_data_collator(processor)
    _, compute_metrics = build_metrics(processor)

    # --- one-shot sanity probe before Trainer spins up ---
    # sanity_probe(model, processor, train_ds, device=Device)

    # Trainer args — notebook-style
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        evaluation_strategy="epoch",   # evaluate once per epoch
        logging_strategy="epoch",      # log once per epoch
        save_strategy="epoch",         # (optional) save at end of each epoch
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        label_smoothing_factor=0.1,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to=[],
        load_best_model_at_end=False,         
        metric_for_best_model= None,   
        greater_is_better=True,
        save_total_limit=2,                          # optional: keep disk usage small
    )
        

    trainer = VisionSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor.tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[],  # early stopping disabled
    )

    trainer.train()

    # Optional: save LoRA adapters (for archival)
    if args.save_lora_dir:
        save_lora_adapters(model, args.save_lora_dir)
        print(f"[save] LoRA adapters -> {args.save_lora_dir}")

    # Optional: merge adapters into base & save a normal TrOCR (best for your router)
    if args.save_merged_dir:
        merge_and_save(args.base_model, args.save_lora_dir or args.out_dir, args.save_merged_dir)

    # Always save the processor too (tokenizer/image size)
    processor.save_pretrained(args.out_dir)
    print(f"[done] training complete; base outputs in {args.out_dir}")
    if args.save_merged_dir:
        print(f"[done] merged model saved to {args.save_merged_dir}")

if __name__ == "__main__":
    main()
