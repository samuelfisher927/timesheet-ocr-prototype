# src/ocr/trocr_time_infer.py
from __future__ import annotations
import os, json, math, re
from typing import Dict, List, Tuple
from PIL import Image
from tqdm import tqdm
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from src.ocr.normalizers import sanitize_amount, sanitize_time
from src.ocr.beam_filter import rescore_candidates  # <-- use your beam filter

def load_model(device: str | None = None):
    """
    Load pretrained TrOCR base model for handwritten text recognition.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Loading TrOCR model on {device} ...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)
    model.eval()
    return processor, model, device

def _is_img(fn: str) -> bool:
    return fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"))

def _approx_seq_logprob(
    seqs: torch.LongTensor,
    scores: List[torch.FloatTensor],
    eos_token_id: int | None = None,
) -> List[float]:
    """
    Heuristic: with return_dict_in_generate=True and output_scores=True,
    `scores` is a list[t] of logits. Approximate per-sequence logprob
    by summing gathered log-softmax at chosen token ids.
    If eos_token_id is provided, stop accumulation at first EOS per sequence.
    """
    # scores: List[batch_logits_t], each [batch*nbest, vocab]
    # seqs: [batch*nbest, T]
    B = seqs.size(0)
    done = [False] * B
    logprobs = [0.0] * B

    with torch.no_grad():
        for t, logits in enumerate(scores):
            lsm = torch.log_softmax(logits, dim=-1)  # [B, V]
            tok = seqs[:, t]                          # [B]
            step_lp = lsm.gather(1, tok.view(-1, 1)).squeeze(1)  # [B]
            for i in range(B):
                if not done[i]:
                    logprobs[i] += float(step_lp[i].item())
                    if eos_token_id is not None and int(tok[i].item()) == eos_token_id:
                        done[i] = True
            if all(done):
                break
    return logprobs

def _score_nbest_local(
    candidates: List[Tuple[str, float]],
    field_hint: str | None = None,
) -> Tuple[str, str, float]:
    """
    Lightweight local rescoring (kept for parity with your original).
    Returns (best_text, best_sanitized, best_score).
    """
    best = None
    for text, logp in candidates:
        raw = (text or "").strip()
        compact = raw.replace(" ", "")
        bonus = 0.0

        if field_hint == "time":
            san = sanitize_time(compact)
            if san: bonus += 1.0
        elif field_hint == "amount":
            san = sanitize_amount(compact)
            if san: bonus += 1.0
        else:
            san = compact

        if ".." in compact or ",," in compact:
            bonus -= 0.5
        if field_hint in ("time", "amount") and re.search(r"[A-Za-z]", compact):
            bonus -= 0.5

        total = logp + bonus
        cand_sanitized = san if san else compact
        if best is None or total > best[2]:
            best = (raw, cand_sanitized, total)

    assert best is not None
    return best

def infer_time_fields(
    crops_dir: str,
    out_json: str,
    num_beams: int = 5,
    nbest: int = 5,
    max_new_tokens: int = 64,
    field_hint: str = "time",  # "time" | "amount" | "text"
    use_beam_filter: bool = True,
):
    """
    Run OCR on all crops under crops_dir.
    Saves predictions to a JSONL manifest with n-best and simple rescoring.
    """
    processor, model, device = load_model()

    # gather all crops
    crop_paths: List[str] = []
    for root, _, files in os.walk(crops_dir):
        for fn in files:
            if _is_img(fn):
                crop_paths.append(os.path.join(root, fn))

    print(f"[info] Found {len(crop_paths)} crops.")
    results: List[Dict] = []

    # Generation kwargs
    gen_kwargs = dict(
        num_beams=max(1, num_beams),
        num_return_sequences=max(1, nbest),
        early_stopping=True,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
    )

    eos_id = getattr(processor.tokenizer, "eos_token_id", None)

    for path in tqdm(crop_paths, desc="OCR"):
        img = Image.open(path).convert("RGB")
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            out = model.generate(pixel_values, **gen_kwargs)

        texts = processor.batch_decode(out.sequences, skip_special_tokens=True)

        # Approximate per-sequence logprobs for rescoring (EOS-aware)
        seq_logps = _approx_seq_logprob(out.sequences, out.scores, eos_token_id=eos_id)

        # Prepare n-best
        nbest_list = [{"text": t, "logprob": lp} for t, lp in zip(texts, seq_logps)]
        candidates = [x["text"] for x in nbest_list]
        candidate_logprobs = [x["logprob"] for x in nbest_list]

        # Choose best: prefer your stricter beam_filter (sanitizer + validators)
        if use_beam_filter:
            field_map = {"time": "time", "amount": "amount"}
            ft = field_map.get(field_hint, "text")
            best_text, _debug = rescore_candidates(candidates, candidate_logprobs, ft)
            # (Optional) keep _debug if you want per-beam notes – omitted in JSON to keep it smaller
        else:
            best_text, _, _ = _score_nbest_local([(x["text"], x["logprob"]) for x in nbest_list], field_hint=field_hint)

        # Final sanitizer for the chosen field
        if field_hint == "time":
            best_sanitized = sanitize_time(best_text) or best_text.strip()
        elif field_hint == "amount":
            best_sanitized = sanitize_amount(best_text) or best_text.strip()
        else:
            best_sanitized = best_text.strip()

        results.append({
            "crop_path": path,
            "pred_text": best_text,
            "pred_text_sanitized": best_sanitized,
            "score": max(candidate_logprobs) if candidate_logprobs else float("-inf"),
            "candidates": candidates,
            "candidate_logprobs": candidate_logprobs,
            "nbest": nbest_list  # <-- keep single, non-duplicated key
        })

    with open(out_json, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[done] OCR results saved to {out_json}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops_dir", required=True, help="Path to crops directory")
    ap.add_argument("--out_json", default="ocr_time_results.jsonl")
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--nbest", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--field_hint", choices=["time", "amount", "text"], default="time",
                    help="Lightweight rescoring hint; does not change model, only n-best ranking.")
    ap.add_argument("--no_beam_filter", action="store_true",
                    help="Disable strict beam_filter rescoring and use local scorer instead.")
    args = ap.parse_args()

    infer_time_fields(
        crops_dir=args.crops_dir,
        out_json=args.out_json,
        num_beams=args.num_beams,
        nbest=args.nbest,
        max_new_tokens=args.max_new_tokens,
        field_hint=args.field_hint,
        use_beam_filter=not args.no_beam_filter,
    )
