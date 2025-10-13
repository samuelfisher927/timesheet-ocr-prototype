# src/ocr/beam_filter.py
from __future__ import annotations
from typing import List, Tuple, Literal
import math
import re

# Use your existing normalizers
from src.ocr.normalizers import sanitize_amount, sanitize_time

FieldType = Literal["amount", "time", "text"]

# Quick character-level prefilters to down-weight obviously-wrong beams
AMOUNT_PREFILTER_RX = re.compile(r"^[()\-$\s\d.,]+$")
TIME_PREFILTER_RX   = re.compile(r"^[\s0-9:;.,\-apmAPM]+$")

def is_valid_amount(s: str) -> bool:
    """True if amount parses cleanly via sanitize_amount."""
    return sanitize_amount(s) is not None

def is_valid_time(s: str) -> bool:
    """True if time parses cleanly via sanitize_time."""
    return sanitize_time(s) is not None

def _normalize_text(s: str) -> str:
    # Collapse whitespace, strip edges
    return re.sub(r"\s+", " ", s or "").strip()

def postprocess_field(s: str, kind: FieldType) -> Tuple[str, bool]:
    """
    Returns (clean_string, sanitized_ok).
    - amount: normalized string if parseable else original-stripped
    - time:   normalized string if parseable else original-stripped
    - text:   whitespace-normalized
    """
    raw_norm = _normalize_text(s)
    if kind == "amount":
        v = sanitize_amount(raw_norm)
        return (v if v is not None else raw_norm, v is not None)
    if kind == "time":
        v = sanitize_time(raw_norm)
        return (v if v is not None else raw_norm, v is not None)
    # text
    return (raw_norm, bool(raw_norm))

def _levenshtein(a: str, b: str) -> int:
    """Tiny Levenshtein to gently penalize messy outputs."""
    if a == b: return 0
    if not a:  return len(b)
    if not b:  return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = prev[j] + 1
            dele = curr[j - 1] + 1
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]

def _prefilter_ok(s: str, field_type: FieldType) -> bool:
    if field_type == "amount":
        return bool(AMOUNT_PREFILTER_RX.match(s))
    if field_type == "time":
        return bool(TIME_PREFILTER_RX.match(s))
    return True  # for text

def rescore_candidates(
    candidates: List[str],
    seq_logprobs: List[float],
    field_type: FieldType
) -> Tuple[str, List[Tuple[str, float, str]]]:
    """
    Given n-best candidates and their sequence logprobs, return:
      - best_raw (string picked from candidates)
      - debug list of (candidate, score, note)
    Score = model_logprob
            + bonus if sanitizer succeeds
            + bonus if regex/validator passes
            - small penalty proportional to edit distance (raw vs clean)
            - penalty if prefilter fails
    """
    assert len(candidates) == len(seq_logprobs)
    rescored: List[Tuple[str, float, str]] = []

    for cand, logp in zip(candidates, seq_logprobs):
        score = float(logp) if math.isfinite(logp) else -1e9
        note_bits = []

        # 0) Prefilter penalty for illegal character sets
        if not _prefilter_ok(cand, field_type):
            score -= 2.0
            note_bits.append("prefilter_fail")

        # 1) Postprocess via normalizers (clean + success flag)
        clean, sanitized = postprocess_field(cand, field_type)
        if sanitized:
            score += 2.0
            note_bits.append("sanitized_ok")

        # 2) Validators on the CLEAN string
        if field_type == "amount" and is_valid_amount(clean):
            score += 1.5
            note_bits.append("amount_valid")
        elif field_type == "time" and is_valid_time(clean):
            score += 1.5
            note_bits.append("time_valid")

        # 3) Gentle distance penalty between raw and clean
        dist = _levenshtein(_normalize_text(cand), clean)
        score -= 0.05 * dist
        if dist:
            note_bits.append(f"lev-{dist}")

        rescored.append((cand, score, ",".join(note_bits) if note_bits else "ok"))

    # Pick best by score; if tie, prefer shorter normalized form
    rescored.sort(key=lambda t: (t[1], -len(_normalize_text(t[0]))), reverse=True)
    best_raw = rescored[0][0]
    return best_raw, rescored
