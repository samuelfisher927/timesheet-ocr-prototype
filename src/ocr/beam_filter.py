# src/ocr/beam_filter.py
from __future__ import annotations
from typing import List, Tuple, Literal
import math
import re

# Normalizers
from src.ocr.normalizers import sanitize_amount, sanitize_time, sanitize_date

FieldType = Literal["amount", "time", "text", "date"]

# Character prefilters to down-weight obviously wrong beams
AMOUNT_PREFILTER_RX = re.compile(r"^[()\-$\s\d.,]+$")
TIME_PREFILTER_RX   = re.compile(r"^[\s0-9:;.,\-apmAPM]+$")
DATE_PREFILTER_RX   = re.compile(r"^[\s0-9/.\-]+$")  # e.g., 01/23/2025, 1-2-25, 1.2.2025

def is_valid_amount(s: str) -> bool:
    return sanitize_amount(s) is not None

def is_valid_time(s: str) -> bool:
    return sanitize_time(s) is not None

def is_valid_date(s: str) -> bool:
    return sanitize_date(s) is not None

def _normalize_text(s: str) -> str:
    # collapse whitespace, strip edges
    return re.sub(r"\s+", " ", (s or "")).strip()

def postprocess_field(s: str, kind: FieldType) -> Tuple[str, bool]:
    """
    Normalize field text and report if sanitizer succeeded.
      - amount/time/date -> (sanitized, True) if valid else (whitespace-normalized, False)
      - text             -> (whitespace-normalized, True)  (no strict validation)
    """
    raw = _normalize_text(s)
    if kind == "amount":
        clean = sanitize_amount(raw)
        return (clean, True) if clean is not None else (raw, False)
    if kind == "time":
        clean = sanitize_time(raw)
        return (clean, True) if clean is not None else (raw, False)
    if kind == "date":
        clean = sanitize_date(raw)
        return (clean, True) if clean is not None else (raw, False)
    # text
    return raw, True

def _levenshtein(a: str, b: str) -> int:
    """Tiny Levenshtein to gently penalize messy outputs."""
    if a == b: return 0
    if not a:  return len(b)
    if not b:  return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins  = prev[j] + 1
            dele = curr[j - 1] + 1
            sub  = prev[j - 1] + (ca != cb)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]

def _prefilter_ok(s: str, field_type: FieldType) -> bool:
    if field_type == "amount":
        return bool(AMOUNT_PREFILTER_RX.match(s))
    if field_type == "time":
        return bool(TIME_PREFILTER_RX.match(s))
    if field_type == "date":
        return bool(DATE_PREFILTER_RX.match(s))
    return True  # text

def rescore_candidates(
    candidates: List[str],
    seq_logprobs: List[float],
    field_type: FieldType,
    length_bias: float = 0.0,   # small positive favors shorter strings; keep 0.0 to disable
) -> Tuple[str, List[Tuple[str, float, str]]]:
    """
    Return best string + debug details [(cand, score, note), ...].
    Score = model_logprob
            + bonus if sanitizer succeeds
            + bonus if validator passes (amount/time/date)
            - small penalty for edit distance (raw vs clean)
            - penalty if prefilter fails
            + (optional) short-length bias
    """
    assert len(candidates) == len(seq_logprobs)
    rescored: List[Tuple[str, float, str]] = []

    for cand, logp in zip(candidates, seq_logprobs):
        score = float(logp) if math.isfinite(logp) else -1e9
        note_bits = []

        raw_norm = _normalize_text(cand)

        # 0) prefilter characters
        if not _prefilter_ok(raw_norm, field_type):
            # a tad stronger on amounts (commonly polluted)
            score -= 3.0 if field_type == "amount" else 2.0
            note_bits.append("prefilter_fail")

        # 1) sanitize/normalize
        clean, sanitized = postprocess_field(raw_norm, field_type)
        if sanitized:
            score += 2.0
            note_bits.append("sanitized_ok")

        # 2) validators on clean
        if field_type == "amount" and is_valid_amount(clean):
            score += 1.5
            note_bits.append("amount_valid")
        elif field_type == "time" and is_valid_time(clean):
            score += 1.5
            note_bits.append("time_valid")
        elif field_type == "date" and is_valid_date(clean):
            score += 1.5
            note_bits.append("date_valid")

        # 3) gentle distance penalty (compare raw vs *clean*)
        dist = _levenshtein(raw_norm, clean)
        score -= 0.05 * dist
        if dist:
            note_bits.append(f"lev-{dist}")

        # 4) optional short-length bias
        if length_bias:
            score += length_bias * (-len(raw_norm))
            note_bits.append(f"len_bias({length_bias})")

        rescored.append((cand, score, ",".join(note_bits) if note_bits else "ok"))

    # pick best by (score desc, length asc)
    rescored.sort(key=lambda t: (t[1], -len(_normalize_text(t[0]))), reverse=True)
    best_raw = rescored[0][0]
    return best_raw, rescored
