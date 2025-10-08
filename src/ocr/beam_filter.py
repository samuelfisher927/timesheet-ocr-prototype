# src/ocr/beam_filter.py
from __future__ import annotations
from typing import List, Tuple, Literal
import math, re

from .postprocess import postprocess_field, is_valid_amount, is_valid_time

FieldType = Literal["amount", "time", "text"]

AMOUNT_PREFILTER_RX = re.compile(r"^[()\-$\s\d.,]+$")
TIME_PREFILTER_RX   = re.compile(r"^[\s0-9:;.,\-apmAPM]+$")

def _levenshtein(a: str, b: str) -> int:
    """Tiny Levenshtein to gently penalize messy outputs."""
    if a == b: return 0
    if not a:  return len(b)
    if not b:  return len(a)
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = prev[j] + 1
            dele = curr[j-1] + 1
            sub = prev[j-1] + (ca != cb)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]

def _prefilter_ok(s: str, field_type: FieldType) -> bool:
    if field_type == "amount":
        return bool(AMOUNT_PREFILTER_RX.match(s))
    if field_type == "time":
        return bool(TIME_PREFILTER_RX.match(s))
    return True

def rescore_candidates(
    candidates: List[str],
    seq_logprobs: List[float],
    field_type: FieldType
) -> Tuple[str, List[Tuple[str, float, str]]]:
    """
    Return best string + debug details [(cand, score, note), ...].
    Score = model_logprob + bonuses (sanitizer/regex) - small penalties (edit dist).
    """
    assert len(candidates) == len(seq_logprobs)
    rescored: List[Tuple[str, float, str]] = []

    for cand, logp in zip(candidates, seq_logprobs):
        score = float(logp) if math.isfinite(logp) else -1e9
        note_bits = []

        # 0) prefilter: heavily penalize illegal alphabet for the field
        if not _prefilter_ok(cand, field_type):
            score -= 2.0
            note_bits.append("prefilter_fail")

        # 1) run sanitizer + validators (uses your normalizers)
        clean, sanitized = postprocess_field(cand, field_type)
        if sanitized:
            score += 2.0
            note_bits.append("sanitized_ok")

        # 2) regex-valid AFTER sanitize
        if field_type == "amount" and is_valid_amount(clean):
            score += 1.5
            note_bits.append("amount_valid")
        elif field_type == "time" and is_valid_time(clean):
            score += 1.5
            note_bits.append("time_valid")

        # 3) gentle edit distance penalty between raw and clean
        #    (smaller distance => less penalty)
        dist = _levenshtein(cand, clean)
        score -= 0.05 * dist
        if dist:
            note_bits.append(f"lev-{dist}")

        rescored.append((cand, score, ",".join(note_bits) if note_bits else "ok"))

    # pick best by score; if tie, prefer the shortest clean form
    rescored.sort(key=lambda t: t[1], reverse=True)
    best_raw = rescored[0][0]
    return best_raw, rescored
