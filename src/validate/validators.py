# src/validate/validators.py
import re
from typing import Optional

# Accept many separators / clutter, then repair to HH:MM.
SEP_CHARS   = r"[:\.\-\/hH]"      # map these to ":"
NOISE_CHARS = r"[^\d:apmAPM]"     # keep digits, colon, and am/pm for parsing

AMPM_RE = re.compile(r"^\s*(\d{1,2})[:hH]?(\d{2})\s*([aApP][mM])?\s*$")

def repair_hhmm(s: str) -> Optional[str]:
    """
    Repair common OCR artifacts to a clean HH:MM (24h). Handles junk like:
      (13.30. -> 13:30, 114.30. -> 14:30, "17.19.. -> 17:19, 1/1.30. -> 11:30
    Also preserves an optional AM/PM suffix if present.
    """
    if not s:
        return None
    s = s.strip()

    # Extract potential AM/PM tag (we'll re-attach after cleaning)
    ampm = None
    m_ampm = re.search(r"([aApP]\s*[mM])", s)
    if m_ampm:
        ampm = m_ampm.group(1).replace(" ", "").lower()  # 'am' or 'pm'

    # 1) Map separators to colon
    s = re.sub(SEP_CHARS, ":", s)

    # 2) Remove noise except digits/colon and potential am/pm
    s = re.sub(NOISE_CHARS, "", s)

    # Collapse to a single colon around the first occurrence
    if s.count(":") > 1:
        first = s.find(":")
        left  = re.sub(r"\D", "", s[:first])
        right = re.sub(r"\D", "", s[first+1:])
        s = f"{left}:{right}"

    # If no colon, try to insert one from digits
    if ":" not in s:
        digits = re.sub(r"\D", "", s)
        if len(digits) == 3:      # 930 -> 9:30
            s = f"{digits[0]}:{digits[1:]}"
        elif len(digits) >= 4:    # 0930 / 1130 / 11430...
            s = f"{digits[:-2]}:{digits[-2:]}"
        else:
            return None

    # Enforce HH:MM (with optional am/pm)
    m = re.match(r"^\D*(\d{1,3}):(\d{2})\D*$", s)
    if not m:
        return None
    hh, mm = m.group(1), m.group(2)

    # Heuristic: drop spurious leading '1' in a 3-digit hour like '114:30' -> '14:30'
    if len(hh) == 3 and hh.startswith("1"):
        hh = hh[1:]

    # Convert to int and range-check
    try:
        h = int(hh)
        m_ = int(mm)
    except ValueError:
        return None
    if not (0 <= m_ <= 59):
        return None

    # If AM/PM present, convert to 24h
    if ampm is not None:
        am = (ampm == "am")
        if h == 12:
            h = 0 if am else 12
        elif not am:
            h = h + 12 if h <= 11 else h
        if not (0 <= h <= 23):
            return None
        return f"{h:02d}:{m_:02d}"

    # No AM/PM → accept 0–23
    if not (0 <= h <= 23):
        return None
    return f"{h:02d}:{m_:02d}"

def normalize_hhmm(s: str) -> Optional[str]:
    """Repair then validate; returns 24h HH:MM or None."""
    # Try robust repair first
    out = repair_hhmm(s)
    if out is not None:
        return out
    # Strict fallback (accepts 9:00 / 09:00 / 09h00 with optional am/pm)
    m = AMPM_RE.match(s or "")
    if not m:
        return None
    h, mm, ampm = int(m.group(1)), int(m.group(2)), (m.group(3).lower() if m.group(3) else None)
    if not (0 <= mm <= 59):
        return None
    if ampm:
        am = (ampm == "am")
        if h == 12:
            h = 0 if am else 12
        elif not am:
            h = h + 12 if h <= 11 else h
    if not (0 <= h <= 23):
        return None
    return f"{h:02d}:{mm:02d}"

def hhmm_to_minutes(s: str) -> Optional[int]:
    n = normalize_hhmm(s)
    if n is None:
        return None
    hh, mm = n.split(":")
    return int(hh) * 60 + int(mm)

def minutes_to_hhmm(m: int) -> str:
    h, r = divmod(m, 60)
    return f"{h:02d}:{r:02d}"
