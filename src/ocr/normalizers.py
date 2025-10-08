# src/ocr/normalizers.py
from __future__ import annotations
import re
from decimal import Decimal, InvalidOperation

def sanitize_amount(raw: str) -> str | None:
    """
    Normalize money-like strings to a canonical format '[-]###.##'.
    Returns None if it can't be parsed confidently.
    """
    if not raw:
        return None
    s = raw.strip()

    # Common OCR confusions and allowed chars
    s = s.replace("O", "0").replace("o", "0")     # O -> 0
    s = s.replace("—", "-").replace("–", "-")     # em/en dash -> hyphen
    s = re.sub(r"[^\d().,\-$]", "", s)            # keep only amount punctuation
    s = re.sub(r"\s+", "", s)
    s = s.replace(",,", ",").replace("..", ".")

    # Drop leading currency symbol for parsing
    s = s.lstrip("$")

    # If both '.' and ',' appear, treat ',' as thousands sep
    if "." in s and "," in s:
        s = s.replace(",", "")
    # If only commas and looks like cents, treat ',' as decimal point
    elif "," in s and re.search(r",\d{2}\)?$", s):
        s = s.replace(",", ".")
    # Remove stray trailing punctuation
    s = re.sub(r"[.,]$", "", s)

    # Remove any remaining thousands commas
    s = s.replace(",", "")

    # --- inside sanitize_amount after removing stray trailing punctuation ---
    # Parentheses mean negative; OCR often drops the closing ')'
    neg = s.startswith("(") or s.endswith(")")
    s = s.strip("()")

    # If we later see a leading '-', keep a single negative
    m = re.match(r"^-+", s)
    if m:
        neg = True
        s = s[len(m.group(0)):]

    try:
        val = Decimal(s)
        if neg:
            val = -val
        return f"{val:.2f}"
    except InvalidOperation:
        return None


def sanitize_time(raw: str) -> str | None:
    """
    Normalize time-like strings to 24h 'HH:MM'. Accepts separators ; , . - and am/pm.
    Returns None if it can't be parsed confidently.
    """
    if not raw:
        return None
    s = raw.strip().lower()
    s = s.replace("a.m.", "am").replace("p.m.", "pm").replace("a.m", "am").replace("p.m", "pm")
    s = s.replace(";", ":").replace(",", ":").replace(".", ":").replace("-", ":")
    s = re.sub(r"\s+", "", s)

    m = re.match(r"^([0-2]?\d):([0-5]\d)(am|pm)?$", s)
    if not m:
        return None

    hh = int(m.group(1))
    mm = int(m.group(2))
    ampm = m.group(3)

    if ampm:
        if ampm == "pm" and hh < 12:
            hh += 12
        if ampm == "am" and hh == 12:
            hh = 0

    if 0 <= hh <= 23 and 0 <= mm <= 59:
        return f"{hh:02d}:{mm:02d}"
    return None
