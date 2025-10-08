# src/tools/test_normalize_amount.py
from src.ocr.normalizers import sanitize_amount

cases = {
    # Trailing junk / commas vs dots
    ",093.45.": "93.45",
    "114.30.": "114.30",
    "166.52.": "166.52",
    ",093.45": "93.45",
    "114.23.": "114.23",
    # Parentheses (missing ')', trailing dot/comma)
    "(150,000.": "-150000.00",
    "(200.02.": "-200.02",
    "(150.059.": "-150.06",   # rounding via Decimal
    "(118.18,": "-118.18",
    # Currency symbols, dashes, weird chars
    "$1,234.50": "1234.50",
    "—1,234.50": "-1234.50",
    "--123.45": "-123.45",
    "O0.50": "0.50",  # O→0; canonical formatting trims extra leading zero
    # Comma as decimal
    "12,34": "12.34",
    "1,234,56": None,  # ambiguous; we expect None (or you can choose to coerce)
    # No digits / nonsense
    "....": None,
    "(": None,
    "-": None,
    # Big values
    "(999,999.99)": "-999999.99",
    "9,999,999.00": "9999999.00",
}

failures = 0
for raw, expected in cases.items():
    got = sanitize_amount(raw)
    ok = (got == expected)
    print(f"{raw!r:>15} -> {got!r:>12}  {'OK' if ok else f'EXPECTED {expected!r}'}")
    if not ok:
        failures += 1

print("\nRESULT:", "PASS ✅" if failures == 0 else f"FAIL ❌  ({failures} mismatches)")
exit(1 if failures else 0)
