# src/tools/smoke_normalizers.py
from src.ocr.normalizers import sanitize_amount, sanitize_time

amount_samples = [
    ",093.45.", "114.30.", "(150,000.", "166.52.", ",093.45",
    "(150.059.", ",118.18.", "(200.02.", "120.32.", "114.23."
]
time_samples = [
    "9;30", "12.05pm", "7-45 AM", "18,00", "12:00 am", "00.15"
]

print("Amounts:")
for s in amount_samples:
    print(f"{s!r} -> {sanitize_amount(s)!r}")

print("\nTimes:")
for s in time_samples:
    print(f"{s!r} -> {sanitize_time(s)!r}")
