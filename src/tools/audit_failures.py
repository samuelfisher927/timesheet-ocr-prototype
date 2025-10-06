# -*- coding: utf-8 -*-
"""
Audit failures from the time_fields_long CSV produced by csv_writer.py.

Usage:
  # Auto-pick latest time_fields_long_*.csv in ./exports
  python -m src.tools.audit_failures --exports ./exports

  # Or specify a particular CSV
  python -m src.tools.audit_failures --csv ./exports/time_fields_long_20251006_161346.csv

Options:
  --include-low-conf     Include rows flagged only for low confidence (even if IsValid==True)
  --copy-crops           Copy crop images for failing rows into a review folder
"""

from __future__ import annotations
import argparse, re, shutil
from pathlib import Path
from datetime import datetime
import pandas as pd

# --- paste the SAME preclean function used in csv_writer.py ---
def preclean_time_text(s):
    s = str(s).strip()
    s = s.replace("O", "0").replace("o", "0").replace("S", "5")
    s = re.sub(r"(?<=\d)[\./\\](?=\d)", ":", s)  # dots/slashes/backslashes between digits → ':'
    s = s.replace(",", ":")
    s = re.sub(r"[^0-9apmAPM: ]+", "", s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r":{2,}", ":", s).lstrip(":").rstrip(":")
    if s.count(":") >= 2:
        parts = [p for p in s.split(":") if p != ""]
        if parts:
            s = f"{parts[0]}:{parts[-1]}"
    m = re.fullmatch(r"(\d{1,2}):(\d{2,})", s)
    if m:
        h, mins = m.group(1), m.group(2)
        s = f"{h}:{mins[-2:]}"
    m = re.fullmatch(r"(\d{3,5})", s)
    if m:
        d = m.group(1)
        if   len(d) == 3: s = f"{d[0]}:{d[1:]}"
        elif len(d) == 4: s = f"{d[:2]}:{d[2:]}"
        elif len(d) == 5: s = f"{d[:2]}:{d[-2:]}"
    return s

def _latest_csv(exports_dir: Path) -> Path | None:
    cands = sorted(exports_dir.glob("time_fields_long_*.csv"))
    return cands[-1] if cands else None

def _split_warnings(w):
    if pd.isna(w) or w is None:
        return []
    # Warnings column was joined with "; "
    return [tok.strip() for tok in str(w).split(";") if tok.strip()]

def main():
    ap = argparse.ArgumentParser(description="Audit failures from time_fields_long CSV")
    ap.add_argument("--exports", type=str, help="Folder containing time_fields_long_*.csv")
    ap.add_argument("--csv", type=str, help="Specific time_fields_long CSV to audit")
    ap.add_argument("--include-low-conf", action="store_true", help="Also include rows only flagged for low confidence")
    ap.add_argument("--copy-crops", action="store_true", help="Copy crop images for failing rows into a review folder")
    args = ap.parse_args()

    if not args.csv and not args.exports:
        ap.error("Provide either --csv <path> or --exports <dir>")

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
    else:
        exports_dir = Path(args.exports)
        latest = _latest_csv(exports_dir)
        if not latest:
            raise FileNotFoundError(f"No time_fields_long_*.csv found in {exports_dir}")
        csv_path = latest

    df = pd.read_csv(csv_path)
    # Normalize warnings to list
    df["WarningsList"] = df["Warnings"].apply(_split_warnings)

    # Failure criteria
    invalid_mask = ~(df.get("IsValid", False).astype(bool))
    if args.include_low_conf:
        low_conf_mask = df["WarningsList"].apply(lambda L: any(w.startswith("low_conf<") for w in L))
        fail_mask = invalid_mask | low_conf_mask
    else:
        fail_mask = invalid_mask

    failures = df[fail_mask].copy()
    if failures.empty:
        print(f"No failures found in {csv_path.name}")
        return

    # Compute cleaned candidate (to see what preclean would produce)
    failures["PrecleanedCandidate"] = failures["RawText"].apply(preclean_time_text)

    # Extract a single "Reason" token (first warning if present)
    def first_reason(L):
        return L[0] if L else ""
    failures["Reason"] = failures["WarningsList"].apply(first_reason)

    # Reorder columns for readability
    cols = [
        "TimesheetID","Page","Row","Col","FieldKey",
        "RawText","PrecleanedCandidate","NormalizedTime","Confidence","Reason",
        "CropPath","_source_file","Warnings"
    ]
    cols = [c for c in cols if c in failures.columns]
    failures = failures[cols]

    # Write review CSV
    out_dir = Path(csv_path).parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"failure_review_{ts}.csv"
    failures.to_csv(out_csv, index=False)
    print(str(out_csv))

    # Print quick summaries
    print("\nTop reasons:")
    print(failures["Reason"].value_counts().head(10).to_string())

    print("\nMost common raw patterns:")
    print(failures["RawText"].value_counts().head(15).to_string())

    # Optionally copy crops to folder for visual audit
    if args.copy_crops:
        review_dir = out_dir / f"failure_crops_{ts}"
        review_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for p in failures.get("CropPath", []):
            if pd.isna(p) or not p:
                continue
            src = Path(str(p))
            # If relative, try relative to repo root (two levels up from exports)
            if not src.is_file():
                try2 = (out_dir.parent / src).resolve()
                if try2.is_file():
                    src = try2
            if src.is_file():
                dst = review_dir / src.name
                try:
                    shutil.copy2(src, dst)
                    copied += 1
                except Exception:
                    pass
        print(f"\nCopied {copied} crop images to {review_dir} (if paths existed).")

if __name__ == "__main__":
    main()