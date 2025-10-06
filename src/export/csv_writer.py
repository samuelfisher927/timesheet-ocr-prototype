# -*- coding: utf-8 -*-
"""
csv_writer.py  (CSV export for timesheet OCR)

Loads OCR JSON results, normalizes/validates time fields, and writes tidy CSVs:
  - time_fields_long.csv  (one row per recognized field)
  - summary.csv           (per TimesheetID stats)

CLI:
    python -m src.export.csv_writer --in ./outputs --out ./exports --conf-thresh 0.35

Programmatic:
    from src.export.csv_writer import run_export
    run_export("./outputs", "./exports", conf_thresh=0.35)
"""

from __future__ import annotations
import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------- Regex helpers ----------
RE_ROWCOL = re.compile(r"(?:row[_\- ]?(?P<row>\d+))|(?:col[_\- ]?(?P<col>\d+))", re.I)
RE_HHMM = re.compile(
    r"""^\s*
    (?P<h>\d{1,2})      # hour 1-2 digits
    (?:
        \s*[:\.\- ]\s*  # optional separator : . - space
        (?P<m>\d{1,2})  # minutes 1-2 digits
    )?
    \s*(?P<ampm>am|pm|a|p|a\.m\.|p\.m\.|AM|PM|A|P)?\s*$""",
    re.X,
)


# ---------- Utility ----------
def _coalesce(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default


def _infer_row_col_from_field_id(field_id: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if not field_id:
        return None, None
    row = col = None
    for m in RE_ROWCOL.finditer(field_id):
        gd = m.groupdict()
        if gd.get("row") and row is None:
            row = int(gd["row"])
        if gd.get("col") and col is None:
            col = int(gd["col"])
    return row, col


def _infer_row_col_from_path(path: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if not path:
        return None, None
    row = col = None
    for part in str(path).replace("\\", "/").split("/"):
        for m in RE_ROWCOL.finditer(part):
            gd = m.groupdict()
            if gd.get("row") and row is None:
                row = int(gd["row"])
            if gd.get("col") and col is None:
                col = int(gd["col"])
    return row, col


# ---------- Time parsing ----------
def preclean_time_text(s: Any) -> str:
    """
    Clean noisy OCR time strings (e.g., ',09130', '(07.30.', '1/1.30.', '2002.')
    into a normalized form like '09:30', '07:30', '11:30', '20:02' before regex parse.
    """
    s = str(s).strip()

    # Common OCR confusions
    s = s.replace("O", "0").replace("o", "0").replace("S", "5")

    # Normalize separators to colons when they are BETWEEN digits
    s = re.sub(r"(?<=\d)[\./\\](?=\d)", ":", s)  # dots/slashes/backslashes between digits → ':'
    s = s.replace(",", ":")                       # commas often used like colons

    # Remove junk (keep digits, am/pm letters, colon, space)
    s = re.sub(r"[^0-9apmAPM: ]+", "", s)
    s = re.sub(r"\s+", "", s)

    # Collapse repeated colons and trim leading/trailing colons
    s = re.sub(r":{2,}", ":", s).lstrip(":").rstrip(":")

    # If there are 3+ chunks split by colon (e.g., '2:1:01'),
    # keep the first as hour and the LAST as minutes (drop middle noise).
    if s.count(":") >= 2:
        parts = [p for p in s.split(":") if p != ""]
        if parts:
            h, m = parts[0], parts[-1]
            s = f"{h}:{m}"

    # If colon exists and minutes have >2 digits (e.g., '1:000' or '11:300'),
    # keep the last two as minutes.
    m = re.fullmatch(r"(\d{1,2}):(\d{2,})", s)
    if m:
        h, mins = m.group(1), m.group(2)
        s = f"{h}:{mins[-2:]}"

    # Handle condensed numeric forms of 3–5 digits:
    # 930  -> 9:30
    # 1130 -> 11:30
    # 09130 -> 09:30  (drop middle artifact)
    m = re.fullmatch(r"(\d{3,5})", s)
    if m:
        d = m.group(1)
        if   len(d) == 3: s = f"{d[0]}:{d[1:]}"
        elif len(d) == 4: s = f"{d[:2]}:{d[2:]}"
        elif len(d) == 5: s = f"{d[:2]}:{d[-2:]}"
    return s

def parse_time_string(raw: Any) -> Tuple[Optional[str], List[str]]:
    """
    Parse simple time strings into HH:MM (24h if AM/PM present; otherwise keep 0–23 if hour>12).
    Returns (normalized_time, warnings).
    """
    warnings: List[str] = []
    if raw is None:
        return None, ["empty"]

    s = preclean_time_text(raw)
    if s == "" or s == ":":
        return None, [f"unparsable('{raw}')"]

    m = RE_HHMM.match(s)
    if not m:
        return None, [f"unparsable('{raw}')"]

    h = int(m.group("h"))
    m_str = m.group("m")
    ampm = (m.group("ampm") or "").lower().replace(".", "")
    minutes = int(m_str) if m_str is not None else 0

    if minutes > 59:
        warnings.append("minutes>59")
        minutes = 59  # clamp

    if ampm in {"am", "a"}:
        hour24 = 0 if h == 12 else h
    elif ampm in {"pm", "p"}:
        hour24 = 12 if h == 12 else h + 12
    else:
        if h > 23:
            warnings.append("hour>23")
            hour24 = h % 24
        else:
            hour24 = h

    return f"{hour24:02d}:{minutes:02d}", warnings


# ---------- Load / Normalize / Validate ----------
def _read_records_from_path(f: Path) -> List[Dict[str, Any]]:
    """Return a list[dict] from .json or .jsonl. On error, return a single row with a json_error warning."""
    try:
        if f.suffix.lower() == ".jsonl":
            items: List[Dict[str, Any]] = []
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                    # Normalize to list-of-dicts
                    if isinstance(rec, list):
                        items.extend(rec)
                    elif isinstance(rec, dict) and "results" in rec and isinstance(rec["results"], list):
                        items.extend(rec["results"])
                    elif isinstance(rec, dict):
                        items.append(rec)
                    else:
                        items.append({"_raw": rec})
                except Exception as e:
                    items.append({"Warnings": [f"jsonl_line_error:{i+1}:{e}"]})
            return items
        else:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
                return data["results"]
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return list(data.values())
            return [{"_raw": data}]
    except Exception as e:
        return [{"Warnings": [f"json_error:{e}"]}]
    
def pick(d: dict, *names, default=None):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default

def infer_fieldkey_from_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    m = re.search(r"(row\d+_clock_(in|out)_(am|pm))", str(path).replace("\\", "/"), re.I)
    if m:
        return m.group(1).lower()
    return None



def load_results(inputs: List[str] | str) -> pd.DataFrame:
    """
    Load one or many JSON/JSONL files (or a directory containing them) and return a normalized DataFrame.
    """
    files: List[Path] = []
    if isinstance(inputs, str):
        p = Path(inputs)
        if p.is_dir():
            files = sorted(p.rglob("*.json")) + sorted(p.rglob("*.jsonl"))
        elif p.is_file():
            files = [p]
        else:
            cwd = Path.cwd()
            raise FileNotFoundError(f"No such file or directory: {inputs} (cwd: {cwd})")
    else:
        for it in inputs:
            p = Path(it)
            if p.is_dir():
                files += sorted(p.rglob("*.json")) + sorted(p.rglob("*.jsonl"))
            elif p.is_file():
                files.append(p)
            else:
                cwd = Path.cwd()
                raise FileNotFoundError(f"No such file or directory: {it} (cwd: {cwd})")

    rows: List[Dict[str, Any]] = []
    for f in files:
        items = _read_records_from_path(f)
        for it in items:

            ts_id     = pick(it, "timesheet_id", "timesheetId", default=f.parent.name)
            page      = pick(it, "page_id", "page")
            field_id  = pick(it, "field_id", "fieldId")
            row       = pick(it, "row")
            col       = pick(it, "col")
            field_key = pick(it, "field_key", "fieldKey")

            # accept pred_text and common alternates
            raw_text  = pick(
                it,
                "raw_text", "text", "rawText",
                "ocr_text", "ocrText",
                "pred_text", "prediction", "pred", "label", "value",
            )

            norm_time = pick(it, "normalized_time", "normalizedTime", "norm_time", "normTime")
            conf      = pick(it, "confidence", "conf", "score", "prob", "probability", "conf_score")
            crop_path = pick(it, "crop_path", "cropPath", "path", "image_path", "imagePath")

            if field_key is None:
                fk = infer_fieldkey_from_path(crop_path)
                if fk:
                    field_key = fk

            if row is None or col is None:
                r2, c2 = _infer_row_col_from_field_id(field_id)
                if row is None: row = r2
                if col is None: col = c2
                if (row is None or col is None) and crop_path:
                    r3, c3 = _infer_row_col_from_path(crop_path)
                    if row is None: row = r3
                    if col is None: col = c3

            rows.append({
                "TimesheetID": ts_id,
                "Page": page,
                "Row": row,
                "Col": col,
                "FieldKey": field_key,
                "RawText": raw_text,
                "NormalizedTime": norm_time,
                "Confidence": conf,
                "CropPath": crop_path,
                "Warnings": it.get("Warnings", []),
                "_source_file": str(f),
            })

    df = pd.DataFrame(rows)
    sort_cols = [c for c in ["TimesheetID", "Page", "Row", "Col"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable", na_position="last").reset_index(drop=True)
    return df


def normalize_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize df['RawText'] into df['NormalizedTime'] (HH:MM) and merge warnings.
    """
    norms: List[Optional[str]] = []
    warns_list: List[List[str]] = []

    for raw in df.get("RawText", [None] * len(df)):
        norm, warns = parse_time_string(raw)
        norms.append(norm)
        warns_list.append(warns)

    # Merge with existing warnings per-row (unique, keep order)
    def merge_warns(existing, new):
        a = existing if isinstance(existing, list) else ([] if pd.isna(existing) else [str(existing)])
        b = new if isinstance(new, list) else ([] if pd.isna(new) else [str(new)])
        seen = set()
        out = []
        for w in [*a, *b]:
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out

    if "Warnings" in df.columns:
        df["Warnings"] = [merge_warns(w, new) for w, new in zip(df["Warnings"], warns_list)]
    else:
        df["Warnings"] = warns_list

    df["NormalizedTime"] = norms
    return df


def validate(df: pd.DataFrame, confidence_threshold: float = 0.35) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add validation warnings and build a summary dataframe.
    - Flags: low confidence, invalid time
    """
    # Low confidence mask
    if "Confidence" in df.columns:
        low_conf = pd.to_numeric(df["Confidence"], errors="coerce") < confidence_threshold
        low_conf = low_conf.fillna(False)
    else:
        low_conf = pd.Series([False] * len(df))

    invalid_time = df["NormalizedTime"].isna()

    # Append warnings
    def add_warn(existing, newmsg, mask):
        out = []
        for w, m in zip(existing, mask):
            if m:
                if isinstance(w, list):
                    if newmsg not in w:
                        w = [*w, newmsg]
                    out.append(w)
                elif pd.isna(w):
                    out.append([newmsg])
                else:
                    out.append(list(dict.fromkeys([w, newmsg])))
            else:
                out.append(w if isinstance(w, list) or pd.isna(w) else [w])
        return out

    df["Warnings"] = add_warn(df["Warnings"], f"low_conf<{confidence_threshold}", low_conf)
    df["Warnings"] = add_warn(df["Warnings"], "invalid_time", invalid_time)

    df["IsValid"] = (~invalid_time).astype(bool)

    # Summary by TimesheetID
    def safe_mean(series: pd.Series) -> float:
        s = pd.to_numeric(series, errors="coerce")
        return float(s.mean()) if len(s) else float("nan")

    grp = df.groupby("TimesheetID", dropna=False)
    summary = pd.DataFrame({
        "Records": grp.size(),
        "ValidRecords": grp["IsValid"].sum(),
        "PctValid": grp["IsValid"].mean().round(4),
        "AvgConfidence": grp["Confidence"].apply(safe_mean).round(4),
    }).reset_index()

    return df, summary


# ---------- CSV writing ----------
def _warnings_to_str(w):
    if isinstance(w, list):
        return "; ".join(map(str, w))
    if pd.isna(w):
        return ""
    return str(w)


def write_csvs(df: pd.DataFrame, summary: pd.DataFrame, out_dir: str) -> Tuple[Path, Path]:
    """
    Writes two CSVs to out_dir:
      - time_fields_long.csv
      - summary.csv
    Returns (detail_path, summary_path).
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = out_dir_p / f"time_fields_long_{ts}.csv"
    summary_path = out_dir_p / f"summary_{ts}.csv"

    df_out = df.copy()
    df_out.insert(0, "RunID", ts)
    # Ensure Warnings is serializable as text
    df_out["Warnings"] = df_out["Warnings"].apply(_warnings_to_str)

    summary_out = summary.copy()
    summary_out.insert(0, "RunID", ts)

    df_out.to_csv(detail_path, index=False)
    summary_out.to_csv(summary_path, index=False)

    return detail_path, summary_path


# ---------- Public one-liner ----------
def run_export(inputs: List[str] | str, out_dir: str, conf_thresh: float = 0.35) -> Tuple[Path, Path]:
    """
    End-to-end: load -> normalize -> validate -> write CSVs.
    """
    df = load_results(inputs)
    df = normalize_times(df)
    df, summary = validate(df, confidence_threshold=conf_thresh)
    return write_csvs(df, summary, out_dir)


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Export OCR timesheet JSON results to CSVs.")
    ap.add_argument("--in", dest="inputs", required=True,
                    help="Path to a JSON file or a directory containing JSON files.")
    ap.add_argument("--out", dest="out_dir", default="./exports",
                    help="Directory to write CSVs (will be created).")
    ap.add_argument("--conf-thresh", dest="conf_thresh", type=float, default=0.35,
                    help="Confidence threshold for warnings (default 0.35).")
    args = ap.parse_args()

    detail_csv, summary_csv = run_export(args.inputs, args.out_dir, conf_thresh=args.conf_thresh)
    print(str(detail_csv))
    print(str(summary_csv))


if __name__ == "__main__":
    main()
