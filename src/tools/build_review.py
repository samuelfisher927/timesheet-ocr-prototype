# src/tools/build_review.py
from __future__ import annotations
import os, json, math, csv, base64, io
from typing import Literal, List, Tuple, Dict
from PIL import Image, ImageOps, ImageDraw, ImageFont

from src.ocr.normalizers import sanitize_amount, sanitize_time

FieldType = Literal["time","amount","text"]

def is_valid_time(v: str) -> bool:
    import re
    return bool(re.match(r"^(?:[01]\d|2[0-3]):[0-5]\d$", v))

def is_valid_amount(v: str) -> bool:
    import re
    return bool(re.match(r"^-?\d+(?:\.\d{2})$", v))

def postprocess_field(raw: str, field_type: FieldType) -> Tuple[str, bool, bool]:
    if field_type == "time":
        clean = sanitize_time(raw)
        return (clean if clean else raw, bool(clean), bool(clean) and is_valid_time(clean))
    if field_type == "amount":
        clean = sanitize_amount(raw)
        return (clean if clean else raw, bool(clean), bool(clean) and is_valid_amount(clean))
    # text passthrough
    return (raw.strip(), False, True if raw.strip() else False)

def softmax_from_logprobs(logps: List[float]) -> List[float]:
    m = max(logps) if logps else 0.0
    exps = [math.exp(lp - m) for lp in logps]
    s = sum(exps) or 1.0
    return [e/s for e in exps]

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

def thumb(path: str, max_w: int = 380, max_h: int = 120) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = ImageOps.contain(img, (max_w, max_h))
    # add thin border
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0,0),(img.width-1,img.height-1)], outline=(200,200,200))
    return img

def to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def build_review(
    crops_root: str,
    infer_jsonl: str,
    out_csv: str,
    out_html: str,
    field_type: FieldType = "time",
    conf_threshold: float = 0.60,
):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)

    data = load_jsonl(infer_jsonl)
    # Optional: if future runs include beams/scores, use them; otherwise degrade gracefully
    # Expect keys:
    # - "pred_text" (required)
    # - optional: "candidates": List[str], "candidate_logprobs": List[float]
    rows_csv = []
    cards_html = []

    for r in data:
        crop_path = r["crop_path"]
        pred_raw  = (r.get("pred_time") or r.get("pred_text") or "").strip()

        # Confidence: use beams if present, else 1.0 for the only string
        cands = r.get("candidates") or [pred_raw]
        logs  = r.get("candidate_logprobs") or [0.0] * len(cands)
        probs = softmax_from_logprobs(logs)
        # Assume best is first; if not, pick max prob’s index
        best_idx = 0 if len(cands) == 1 else int(max(range(len(cands)), key=lambda i: probs[i]))
        model_conf = probs[best_idx]

        clean, sanitized, valid = postprocess_field(pred_raw, field_type)

        # Small confidence bumps for clean & valid
        bonus = (0.15 if sanitized else 0.0) + (0.15 if valid else 0.0)
        final_conf = max(0.0, min(0.99, model_conf + bonus))

        needs_review = int((final_conf < conf_threshold) or (not valid))

        # CSV row
        rows_csv.append({
            "crop_path": crop_path,
            "pred_raw": pred_raw,
            "clean": clean,
            "sanitized": int(sanitized),
            "valid": int(valid),
            "model_conf": round(model_conf, 4),
            "final_conf": round(final_conf, 4),
            "needs_review": needs_review,
        })

        # HTML card
        try:
            img = thumb(crop_path)
            img_uri = to_data_uri(img)
        except Exception:
            img_uri = ""
        tag = "⚠️ Review" if needs_review else "✅ OK"
        cards_html.append(f"""
<div class="card {'need' if needs_review else 'ok'}">
  <img src="{img_uri}" />
  <div class="meta"><code>{os.path.basename(crop_path)}</code></div>
  <div class="pred">raw: <b>{pred_raw}</b><br/>clean: <b>{clean}</b></div>
  <div class="conf">conf: {final_conf:.2f} ({'sanitized' if sanitized else 'raw'} | {'valid' if valid else 'invalid'})</div>
  <div class="flag">{tag}</div>
</div>""")

    # Write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()) if rows_csv else [])
        w.writeheader()
        for row in rows_csv:
            w.writerow(row)

    # Write HTML gallery
    html = f"""<!doctype html>
<html><head>
<meta charset="utf-8"/>
<title>OCR Review</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, Segoe UI, Roboto, Helvetica, Arial; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 12px; }}
.card {{ border: 1px solid #ddd; border-radius: 10px; padding: 10px; }}
.card.need {{ border-color: #e99; background:#fff7f7; }}
.card.ok {{ border-color: #9e9; background:#f7fff7; }}
.card img {{ width: 100%; height: auto; border-radius: 6px; }}
.meta {{ color:#666; font-size: 12px; margin: 6px 0; }}
.pred {{ font-size: 14px; }}
.conf {{ color:#333; font-size: 13px; margin-top:4px; }}
.flag {{ margin-top:6px; font-weight:600; }}
</style>
</head>
<body>
<h2>OCR Review — field_type={field_type} — threshold={conf_threshold}</h2>
<div class="grid">
{''.join(cards_html)}
</div>
</body></html>"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops_root", required=True, help="Root folder containing the crops used for inference")
    ap.add_argument("--infer_jsonl", required=True, help="Path to JSONL produced by trocr_time_infer")
    ap.add_argument("--out_csv", default="exports/review_panel.csv")
    ap.add_argument("--out_html", default="exports/review_panel.html")
    ap.add_argument("--field_type", choices=["time","amount","text"], default="time")
    ap.add_argument("--conf_threshold", type=float, default=0.60)
    args = ap.parse_args()
    build_review(args.crops_root, args.infer_jsonl, args.out_csv, args.out_html, args.field_type, args.conf_threshold)
