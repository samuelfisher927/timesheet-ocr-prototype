from __future__ import annotations
import os, json, csv, time
from typing import List, Dict
from flask import Flask, request, send_from_directory, jsonify, Response

# === CONFIG ===
INFER_JSONL = os.environ.get("INFER_JSONL", "exports/ocr_time_results_preprocessed.jsonl")
CROPS_ROOT  = os.environ.get("CROPS_ROOT", "exports/preprocessed")
FIELD_TYPE  = os.environ.get("FIELD_TYPE", "time")  # time|amount|text
LABEL_OUT   = os.environ.get("LABEL_OUT", f"exports/labels_{FIELD_TYPE}.csv")

# === APP ===
app = Flask(__name__, static_url_path="")

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

DATA = load_jsonl(INFER_JSONL)
for r in DATA:
    # unify key for predicted text
    r["pred_raw"] = (r.get("pred_time") or r.get("pred_text") or "").strip()

os.makedirs(os.path.dirname(LABEL_OUT) or ".", exist_ok=True)
if not os.path.exists(LABEL_OUT):
    with open(LABEL_OUT, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["crop_path","field_type","pred_raw","label","reviewer","ts"]).writeheader()

@app.route("/")
def index():
    return Response("""
<!doctype html><meta charset="utf-8">
<title>OCR Click-to-Correct</title>
<style>
body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial;margin:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}
.card{border:1px solid #ddd;border-radius:10px;padding:10px}
.card img{max-width:100%;border-radius:6px}
.row{display:flex;gap:8px;align-items:center;margin-top:6px}
.small{color:#666;font-size:12px}
.btn{padding:6px 10px;border:1px solid #888;border-radius:8px;background:#f5f5f5;cursor:pointer}
.btn:active{transform:translateY(1px)}
input[type=text]{width:100%;padding:6px 8px;border:1px solid #bbb;border-radius:8px}
.tags{font-size:12px;color:#555}
kbd{border:1px solid #bbb;border-bottom-width:2px;border-radius:4px;padding:0 6px;margin:0 2px}
</style>
<h2>OCR Click-to-Correct (<span id="count"></span>) — Tip: <kbd>Enter</kbd> save, <kbd>Ctrl</kbd>+<kbd>→</kbd> next</h2>
<div class="grid" id="grid"></div>
<script>
async function fetchData(){const r=await fetch('/data');return r.json()}
async function saveOne(item, label){
  const body={crop_path:item.crop_path, field_type:item.field_type, pred_raw:item.pred_raw, label}
  const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
  return r.ok
}
function cardHTML(item){
  const fn=item.crop_path.split(/[\\\\/]/).pop()
  return `
  <div class="card" data-id="${item.idx}">
    <img src="/crop?path=${encodeURIComponent(item.crop_path)}">
    <div class="small">${fn}</div>
    <div class="tags">pred: <b>${item.pred_raw || "(empty)"}</b> · type: ${item.field_type}</div>
    <div class="row"><input type="text" value="${item.suggest || ""}" placeholder="Type correct value (e.g., 07:45)"></div>
    <div class="row">
      <button class="btn save">Save</button>
      <button class="btn skip">Skip</button>
    </div>
  </div>`
}
function wire(card, item, focusFirst){
  const inp=card.querySelector('input'); const saveBtn=card.querySelector('.save'); const skipBtn=card.querySelector('.skip');
  async function doSave(){ if(!inp.value.trim()) return; const ok=await saveOne(item, inp.value.trim()); if(ok){ card.remove(); updateCount(); } }
  saveBtn.onclick=doSave; skipBtn.onclick=()=>{card.remove(); updateCount();}
  inp.addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ doSave() } })
  if(focusFirst) inp.focus()
}
function updateCount(){ document.getElementById('count').textContent = document.querySelectorAll('.card').length + " remaining" }
(async ()=>{
  const data=await fetchData(); const grid=document.getElementById('grid');
  data.items.forEach((it,i)=>{ const tmp=document.createElement('div'); tmp.innerHTML=cardHTML(it).trim(); const card=tmp.firstChild; grid.appendChild(card); wire(card,it,i===0) })
  updateCount();
  window.addEventListener('keydown',e=>{ if(e.ctrlKey && e.key==='ArrowRight'){ const first=document.querySelector('.card'); if(first){ first.remove(); updateCount(); } }})
})();
</script>""", mimetype="text/html")

@app.route("/data")
def data():
    # Prepare a lightweight list for client
    items=[]
    for idx, r in enumerate(DATA):
        items.append({
            "idx": idx,
            "crop_path": r["crop_path"],
            "field_type": FIELD_TYPE,
            "pred_raw": r.get("pred_raw",""),
            "suggest": r.get("pred_raw",""),  # prefill; you can later prefill with sanitizer output
        })
    return jsonify({"items": items})

@app.route("/crop")
def crop():
    # Serve images by path (under CROPS_ROOT for safety)
    p = request.args.get("path","")
    abspath = os.path.abspath(p)
    root = os.path.abspath(CROPS_ROOT)
    if not abspath.startswith(root):
        return "forbidden", 403
    return send_from_directory(os.path.dirname(abspath), os.path.basename(abspath))

@app.route("/save", methods=["POST"])
def save():
    data = request.get_json(force=True)
    row = {
        "crop_path": data["crop_path"],
        "field_type": data["field_type"],
        "pred_raw": data.get("pred_raw",""),
        "label": data.get("label",""),
        "reviewer": os.environ.get("REVIEWER","sam"),
        "ts": int(time.time())
    }
    with open(LABEL_OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["crop_path","field_type","pred_raw","label","reviewer","ts"])
        w.writerow(row)
    return jsonify({"ok": True})
