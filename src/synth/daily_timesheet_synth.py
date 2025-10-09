# src/synth/daily_timesheet_synth.py
# Generate full DAILY TIMESHEET pages (typed template + handwritten entries),
# plus per-cell crops and labels JSONL/CSV for your OCR pipeline.

import os, io, json, csv, random, argparse, pathlib
from dataclasses import dataclass
from typing import List, Tuple, Dict
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np

# ---------------- UI fonts ----------------
def load_ui_font(size=28):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "assets/fonts_ui/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()

# ---------------- page/layout constants ----------------
PAGE_SIZE = (1700, 2200)              # px
MARGIN_L, MARGIN_R = 80, 80
MARGIN_T, MARGIN_B = 240, 120         # extra top space so title never overlaps header boxes
HEADER_GAP = 90                       # gap below title before header boxes/table
HEADER_ROW_H = 120                    # header row height (same as data row height)
ROWS_DATA = 12                        # number of handwritten rows

# Table columns (3-tuple now)
COLS = [
    ("employee_name", "text",   260),
    ("in_am",         "time",   170),
    ("out_am",        "time",   170),
    ("lunch",         "time",   140),
    ("in_pm",         "time",   170),
    ("out_pm",        "time",   170),
    ("total_hours",   "amount", 170),
    ("signature",     "text",   210),
]

# Pretty printed labels for the header row (typed)
COL_LABELS: Dict[str, str] = {
    "employee_name": "Employee Name",
    "in_am":         "Clock In\n(AM)",
    "out_am":        "Clock Out\n(AM)",
    "lunch":         "Lunch",
    "in_pm":         "Clock In\n(PM)",
    "out_pm":        "Clock Out\n(PM)",
    "total_hours":   "Total\nHours",
    "signature":     "Signature",
}

# Header info boxes (typed labels; handwritten values inside boxes)
def header_positions():
    box_h = 50
    vgap  = 24

    # Shift Company to the right so its label isn't clipped
    x_company = MARGIN_L + 120
    y_company = MARGIN_T - 90

    # Supervisor goes directly below Company
    x_super   = x_company
    y_super   = y_company + box_h + vgap

    # Date stays on the right, aligned with Company row
    x_date    = PAGE_SIZE[0] - MARGIN_R - 260
    y_date    = y_company

    return {
        "company":    (x_company, y_company, 500, box_h),
        "supervisor": (x_super,   y_super,   480, box_h),
        "date":       (x_date,    y_date,    240, box_h),
    }

@dataclass
class FieldDef:
    name: str
    ftype: str
    box: Tuple[int,int,int,int]  # x,y,w,h

# ---------------- samplers ----------------
BUS_HOURS = list(range(7, 20))
MIN_BUCKET = [0, 5, 10, 15, 20, 30, 45]
EMP_NAMES = ["Sam Fisher","Alex Morgan","Riley Chen","Jordan Bell","Jamie Lee",
             "Taylor Brooks","Priya Shah","Diego Martinez","Dana Kim","Chris Park"]

def s_name(): return random.choice(EMP_NAMES)
def s_date():
    m = random.randint(1,12); d = random.randint(1,28); y = random.choice([2024, 2025])
    return f"{m:02d}/{d:02d}/{y}"
def s_time():
    hh = random.choice(BUS_HOURS); mm = random.choice(MIN_BUCKET)
    sep = random.choices([":", ".", ":"], weights=[0.85,0.1,0.05])[0]
    return f"{hh:02d}{sep}{mm:02d}"
def s_lunch():
    mm = random.choice([0,10,15,20,30,45,60])
    return f"{mm//60:02d}:{mm%60:02d}"
def s_hours():
    base = random.choice([7,7.5,8,8,8.5,9])
    return f"{base:.2f}".replace(".", random.choice([".", ".", ","]))

# ---------------- rendering utils ----------------
def ensure_dir(p): pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def load_fonts(fonts_dir: str, size_range=(24, 40)) -> List[ImageFont.FreeTypeFont]:
    fonts=[]
    if os.path.isdir(fonts_dir):
        for fname in os.listdir(fonts_dir):
            if fname.lower().endswith((".ttf",".otf")):
                fpath=os.path.join(fonts_dir,fname)
                for sz in range(size_range[0], size_range[1]+1, 2):
                    try: fonts.append(ImageFont.truetype(fpath, sz))
                    except: pass
    return fonts if fonts else [ImageFont.load_default()]

def pick_font(fonts): return random.choice(fonts)

def page_noise(img: Image.Image) -> Image.Image:
    # apply *after* everything is drawn so labels & handwriting move together
    angle = random.uniform(-2.0, 2.0)
    out = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255,255,255))
    if random.random() < 0.9:
        out = ImageEnhance.Contrast(out).enhance(random.uniform(0.85, 1.25))
    if random.random() < 0.6:
        out = ImageEnhance.Brightness(out).enhance(random.uniform(0.95, 1.10))
    if random.random() < 0.45:
        out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2)))
    if random.random() < 0.6:
        arr = np.array(out).astype(np.int16)
        noise = np.random.normal(0, 3.5, size=arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)
    if random.random() < 0.5:
        arr = np.array(out)
        h, w, _ = arr.shape
        n_pix = random.randint(int(0.0005*h*w), int(0.002*h*w))
        ys = np.random.randint(0, h, n_pix)
        xs = np.random.randint(0, w, n_pix)
        vals = np.random.choice([0, 255], size=n_pix)[:, None]
        arr[ys, xs] = np.repeat(vals, 3, axis=1)
        out = Image.fromarray(arr)
    if random.random() < 0.8:
        buf = io.BytesIO(); out.save(buf, format="JPEG", quality=random.randint(72, 92))
        buf.seek(0); out = Image.open(buf).convert("RGB")
    return out

def draw_grid(draw: ImageDraw.ImageDraw, x0, y0, widths, row_h, rows):
    x = x0
    for w in widths:
        draw.line((x, y0, x, y0 + rows*row_h), fill=(120,120,120), width=2)
        x += w
    draw.line((x, y0, x, y0 + rows*row_h), fill=(120,120,120), width=2)
    for r in range(rows+1):
        y = y0 + r*row_h
        draw.line((x0, y, x, y), fill=(120,120,120), width=2)

def col_xs(x0, widths):
    xs=[x0]
    for w in widths: xs.append(xs[-1]+w)
    return xs

# ---------------- layout calc ----------------
def build_layout():
    table_top = MARGIN_T + HEADER_GAP
    row_h = HEADER_ROW_H
    widths = [w for _, _, w in COLS]
    x0 = MARGIN_L
    xs = col_xs(x0, widths)

    # header row cells (typed labels)
    header_y = table_top
    header_cells: List[FieldDef] = []
    for ci, (col_name, ftype, w) in enumerate(COLS):
        x1 = xs[ci]
        box = (x1, header_y, w, row_h)
        header_cells.append(FieldDef(col_name, ftype, box))

    # data rows (handwriting)
    fields: List[FieldDef] = []
    for r in range(ROWS_DATA):
        y = table_top + row_h * (r + 1)  # skip header row
        for ci, (col_name, ftype, w) in enumerate(COLS):
            x1 = xs[ci]
            box = (x1 + 8, y + 8, w - 16, row_h - 16)  # inset for handwriting
            name = f"{col_name}_r{r+1}"
            fields.append(FieldDef(name, ftype, box))
    return header_cells, fields, widths, row_h, table_top, x0

# ---------------- helpers for typed labels ----------------
def draw_top_ui_labels(d, ui_font, header_boxes):
    labels = {"company": "Company:", "supervisor": "Supervisor:", "date": "Date:"}
    for k, title in labels.items():
        x,y,w,h = header_boxes[k]
        tw = d.textlength(title, font=ui_font)
        d.text((x - tw - 12, y + (h - ui_font.size)//2), title, fill=(30,30,30), font=ui_font)

def print_header_row(d, ui_font, header_cells):
    for fd in header_cells:
        name = fd.name
        label = COL_LABELS.get(name, name)
        x, y, w, h = fd.box
        # center (supports multi-line labels)
        lines = str(label).split("\n")
        line_h = ui_font.getbbox("Ag")[3] - ui_font.getbbox("Ag")[1]
        total_h = len(lines)*line_h + (len(lines)-1)*6
        y0 = y + (h - total_h)//2
        for j, ln in enumerate(lines):
            tw = d.textlength(ln, font=ui_font)
            tx = x + (w - tw)//2
            ty = y0 + j*(line_h + 6)
            d.text((tx, ty), ln, fill=(30,30,30), font=ui_font)

def draw_signature_scribble(d: ImageDraw.ImageDraw, box, ink=(20,20,20)):
    # quick pen-like squiggle inside the box
    x, y, w, h = box
    pad = 10
    left = x + pad
    right = x + w - pad
    top = y + h//2 - h//6
    bottom = y + h//2 + h//6

    # random polyline across the width
    n_pts = random.randint(6, 12)
    xs = np.linspace(left, right, n_pts)
    amp = random.uniform(4, 10)
    base = (top + bottom) / 2
    pts = []
    phase = random.uniform(0, np.pi)
    for i, xx in enumerate(xs):
        yy = base + amp * np.sin(phase + i * random.uniform(0.7, 1.4))
        # tiny jitter
        yy += random.uniform(-2, 2)
        pts.append((float(xx), float(yy)))

    # multi-pass line to look inkier
    for _ in range(2):
        jittered = [(px + random.uniform(-0.8,0.8), py + random.uniform(-0.8,0.8)) for px,py in pts]
        d.line(jittered, fill=ink, width=random.randint(2,3))

# ---------------- record synthesis ----------------
def sample_record_row():
    has_sig = random.random() < 0.6  # ~60% signed
    return {
        "employee_name": s_name(),
        "in_am": s_time(),
        "out_am": s_time(),
        "lunch": s_lunch(),
        "in_pm": s_time(),
        "out_pm": s_time(),
        "total_hours": s_hours(),
        # keep text empty; presence is what matters
        "signature": "",
        "signature_present": has_sig,
    }

def sample_page():
    rows = [sample_record_row() for _ in range(ROWS_DATA)]
    head = {
        "company":   random.choice(["Acme LLC","Northside Labs","General Widgets",""]),
        "supervisor":random.choice(["M. Rivera","K. Patel","L. Johnson",""]),
        "date":      s_date(),
    }
    return head, rows

# ---------------- main synth/render ----------------
def render_page(fonts, out_root, idx, rotate_cells=True, cell_rot_deg=3.0):
    page = Image.new("RGB", PAGE_SIZE, (250, 248, 240))
    d = ImageDraw.Draw(page)

    ui_font_title = load_ui_font(size=36)
    ui_font = load_ui_font(size=24)

    # Typed title
    title = "DAILY TIMESHEET"
    title_x = PAGE_SIZE[0] // 2 - d.textlength(title, font=ui_font_title) // 2
    d.text((title_x, 40), title, fill=(40, 40, 40), font=ui_font_title)

    # Header input boxes (typed labels, handwritten values inside)
    header_boxes = header_positions()
    for _, box in header_boxes.items():
        x, y, w, h = box
        d.rectangle((x, y, x + w, y + h), outline=(150, 150, 150), width=2)
    draw_top_ui_labels(d, ui_font, header_boxes)

    # Build table layout
    header_cells, fields, widths, row_h, table_top, x0 = build_layout()

    # Full grid: header(1) + data(ROWS_DATA)
    draw_grid(d, x0, table_top, widths, row_h, ROWS_DATA + 1)

    # Typed header row (column names)
    print_header_row(d, ui_font, header_cells)

    # Synthetic content for boxes
    header, rows = sample_page()

    # handwriting settings
    ink_rgb = (random.randint(5, 50),) * 3
    ink_rgba = ink_rgb + (255,)

    # random placement helper for handwriting within a cell
    def random_in_box(draw_obj, text, font, box, pad=6):
        x, y, w, h = box
        try:
            bbox = font.getbbox(text)
            th = (bbox[3] - bbox[1]) if bbox else font.size
        except Exception:
            th = font.size
        tw = draw_obj.textlength(text, font=font)
        max_dx = max(0, int(w - tw - pad*2))
        max_dy = max(0, int(h - th - pad*2))
        tx = x + pad + (0 if max_dx <= 0 else random.randint(0, max_dx))
        ty = y + pad + (0 if max_dy <= 0 else random.randint(0, max_dy))
        return tx, ty, tx - x, ty - y

    # Fill header boxes with HANDWRITING values (labels are typed above)
    for k, box in header_boxes.items():
        text = header[k]
        f = pick_font(fonts)
        tx, ty, lx, ly = random_in_box(d, text, f, box, pad=6)
        if rotate_cells:
            tile = Image.new("RGBA", (box[2], box[3]), (0,0,0,0))
            td = ImageDraw.Draw(tile)
            td.text((lx, ly), text, fill=ink_rgba, font=f)
            rot = random.uniform(-cell_rot_deg, cell_rot_deg)
            tile = tile.rotate(rot, resample=Image.BICUBIC, expand=False)
            page.paste(tile, (box[0], box[1]), tile)
        else:
            d.text((tx, ty), text, fill=ink_rgb, font=f)

    # Prepare rows by column order
    col_order = [c[0] for c in COLS]
    rows_by_order = []
    for r in range(ROWS_DATA):
        row = rows[r]
        rows_by_order.append([
            row["employee_name"], row["in_am"], row["out_am"], row["lunch"],
            row["in_pm"], row["out_pm"], row["total_hours"], row["signature"]
        ])

    # Render each DATA cell with handwriting
    for fd in fields:
        base, r_s = fd.name.rsplit("_r", 1)
        r_idx = int(r_s) - 1
        c_idx = col_order.index(base)
        text = rows_by_order[r_idx][c_idx]
        x, y, w, h = fd.box
        f = pick_font(fonts)

        # dotted guide in signature column
        if base == "signature":
            # draw scribble only if present
            if rows[r_idx]["signature_present"]:
                draw_signature_scribble(d, (x, y, w, h), ink=ink_rgb)
            continue  # no handwriting text for signature cells

        tx, ty, lx, ly = random_in_box(d, text, f, fd.box, pad=6)
        if rotate_cells:
            tile = Image.new("RGBA", (w, h), (0,0,0,0))
            td = ImageDraw.Draw(tile)
            td.text((lx, ly), text, fill=ink_rgba, font=f)
            rot = random.uniform(-cell_rot_deg, cell_rot_deg)
            tile = tile.rotate(rot, resample=Image.BICUBIC, expand=False)
            page.paste(tile, (x, y), tile)
        else:
            d.text((tx, ty), text, fill=ink_rgb, font=f)

    # Global page artifacts (whole template together)
    page = page_noise(page)

    # Save
    pages_dir = os.path.join(out_root, "pages"); ensure_dir(pages_dir)
    fname = f"timesheet_{idx:05d}.jpg"
    path = os.path.join(pages_dir, fname)
    page.save(path, quality=90)
    return page, path, fields, header, rows

# ---------------- crop/export ----------------
def crop(img: Image.Image, box):
    x,y,w,h = box
    return img.crop((x,y,x+w,y+h))

def run(args):
    random.seed(args.seed)
    ensure_dir(args.out_root)
    fonts = load_fonts(args.fonts_dir)

    crops_dir = os.path.join(args.out_root, "crops"); ensure_dir(crops_dir)
    tj = open(os.path.join(args.out_root,"train.jsonl"),"w",encoding="utf-8")
    vj = open(os.path.join(args.out_root,"val.jsonl"),"w",encoding="utf-8")
    csvf = open(os.path.join(args.out_root,"labels.csv"),"w",newline="",encoding="utf-8")
    csvw = csv.writer(csvf); csvw.writerow(["split","page_path","field_name","field_type","crop_path","text","signature_present"])

    n = args.n
    for i in range(n):
        img, page_path, fields, header, rows = render_page(fonts, args.out_root, i)

        # header crops (optional supervision for header handwriting)
        for k,(x,y,w,h) in header_positions().items():
            cp = crop(img, (x,y,w,h))
            cp_name = f"{os.path.basename(page_path).replace('.jpg','')}_{k}.jpg"
            cp_path = os.path.join(crops_dir, cp_name); cp.save(cp_path, quality=92)
            truth = header[k]
            row = {"image": cp_path, "text": truth, "field_type": "text", "page_image": page_path, "field_name": k, "bbox_xywh": (x,y,w,h)}
            is_val = random.random() < args.val_pct
            (vj if is_val else tj).write(json.dumps(row)+"\n")
            csvw.writerow(["val" if is_val else "train", page_path, k, "text", cp_path, truth, ""])

        # table crops (DATA rows only)
        valcut = args.val_pct
        col_order = [c[0] for c in COLS]
        for r in range(ROWS_DATA):
            rdat = rows[r]
            mapping = {
                "employee_name": rdat["employee_name"], "in_am": rdat["in_am"],
                "out_am": rdat["out_am"], "lunch": rdat["lunch"],
                "in_pm": rdat["in_pm"], "out_pm": rdat["out_pm"],
                "total_hours": rdat["total_hours"], "signature": rdat["signature"],
            }
            for col_name, ftype, _ in COLS:
                name = f"{col_name}_r{r+1}"
                fd = next(f for f in fields if f.name == name)
                cp = crop(img, fd.box)
                cp_name = f"{os.path.basename(page_path).replace('.jpg','')}_{name}.jpg"
                cp_path = os.path.join(crops_dir, cp_name); cp.save(cp_path, quality=92)

                truth = mapping[col_name]
                sig_flag = ""
                if col_name == "signature":
                    sig_flag = "1" if rdat["signature_present"] else "0"

                row_json = {
                    "image": cp_path,
                    "text": truth,
                    "field_type": ftype,
                    "page_image": page_path,
                    "field_name": name,
                    "bbox_xywh": fd.box,
                    "signature_present": (sig_flag == "1")
                }
                is_val = random.random() < valcut
                (vj if is_val else tj).write(json.dumps(row_json) + "\n")
                csvw.writerow(["val" if is_val else "train", page_path, name, ftype, cp_path, truth, sig_flag])


    tj.close(); vj.close(); csvf.close()
    print(f"[done] wrote pages/crops/labels to {args.out_root}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="e.g., exports/datasets/daily_timesheet_synth/v1")
    ap.add_argument("--fonts_dir", default="assets/fonts", help="dir with handwriting .ttf/.otf")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--val_pct", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args)
