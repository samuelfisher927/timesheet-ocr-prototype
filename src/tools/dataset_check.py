from __future__ import annotations
import os, json, argparse
from src.ocr.normalizers import sanitize_time, sanitize_amount

def check_split(jsonl_path: str, root: str):
    n = 0; missing = 0; bad_time = 0; bad_amount = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            n += 1
            obj = json.loads(line)
            img_rel = obj["image"].replace("\\","/")
            text = obj["text"].strip()
            img_abs = os.path.join(root, img_rel)
            if not os.path.exists(img_abs):
                missing += 1
            # heuristic: time has ':'; amount has digits + maybe '.' and '-'
            if ":" in text:
                if sanitize_time(text) != text:
                    bad_time += 1
            else:
                clean_amt = sanitize_amount(text)
                if clean_amt != text:
                    bad_amount += 1
    return n, missing, bad_time, bad_amount

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="exports/datasets/from_jsonl/v1")
    args = ap.parse_args()
    train = os.path.join(args.root, "train.jsonl")
    val   = os.path.join(args.root, "val.jsonl")
    for p in (train, val):
        if not os.path.exists(p):
            print(f"[warn] missing: {p}")
    for split in ("train","val"):
        p = os.path.join(args.root, f"{split}.jsonl")
        if not os.path.exists(p): continue
        n, missing, bad_t, bad_a = check_split(p, args.root)
        print(f"{split:5}: {n:5} samples | missing imgs={missing} | bad_time={bad_t} | bad_amount={bad_a}")
    # quick peek
    print("\n[first 3 lines of train.jsonl]")
    try:
        with open(train, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i==3: break
                print(line.rstrip())
    except FileNotFoundError:
        pass
