# src/tools/test_preprocess.py
import os, cv2
from tqdm import tqdm
from src.ocr.preprocess import preprocess_image

def run(in_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    imgs = []
    for root, _, files in os.walk(in_dir):
        for fn in files:
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                imgs.append(os.path.join(root, fn))

    for p in tqdm(imgs, desc="preprocess"):
        img = cv2.imread(p)
        out, dbg = preprocess_image(img, return_debug=True)
        rel = os.path.relpath(p, in_dir).replace("\\", "__")
        cv2.imwrite(os.path.join(out_dir, f"{rel}__final.png"), out)
        # optional debug dumps (comment out if too many files)
        # cv2.imwrite(os.path.join(out_dir, f"{rel}__bin.png"), dbg["bin_rot"])

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", default="exports/preprocessed")
    args = ap.parse_args()
    run(args.in_dir, args.out_dir)
