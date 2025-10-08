from __future__ import annotations
import os, json, argparse

def main(path, images_root):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i==2: break
            obj = json.loads(line)
            keys = list(obj.keys())
            print(f"\n--- LINE {i} KEYS:", keys)
            img = obj.get("image") or obj.get("image_path") or (obj.get("image_id") and f"{obj['image_id']}.png")
            print("image ref:", img, "-> exists:", os.path.exists(os.path.join(images_root, str(img))) if img else False)
            fields = obj.get("fields") or obj.get("labels") or obj.get("annotations")
            print("fields type:", type(fields).__name__)
            if isinstance(fields, dict):
                print("sample field keys:", list(fields.keys())[:10])
            elif isinstance(fields, list):
                print("sample item keys:", list((fields[0] or {}).keys()) if fields else [])
            else:
                print("no fields found")
    print("\n[done] probe complete")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_jsonl", required=True)
    ap.add_argument("--images_root", required=True)
    args = ap.parse_args()
    main(args.labels_jsonl, args.images_root)
