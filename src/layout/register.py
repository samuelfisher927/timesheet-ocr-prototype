# src/layout/register.py
# Homography register -> warp -> crop fixed boxes.
# Usage (example):
#   from layout.register import register_and_crop
#   crops = register_and_crop("scan.jpg", "assets/templates/canonical_timesheet.jpg", FIXED_BOXES, out_size=(1700,2200))

import cv2
import numpy as np
from typing import List, Tuple, Optional

# ---- Core ORB+RANSAC registration ----
def estimate_homography(src_bgr, ref_bgr, max_features=3000, good_match_ratio=0.35) -> Optional[np.ndarray]:
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(src_bgr, None)
    kp2, des2 = orb.detectAndCompute(ref_bgr, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < good_match_ratio * n.distance:
            good.append(m)
    if len(good) < 8:
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def warp_to_template(src_bgr, ref_bgr, out_size) -> Optional[np.ndarray]:
    H = estimate_homography(src_bgr, ref_bgr)
    if H is None:
        return None
    W, Ht = out_size
    warped = cv2.warpPerspective(src_bgr, H, (W, Ht), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped

# ---- Cropping helpers ----
def crop_xywh(img_bgr, box_xywh: Tuple[int,int,int,int]):
    x, y, w, h = box_xywh
    return img_bgr[y:y+h, x:x+w].copy()

def register_and_crop(
    src_path: str,
    template_path: str,
    boxes_xywh: List[Tuple[int,int,int,int]],
    out_size: Tuple[int,int] = (1700, 2200),
):
    src = cv2.imread(src_path, cv2.IMREAD_COLOR)
    ref = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if src is None or ref is None:
        raise FileNotFoundError("Could not read source or template image.")
    warped = warp_to_template(src, ref, out_size=out_size)
    if warped is None:
        # Fallback: naive resize to out_size (still returns crops; less accurate)
        warped = cv2.resize(src, out_size, interpolation=cv2.INTER_LINEAR)
    crops = [crop_xywh(warped, b) for b in boxes_xywh]
    return warped, crops
