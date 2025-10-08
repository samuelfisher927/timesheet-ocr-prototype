# src/ocr/preprocess.py
from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple

def _auto_contrast(img_gray: np.ndarray) -> np.ndarray:
    # Stretch histogram to [0, 255] based on 1%–99% percentiles
    lo, hi = np.percentile(img_gray, (1, 99))
    if hi <= lo:  # degenerate
        return img_gray
    out = np.clip((img_gray - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return out

def _estimate_skew_angle(img_bin: np.ndarray) -> float:
    """
    Estimate skew using minAreaRect on detected text lines.
    Returns angle in degrees. Positive means rotate counter-clockwise to correct.
    """
    # Find edges then dilate to aggregate text lines
    edges = cv2.Canny(img_bin, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    dil = cv2.dilate(edges, kernel, iterations=1)

    # Find contours; if none, no skew
    cnts, _ = cv2.findContours(dil, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0

    rects = []
    for c in cnts:
        if cv2.contourArea(c) < 30:  # tiny noise
            continue
        rect = cv2.minAreaRect(c)
        (w, h) = rect[1]
        if w * h < 50:  # ignore tiny blobs
            continue
        angle = rect[2]
        # OpenCV returns angle in [-90, 0); normalize so that near-horizontal lines → small angles
        if w < h:
            angle = angle + 90
        rects.append(angle)

    if not rects:
        return 0.0

    # Robust average (median) to reduce outliers
    angle = float(np.median(rects))
    # Clamp to a reasonable range
    angle = max(min(angle, 20.0), -20.0)
    return angle

def _rotate(image: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 0.5:
        return image
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def preprocess_image(
    image_bgr: np.ndarray,
    return_debug: bool = False
) -> Tuple[np.ndarray, dict] | np.ndarray:
    """
    Minimal, fast cleanup for handwritten crops:
      1) gray + auto-contrast
      2) gentle denoise
      3) adaptive binarization
      4) deskew (small angles)
      5) light morphology (close->erode) to connect broken strokes
    Returns a single-channel uint8 image suitable for OCR.
    """
    dbg = {}

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dbg["gray"] = gray

    gray = _auto_contrast(gray)
    dbg["autocontrast"] = gray

    # Gentle denoise without killing edges
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=5)
    dbg["denoise"] = gray

    # Adaptive threshold (Gaussian)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # white text on black background for morphology
        21,  # block size (odd)
        10   # C
    )
    dbg["bin_init"] = bin_img

    # Deskew on a copy (use binary for angle detection)
    angle = _estimate_skew_angle(bin_img)
    dbg["angle_deg"] = angle

    # Rotate original gray and re-binarize for cleaner result
    rot_gray = _rotate(gray, angle)
    bin_img = cv2.adaptiveThreshold(
        rot_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 10
    )
    dbg["bin_rot"] = bin_img

    # Morphology to connect strokes & remove specks
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    bin_morph = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    bin_morph = cv2.erode(bin_morph, kernel_erode, iterations=1)
    dbg["bin_morph"] = bin_morph

    # Return to black text on white (more typical for OCR models)
    final = cv2.bitwise_not(bin_morph)
    dbg["final"] = final

    return (final, dbg) if return_debug else final
