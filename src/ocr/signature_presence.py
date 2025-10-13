# src/ocr/signature_presence.py
import cv2
import numpy as np

def signature_present(bgr_crop, min_stroke_ratio=0.003, min_components=10):
    """
    Return True if there's ink-like content.
    - binarize adaptively
    - count dark pixels and connected components
    """
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    # light blur to ignore paper texture
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    # adaptive threshold (invert: ink -> white)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 21, 10)
    H, W = th.shape
    white = th.sum() / 255
    stroke_ratio = white / (H*W)
    # small morphology to connect pen strokes
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2,2), np.uint8), iterations=1)
    # component count (exclude tiny specks)
    cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in cnts if cv2.contourArea(c) >= 6]
    return (stroke_ratio >= min_stroke_ratio) and (len(big) >= min_components)
