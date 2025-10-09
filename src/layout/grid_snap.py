# src/layout/grid_snap.py
# Snap arbitrary detections onto a known grid (rows x columns) and assign 1:1 with Hungarian.

from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

def _centroid(box):
    x,y,w,h = box
    return (x + w/2.0, y + h/2.0)

def cluster_rows_y(dets: List[Dict], row_tops: List[int], row_h: int) -> List[List[Dict]]:
    """Bin detections into rows by y-center proximity to row anchors (header+data rows)."""
    rows = [[] for _ in row_tops]
    for d in dets:
        _, cy = _centroid(d["bbox"])
        idx = int(np.clip(round((cy - row_tops[0]) / row_h), 0, len(row_tops)-1))
        rows[idx].append(d)
    return rows

def assign_row_to_columns(row_dets: List[Dict], col_centers_x: List[int]) -> List[Optional[Dict]]:
    """Assign detections in one row to expected column centers with minimal X-distance."""
    if not row_dets:
        return [None]*len(col_centers_x)
    D = np.zeros((len(col_centers_x), len(row_dets)), dtype=float)
    xs = [ _centroid(d["bbox"])[0] for d in row_dets ]
    for i, cx in enumerate(col_centers_x):
        D[i,:] = [abs(cx - x) for x in xs]

    if HAS_SCIPY:
        ri, cj = linear_sum_assignment(D)
        assigned = [None]*len(col_centers_x)
        for i, j in zip(ri, cj):
            assigned[i] = row_dets[j]
        return assigned
    else:
        # greedy fallback
        assigned = [None]*len(col_centers_x)
        used = set()
        for i, cx in enumerate(col_centers_x):
            j = int(np.argmin([abs(cx - x) if k not in used else 1e9 for k,x in enumerate(xs)]))
            if j not in used:
                assigned[i] = row_dets[j]
                used.add(j)
        return assigned

def snap_and_assign(
    detections: List[Dict],
    table_top: int,
    row_h: int,
    n_rows_total: int,               # header(1) + data(ROWS_DATA)
    x0: int,
    col_widths: List[int],
    class_names: List[str],          # len == n_cols, order must match COLS
):
    """Return a dictionary {(row_idx, col_idx): det or None} for grid slots."""
    # expected anchors
    col_centers = []
    x = x0
    for w in col_widths:
        col_centers.append(int(x + w/2))
        x += w
    row_tops = [int(table_top + i*row_h) for i in range(n_rows_total)]

    # bin by cls -> we’ll keep class info but assignment is geometry-driven
    rows_binned = cluster_rows_y(detections, row_tops, row_h)

    grid = {}
    for r_idx in range(n_rows_total):
        row_dets = rows_binned[r_idx]
        assigned = assign_row_to_columns(row_dets, col_centers)
        for c_idx, det in enumerate(assigned):
            grid[(r_idx, c_idx)] = det
    return grid
