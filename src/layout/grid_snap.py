# src/layout/grid_snap.py
# Snap YOLO detections onto a known table grid (rows x columns) and assign 1:1.

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Iterable
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

Det = Dict[str, object]
GridKey = Tuple[int, int]

def _centroid(box):
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)

def _row_center(row_top: float, row_h: float) -> float:
    return row_top + 0.5 * row_h

def cluster_rows_y(
    dets: List[Det],
    row_tops: List[int],
    row_h: int,
    max_dev: float = 0.55,   # reject if |yc - row_center| > max_dev * row_h (0.55 ~ forgiving)
) -> List[List[Det]]:
    """Bin detections into rows by y-center; drop clear outliers."""
    rows = [[] for _ in row_tops]
    for d in dets:
        _, cy = _centroid(d["bbox"])
        # nearest row index
        idx = int(np.clip(round((cy - row_tops[0]) / row_h), 0, len(row_tops) - 1))
        rc = _row_center(row_tops[idx], row_h)
        if abs(cy - rc) <= max_dev * row_h:
            rows[idx].append(d)
    return rows

def assign_row_to_columns(row_dets: List[Det], col_centers_x: List[int]) -> List[Optional[Det]]:
    """Assign detections in one row to expected column centers with minimal X-distance."""
    if not row_dets:
        return [None] * len(col_centers_x)
    xs = [_centroid(d["bbox"])[0] for d in row_dets]
    D = np.zeros((len(col_centers_x), len(row_dets)), dtype=float)
    for i, cx in enumerate(col_centers_x):
        D[i, :] = [abs(cx - x) for x in xs]

    assigned = [None] * len(col_centers_x)
    if HAS_SCIPY:
        ri, cj = linear_sum_assignment(D)
        for i, j in zip(ri, cj):
            assigned[i] = row_dets[j]
    else:
        used = set()
        for i, cx in enumerate(col_centers_x):
            j = int(np.argmin([abs(cx - x) if k not in used else 1e9 for k, x in enumerate(xs)]))
            if j not in used:
                assigned[i] = row_dets[j]
                used.add(j)
    return assigned

def snap_and_assign(
    detections: List[Det],
    table_top: int,
    row_h: int,
    n_rows_total: int,               # number of DATA rows (not including typed header rows)
    x0: int,
    col_widths: List[int],
    header_rows: int = 0,            # ✅ skip this many header bands before data rows
) -> Dict[GridKey, Optional[Det]]:
    """Return {(row_idx, col_idx): det or None} for data-row grid slots."""
    # column centers
    col_centers = []
    x = x0
    for w in col_widths:
        col_centers.append(int(x + w / 2))
        x += w

    # shift start to first DATA row
    table_top_eff = int(table_top + header_rows * row_h)
    row_tops = [int(table_top_eff + i * row_h) for i in range(n_rows_total)]

    rows_binned = cluster_rows_y(detections, row_tops, row_h)

    grid: Dict[GridKey, Optional[Det]] = {}
    for r_idx in range(n_rows_total):
        row_dets = rows_binned[r_idx]
        assigned = assign_row_to_columns(row_dets, col_centers)
        for c_idx, det in enumerate(assigned):
            if det is None:
                grid[(r_idx, c_idx)] = None
            else:
                prev = grid.get((r_idx, c_idx))
                if (prev is None) or (float(det.get("score", 0.0)) > float(prev.get("score", 0.0))):
                    grid[(r_idx, c_idx)] = det
    return grid

def snap_to_grid(
    detections: List[Det],
    table_top: int,
    row_h: int,
    n_rows_total: int,               # DATA rows count; table_top is FIRST header band top
    x0: int,
    col_widths: List[int],
    class_names: Optional[List[str]] = None,   # unused (kept for future checks)
    ignore_classes: Optional[Iterable[str]] = ("date",),
    header_rows: int = 0,            # ✅ expose here too
) -> Dict[GridKey, Optional[Det]]:
    """
    Public wrapper used by pipeline.py.
    - Filters out header-only classes (e.g., 'date').
    - Skips 'header_rows' bands before snapping to DATA rows.
    """
    if ignore_classes:
        ig = set(ignore_classes)
        detections = [d for d in detections if str(d.get("cls", "")) not in ig]

    return snap_and_assign(
        detections=detections,
        table_top=table_top,
        row_h=row_h,
        n_rows_total=n_rows_total,
        x0=x0,
        col_widths=col_widths,
        header_rows=header_rows,
    )
