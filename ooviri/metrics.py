"""Mask agreement scores, used here as a self-check. The full evaluation belongs to P1."""
from __future__ import annotations

import numpy as np


def _prep(pred: np.ndarray, ref: np.ndarray, region: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    p, r = np.asarray(pred, bool), np.asarray(ref, bool)
    if region is not None:
        keep = np.asarray(region, bool)
        p, r = p[keep], r[keep]
    return p, r


def iou(pred: np.ndarray, ref: np.ndarray, region: np.ndarray | None = None) -> float:
    """Intersection over union of two masks; 1.0 when both are empty."""
    p, r = _prep(pred, ref, region)
    union = np.logical_or(p, r).sum()
    return 1.0 if union == 0 else float(np.logical_and(p, r).sum() / union)


def f1(pred: np.ndarray, ref: np.ndarray, region: np.ndarray | None = None) -> float:
    """F1 score of the "occluded" class; 1.0 when both masks are empty."""
    p, r = _prep(pred, ref, region)
    tp = np.logical_and(p, r).sum()
    total = p.sum() + r.sum()
    return 1.0 if total == 0 else float(2.0 * tp / total)
