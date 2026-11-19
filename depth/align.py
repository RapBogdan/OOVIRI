"""Alinierea scarii adancimii relative la metri """
import numpy as np


def align_scale(relative: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Aliniază o disparitate relativă (mare = aproape) la adâncimea `reference` (metri).

    Scară + offset prin cele mai mici pătrate în spațiul disparității: s·relative + t ≈ 1/reference,
    pe pixelii cu reference > 0. Întoarce adâncimea în metri.
    `reference` este adâncimea reală NYU (evaluare) sau ieșirea ZoeDepth (fotografii proprii).
    """
    valid = reference > 0
    x = relative[valid].astype(np.float64)
    A = np.stack([x, np.ones_like(x)], axis=1)
    s, t = np.linalg.lstsq(A, 1.0 / reference[valid], rcond=None)[0]
    return (1.0 / np.clip(s * relative + t, 1e-3, None)).astype(np.float32)
