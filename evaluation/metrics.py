"""Metrici de evaluare (Persoana 1)."""
import numpy as np

# Crop-ul Eigen (Eigen et al., 2014): zona din imaginea NYU 480×640 pe care se evaluează în articole.
EIGEN_CROP = (slice(45, 471), slice(41, 601))
MAX_DEPTH = 10.0  # metri, plafonul standard pe NYU


def depth_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """AbsRel, RMSE și δ1 în crop-ul Eigen, pe pixelii cu adevăr valid (0 < gt ≤ 10 m)."""
    p, g = pred[EIGEN_CROP], gt[EIGEN_CROP]
    valid = (g > 0) & (g <= MAX_DEPTH)
    p, g = np.clip(p[valid], 1e-3, MAX_DEPTH), g[valid]
    return {
        "AbsRel": float(np.mean(np.abs(p - g) / g)),
        "RMSE": float(np.sqrt(np.mean((p - g) ** 2))),
        "delta1": float(np.mean(np.maximum(p / g, g / p) < 1.25)),
    }
