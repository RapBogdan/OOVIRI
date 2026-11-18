"""Reading and writing images and depth maps."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Ten stops of the viridis colour map; interpolated linearly in between.
_VIRIDIS = np.array([
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
    [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37],
], dtype=np.float32)


def load_rgb(path: str | Path) -> np.ndarray:
    """Image as (H, W, 3) uint8."""
    return np.array(Image.open(path).convert("RGB"))


def load_depth(path: str | Path) -> np.ndarray:
    """Depth map in metres as float32. .npy holds metres; 16-bit PNG/TIFF holds millimetres."""
    p = Path(path)
    if p.suffix.lower() == ".npy":
        depth = np.load(p)
    else:
        depth = np.array(Image.open(p)).astype(np.float32) / 1000.0
    depth = np.squeeze(np.asarray(depth, dtype=np.float32))
    if depth.ndim != 2:
        raise ValueError(f"{p} is not a single-channel depth map (shape {depth.shape})")
    return depth


def resize_depth(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a depth map to (H, W)."""
    img = Image.fromarray(np.ascontiguousarray(depth, dtype=np.float32))
    return np.asarray(img.resize((shape[1], shape[0]), Image.Resampling.BILINEAR), dtype=np.float32)


def save_image(path: str | Path, array: np.ndarray) -> Path:
    """Save a uint8 image, or a boolean mask as a black and white PNG."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(array)
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    Image.fromarray(a).save(p)
    return p


def colorize_depth(depth: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Depth map as a colour picture: near = bright yellow, far = dark purple, holes = black."""
    ok = np.isfinite(depth) & (depth > 0)
    if not ok.any():
        return np.zeros((*depth.shape, 3), np.uint8)
    lo = float(depth[ok].min()) if vmin is None else vmin
    hi = float(depth[ok].max()) if vmax is None else vmax
    t = 1.0 - np.clip((np.where(ok, depth, hi) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    pos = t * (len(_VIRIDIS) - 1)
    stops = np.arange(len(_VIRIDIS))
    rgb = np.dstack([np.interp(pos, stops, _VIRIDIS[:, c]) for c in range(3)])
    rgb[~ok] = 0.0
    return (rgb + 0.5).astype(np.uint8)
