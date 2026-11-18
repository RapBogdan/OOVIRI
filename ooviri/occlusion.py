"""The per-pixel occlusion test: is the real surface in front of the virtual one?"""
from __future__ import annotations

import numpy as np

from .placement import PlacedSprite


def valid_depth(depth: np.ndarray) -> np.ndarray:
    """Pixels with a usable depth (sensors and some models leave holes as 0, NaN or inf)."""
    return np.isfinite(depth) & (depth > 0)


def _check(scene_depth: np.ndarray, placed: PlacedSprite) -> None:
    if scene_depth.shape != placed.depth.shape:
        raise ValueError(
            f"depth map is {scene_depth.shape}, the image frame is {placed.depth.shape}; "
            "resize the depth map to the image first (ooviri.depthio.resize_depth)"
        )


def occlusion_mask(scene_depth: np.ndarray, placed: PlacedSprite, eps: float = 0.0) -> np.ndarray:
    """True where the virtual object is hidden by the real scene.

    A pixel of the object is occluded when the scene surface there is closer to the
    camera than the object's surface. eps (metres) demands a clearer gap before
    declaring occlusion. Pixels with no usable depth never occlude.
    """
    _check(scene_depth, placed)
    return placed.footprint & valid_depth(scene_depth) & (scene_depth < placed.depth - eps)


def margin(scene_depth: np.ndarray, placed: PlacedSprite) -> np.ndarray:
    """Signed gap in metres: positive = real surface in front (occluded), negative = behind (visible).

    -inf outside the object and where the depth is unusable. The size of the gap says
    how sure the depth test is; the soft vote uses it.
    """
    _check(scene_depth, placed)
    out = np.full(scene_depth.shape, -np.inf, np.float32)
    ok = placed.footprint & valid_depth(scene_depth)
    out[ok] = placed.depth[ok] - scene_depth[ok]
    return out
