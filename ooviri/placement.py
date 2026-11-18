"""Placing a sprite in the image: position, depth and scale -> full-frame colour, alpha and depth."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

from .camera import Camera
from .sprite import Sprite

ALPHA_MIN = 0.02  # below this coverage a pixel does not belong to the object


@dataclass(frozen=True)
class Placement:
    """Where the virtual object goes."""

    x: float            # pixel column of the object's centre
    y: float            # pixel row of the object's centre
    z: float            # depth of the object's centre plane, metres
    scale: float = 1.0  # multiplier on the object's physical size

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Placement":
        return cls(x=float(d["x"]), y=float(d["y"]), z=float(d["z"]), scale=float(d.get("scale", 1.0)))


@dataclass
class PlacedSprite:
    """A sprite projected into the image frame."""

    rgb: np.ndarray        # (H, W, 3) float32 in 0..1, zero outside the footprint
    alpha: np.ndarray      # (H, W) float32 in 0..1
    depth: np.ndarray      # (H, W) float32 metres, +inf outside the footprint
    footprint: np.ndarray  # (H, W) bool, pixels the object covers
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive), clipped to the frame


def _resize(channel: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(np.ascontiguousarray(channel, dtype=np.float32))
    return np.asarray(img.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def sprite_pixel_size(sprite: Sprite, placement: Placement, camera: Camera) -> tuple[int, int]:
    """(width, height) in pixels: physical size times focal length, divided by depth."""
    th = max(1, int(round(sprite.height_m * placement.scale * camera.pixels_per_metre(placement.z))))
    tw = max(1, int(round(th * sprite.aspect)))
    return tw, th


def place(sprite: Sprite, placement: Placement, camera: Camera) -> PlacedSprite:
    """Project the sprite into a (camera.height, camera.width) frame."""
    H, W = camera.height, camera.width
    tw, th = sprite_pixel_size(sprite, placement, camera)

    a_src = sprite.rgba[..., 3].astype(np.float32) / 255.0
    rgb_src = sprite.rgba[..., :3].astype(np.float32) / 255.0
    # Resize premultiplied values, then divide by the resized alpha: no dark halo, no relief bleeding.
    a = np.clip(_resize(a_src, (tw, th)), 0.0, 1.0)
    safe = np.maximum(a, 1e-6)
    rgb = np.dstack([_resize(rgb_src[..., c] * a_src, (tw, th)) / safe for c in range(3)])
    rel = _resize(sprite.relief * a_src, (tw, th)) / safe * placement.scale

    x0 = int(round(placement.x - tw / 2.0))
    y0 = int(round(placement.y - th / 2.0))
    ix0, iy0, ix1, iy1 = max(x0, 0), max(y0, 0), min(x0 + tw, W), min(y0 + th, H)

    out_rgb = np.zeros((H, W, 3), np.float32)
    out_a = np.zeros((H, W), np.float32)
    out_d = np.full((H, W), np.inf, np.float32)
    if ix1 > ix0 and iy1 > iy0:
        src = (slice(iy0 - y0, iy1 - y0), slice(ix0 - x0, ix1 - x0))
        dst = (slice(iy0, iy1), slice(ix0, ix1))
        out_a[dst] = a[src]
        out_rgb[dst] = np.clip(rgb[src], 0.0, 1.0)
        out_d[dst] = placement.z + rel[src]
    footprint = out_a > ALPHA_MIN
    out_a[~footprint] = 0.0
    out_rgb[~footprint] = 0.0
    out_d[~footprint] = np.inf
    return PlacedSprite(rgb=out_rgb, alpha=out_a, depth=out_d, footprint=footprint,
                        bbox=(ix0, iy0, max(ix1, ix0), max(iy1, iy0)))


def random_placements(depth_ref: np.ndarray, sprite: Sprite, camera: Camera, n: int, seed: int = 0,
                      border: float = 0.15) -> list[Placement]:
    """Reproducible placements that tend to be partly occluded.

    The depth Z is drawn between the nearest and the farthest surfaces found around
    the chosen position in a reference depth map (the ground truth on NYU), so some
    of the scene ends up in front of the object and some behind it.
    """
    rng = np.random.default_rng(seed)
    H, W = depth_ref.shape
    valid = np.isfinite(depth_ref) & (depth_ref > 0)
    out: list[Placement] = []
    attempts = 0
    while len(out) < n and attempts < 200 * max(n, 1):
        attempts += 1
        x = float(rng.uniform(border * W, (1.0 - border) * W))
        y = float(rng.uniform(border * H, (1.0 - border) * H))
        r, c = int(y), int(x)
        if not valid[r, c]:
            continue
        half = max(4, int(0.5 * sprite.height_m * camera.pixels_per_metre(float(depth_ref[r, c]))))
        win = depth_ref[max(0, r - half):r + half + 1, max(0, c - half):c + half + 1]
        vals = win[np.isfinite(win) & (win > 0)]
        if vals.size < 16:
            continue
        lo, hi = np.percentile(vals, [10, 90])
        z = max(0.3, float(rng.uniform(0.9 * lo, 1.1 * hi)))
        out.append(Placement(x=x, y=y, z=z))
    return out
