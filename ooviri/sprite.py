"""2.5D sprites: an RGBA image plus a per-pixel relief (depth offset in metres).

A sprite is "2.5D" because every pixel carries exactly one depth value: enough
to make a sphere bulge towards the camera, not enough to look behind it.
Axes follow the image: x right, y down, z into the scene (away from the camera),
so a relief of -0.2 means "20 cm closer to the camera than the object's centre".
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

_SS = 3   # supersampling factor, gives anti-aliased silhouettes
_PAD = 2  # transparent border in pixels, keeps edges soft after resizing

# Direction towards the light (top-left, in front of the object) and the Blinn half vector.
_LIGHT = np.array([-0.45, -0.60, -0.66])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)
_HALF = _LIGHT + np.array([0.0, 0.0, -1.0])
_HALF = _HALF / np.linalg.norm(_HALF)


@dataclass
class Sprite:
    """A virtual object ready to be placed in a scene."""

    rgba: np.ndarray    # (h, w, 4) uint8, straight alpha
    relief: np.ndarray  # (h, w) float32 metres, offset from the object's centre plane
    height_m: float     # physical height covered by the whole sprite image
    name: str = "sprite"

    def __post_init__(self) -> None:
        if self.rgba.ndim != 3 or self.rgba.shape[2] != 4 or self.rgba.dtype != np.uint8:
            raise ValueError("rgba must be an (h, w, 4) uint8 array")
        if self.relief.shape != self.rgba.shape[:2]:
            raise ValueError("relief must have the same height and width as rgba")
        if self.height_m <= 0:
            raise ValueError("height_m must be positive")
        self.relief = np.asarray(self.relief, dtype=np.float32)

    @property
    def aspect(self) -> float:
        """Width divided by height."""
        return self.rgba.shape[1] / self.rgba.shape[0]

    def save(self, directory: str | Path) -> Path:
        """Write <name>.png (RGBA), <name>_relief.npy and <name>.json; returns the PNG path."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.rgba).save(d / f"{self.name}.png")
        np.save(d / f"{self.name}_relief.npy", self.relief)
        meta = {"name": self.name, "height_m": self.height_m}
        (d / f"{self.name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return d / f"{self.name}.png"

    @classmethod
    def load(cls, directory: str | Path, name: str) -> "Sprite":
        d = Path(directory)
        meta = json.loads((d / f"{name}.json").read_text(encoding="utf-8"))
        rgba = np.array(Image.open(d / f"{name}.png").convert("RGBA"))
        relief = np.load(d / f"{name}_relief.npy")
        return cls(rgba=rgba, relief=relief, height_m=float(meta["height_m"]), name=name)

    @classmethod
    def from_png(cls, path: str | Path, height_m: float, name: str | None = None) -> "Sprite":
        """Any RGBA picture as a flat sprite (relief 0 everywhere)."""
        rgba = np.array(Image.open(path).convert("RGBA"))
        relief = np.zeros(rgba.shape[:2], np.float32)
        return cls(rgba=rgba, relief=relief, height_m=height_m, name=name or Path(path).stem)


def _shade(normals: np.ndarray, color, ambient: float = 0.30, specular: float = 0.25,
           shininess: float = 40.0) -> np.ndarray:
    """Lambert shading plus a small highlight, so the primitive reads as a solid object."""
    lambert = np.clip(normals @ _LIGHT, 0.0, 1.0)
    spec = np.clip(normals @ _HALF, 0.0, 1.0) ** shininess
    base = np.asarray(color, np.float32) / 255.0
    rgb = base[None, None, :] * (ambient + (1.0 - ambient) * lambert)[..., None] + specular * spec[..., None]
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _block_mean(a: np.ndarray, k: int) -> np.ndarray:
    h, w = a.shape[0] // k, a.shape[1] // k
    return a.reshape(h, k, w, k, *a.shape[2:]).mean(axis=(1, 3))


def _finish(rgb: np.ndarray, alpha: np.ndarray, relief: np.ndarray, height_m: float, name: str) -> Sprite:
    """Downsample the supersampled render (alpha-weighted) and add the transparent border."""
    alpha = alpha.astype(np.float32)
    a = _block_mean(alpha, _SS)
    safe = np.maximum(a, 1e-6)
    rgb_s = _block_mean(rgb * alpha[..., None], _SS) / safe[..., None]
    rel_s = _block_mean(relief * alpha, _SS) / safe
    rgb_s[a <= 0] = 0.0
    rel_s[a <= 0] = 0.0
    rgba = (np.clip(np.dstack([rgb_s, a]), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    h = rgba.shape[0]
    rgba = np.pad(rgba, ((_PAD, _PAD), (_PAD, _PAD), (0, 0)))
    rel_s = np.pad(rel_s.astype(np.float32), _PAD)
    return Sprite(rgba=rgba, relief=rel_s, height_m=height_m * (h + 2 * _PAD) / h, name=name)


def make_sphere(radius_m: float = 0.20, px: int = 256, color=(242, 140, 40), name: str = "sphere") -> Sprite:
    """A shaded ball. Its centre pixel is radius_m closer to the camera than its outline."""
    n = px * _SS
    t = (np.arange(n, dtype=np.float32) + 0.5) / n * 2.0 - 1.0
    u, v = np.meshgrid(t, t)
    r2 = u * u + v * v
    inside = r2 <= 1.0
    nz = -np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    rgb = _shade(np.dstack([u, v, nz]), color)
    return _finish(rgb, inside, radius_m * nz, 2.0 * radius_m, name)


def make_cylinder(radius_m: float = 0.15, height_m: float = 0.55, px: int = 256, color=(196, 64, 140),
                  name: str = "cylinder") -> Sprite:
    """An upright cylinder seen from the side; the middle column is the closest."""
    h = px * _SS
    w = max(8, int(round(px * 2.0 * radius_m / height_m))) * _SS
    t = (np.arange(w, dtype=np.float32) + 0.5) / w * 2.0 - 1.0
    u = np.tile(t, (h, 1))
    nz = -np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0))
    rgb = _shade(np.dstack([u, np.zeros_like(u), nz]), color)
    rows = np.linspace(1.04, 0.90, h, dtype=np.float32)[:, None, None]  # slightly darker towards the base
    rgb = np.clip(rgb * rows, 0.0, 1.0)
    return _finish(rgb, np.ones((h, w), bool), radius_m * nz, height_m, name)


def make_box(size_m=(0.40, 0.40, 0.40), yaw_deg: float = 35.0, pitch_deg: float = 20.0, px: int = 256,
             color=(40, 160, 150), name: str = "box") -> Sprite:
    """A box turned so that three faces show, rendered by casting one ray per pixel (orthographic)."""
    half = np.asarray(size_m, np.float64) / 2.0
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    rot = rot_x @ rot_y                 # box frame -> view frame
    ext = np.abs(rot) @ half            # half extents of the turned box along the view axes
    h = px * _SS
    w = max(8, int(round(px * ext[0] / ext[1]))) * _SS
    xs = ((np.arange(w) + 0.5) / w * 2.0 - 1.0) * ext[0]
    ys = ((np.arange(h) + 0.5) / h * 2.0 - 1.0) * ext[1]
    gx, gy = np.meshgrid(xs, ys)
    z0 = -10.0                           # rays start well in front of the box and travel along +z
    origin = np.dstack([gx, gy, np.full_like(gx, z0)]) @ rot   # ray origins in the box frame
    direction = rot[2, :]                                        # the +z view axis in the box frame
    t_lo = np.empty_like(origin)
    t_hi = np.empty_like(origin)
    for i in range(3):                   # slab test, one pair of faces at a time
        if abs(direction[i]) < 1e-9:
            inside = np.abs(origin[..., i]) <= half[i]
            t_lo[..., i] = np.where(inside, -np.inf, np.inf)
            t_hi[..., i] = np.where(inside, np.inf, -np.inf)
        else:
            a = (-half[i] - origin[..., i]) / direction[i]
            b = (half[i] - origin[..., i]) / direction[i]
            t_lo[..., i] = np.minimum(a, b)
            t_hi[..., i] = np.maximum(a, b)
    t_near = t_lo.max(axis=-1)
    hit = t_near <= t_hi.min(axis=-1)
    axis = t_lo.argmax(axis=-1)          # which face the ray entered through
    normals_box = np.zeros_like(origin)
    for i in range(3):
        normals_box[..., i] = np.where(axis == i, -np.sign(direction[i]), 0.0)
    normals = normals_box @ rot.T        # back to the view frame
    rgb = _shade(normals.astype(np.float32), color, specular=0.08)
    relief = np.where(hit, z0 + t_near, 0.0)
    return _finish(rgb, hit, relief.astype(np.float32), 2.0 * float(ext[1]), name)


PRIMITIVES: dict[str, Callable[..., Sprite]] = {
    "sphere": make_sphere,
    "box": make_box,
    "cylinder": make_cylinder,
}
