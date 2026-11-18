"""A synthetic room and three simulated depth models, for development and tests only.

Nothing here is a result of the project: the scene is drawn by code and the "models"
are the exact depth plus the kinds of error real estimators make (a scale bias,
blurred edges, slow drift). It lets the P2 pipeline run before any real depth map exists.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

from .camera import NYU, Camera


def make_scene(seed: int = 0, camera: Camera = NYU) -> tuple[np.ndarray, np.ndarray]:
    """A room with a back wall, a floor and four boxes. Returns (rgb uint8, exact depth in metres)."""
    rng = np.random.default_rng(seed)
    H, W = camera.height, camera.width
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wall_z, cam_h, horizon = 4.5, 1.4, 0.40 * H

    depth = np.full((H, W), wall_z, np.float32)
    z_floor = cam_h * camera.fy / np.maximum(yy - horizon, 1e-3)
    floor = (yy > horizon) & (z_floor < wall_z)
    depth[floor] = z_floor[floor]

    rgb = np.empty((H, W, 3), np.float32)
    rgb[:] = np.array([206, 199, 186], np.float32) * (0.92 + 0.08 * (1.0 - yy / H))[..., None]
    world_x = (xx - W / 2.0) * depth / camera.fx
    checker = (np.floor(world_x / 0.35) % 2) != (np.floor(depth / 0.6) % 2)
    floor_rgb = np.where(checker[..., None], np.array([150, 112, 78], np.float32),
                         np.array([170, 130, 94], np.float32))
    fade = np.clip(1.15 - depth / 6.0, 0.6, 1.1)
    rgb[floor] = (floor_rgb * fade[..., None])[floor]

    boxes = [  # depth, centre x, width, height (metres), colour; drawn far to near
        (3.6, -1.15, 1.10, 1.70, (118, 92, 66)),
        (2.7, 0.75, 1.20, 0.80, (72, 112, 142)),
        (2.0, 1.00, 0.45, 1.10, (96, 140, 96)),
        (1.6, -0.20, 0.24, 1.55, (188, 72, 60)),
    ]
    for z, xc, w, h, col in boxes:
        z += rng.uniform(-0.05, 0.05)
        xc += rng.uniform(-0.05, 0.05)
        y_bottom = horizon + cam_h * camera.fy / z
        y_top = y_bottom - h * camera.fy / z
        r0, r1 = int(max(0, round(y_top))), int(min(H, round(y_bottom)))
        c0 = int(max(0, round(W / 2.0 + (xc - w / 2.0) * camera.fx / z)))
        c1 = int(min(W, round(W / 2.0 + (xc + w / 2.0) * camera.fx / z)))
        if r1 <= r0 or c1 <= c0:
            continue
        region = (slice(r0, r1), slice(c0, c1))
        closer = depth[region] > z
        shade = np.linspace(1.08, 0.86, r1 - r0, dtype=np.float32)[:, None, None]
        colour = np.asarray(col, np.float32)[None, None, :] * shade * np.ones((1, c1 - c0, 1), np.float32)
        rgb[region] = np.where(closer[..., None], colour, rgb[region])
        depth[region] = np.where(closer, z, depth[region])

    rgb += rng.normal(0.0, 2.0, rgb.shape)
    return np.clip(rgb, 0, 255).astype(np.uint8), depth


def gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with edge padding."""
    r = max(1, int(round(3 * sigma)))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    p = np.pad(a, ((r, r), (0, 0)), mode="edge")
    a1 = np.tensordot(sliding_window_view(p, 2 * r + 1, axis=0), k, axes=([-1], [0]))
    p = np.pad(a1, ((0, 0), (r, r)), mode="edge")
    return np.tensordot(sliding_window_view(p, 2 * r + 1, axis=1), k, axes=([-1], [0])).astype(np.float32)


def _slow_field(shape: tuple[int, int], rng: np.random.Generator, amplitude: float) -> np.ndarray:
    """A smooth random field in [-amplitude, amplitude], changing slowly across the image."""
    coarse = rng.uniform(-amplitude, amplitude, (5, 7)).astype(np.float32)
    img = Image.fromarray(coarse).resize((shape[1], shape[0]), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32)


def simulate_models(depth: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    """Three imperfect copies of the exact depth, each wrong in its own way."""
    rng = np.random.default_rng(seed + 1000)
    noise = lambda s: rng.normal(0.0, s, depth.shape).astype(np.float32)  # noqa: E731
    return {
        # sharp edges, but a leftover scale and shift error after alignment
        "sim_bias": depth * 1.05 - 0.04 + noise(0.010),
        # right on average, but depth edges are smeared
        "sim_blur": gaussian_blur(depth, sigma=6.0) + noise(0.010),
        # right at edges, but drifts by a few percent across the image
        "sim_drift": depth * (1.0 + _slow_field(depth.shape, rng, 0.09)) + noise(0.015),
    }
