"""Tape-measure control points for our own photos, where no ground-truth depth exists.

A depth map stores, for every pixel, the distance along the camera's viewing axis.
A tape measure gives the distance along the floor from the spot under the phone.
`axis_depth` converts one into the other, so a model's metric depth can be checked
on a photo at a handful of measured points.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .camera import Camera


@dataclass(frozen=True)
class ControlPoint:
    name: str
    pixel: tuple[float, float]  # (x, y) in the photo
    depth: float                # expected depth along the camera axis, metres
    approximate: bool = False   # True when the tape distance was taken as the depth directly


def axis_depth(pixel: tuple[float, float], floor_distance: float, target_height: float,
               phone_height: float, camera: Camera) -> float:
    """Depth along the camera axis of a point measured with a tape.

    floor_distance  metres along the floor, from the spot under the phone to the spot under the point
    target_height   height of the point above the floor (0 for a point on the floor)
    phone_height    height of the phone above the floor

    The straight line from the lens to the point is the hypotenuse of those two; the
    pixel tells how far that line leans away from the viewing axis, whatever the tilt
    of the phone was. Exact for points in front of the phone, a little off to the sides.
    """
    line = math.hypot(floor_distance, phone_height - target_height)
    u = (pixel[0] - camera.width / 2.0) / camera.fx
    v = (pixel[1] - camera.height / 2.0) / camera.fy
    return line / math.sqrt(1.0 + u * u + v * v)


def implied_pitch_deg(pixel: tuple[float, float], floor_distance: float, target_height: float,
                      phone_height: float, camera: Camera) -> float:
    """Downward tilt of the phone that one measured point implies.

    Several points of the same photo should agree; if they do, the measurements, the
    phone height and the focal length are consistent with each other.
    """
    below_horizon = math.atan2(phone_height - target_height, floor_distance)
    below_axis = math.atan((pixel[1] - camera.height / 2.0) / camera.fy)
    return math.degrees(below_horizon - below_axis)


def load_control_points(path: str | Path, photo: str) -> tuple[list[ControlPoint], Camera, list[float]]:
    """Read one photo's points from the measurements file.

    Returns the points with their expected depth, the phone camera for that photo and
    the tilt implied by each floor-measured point (for the consistency check).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if photo not in data["poze"]:
        raise KeyError(f"{photo} nu apare in {path}; exista: {', '.join(data['poze'])}")
    entry = data["poze"][photo]
    width, height = entry["dimensiune"]
    camera = Camera.from_focal35(width, height, float(data["telefon"]["focal35"]))
    phone_height = entry.get("inaltime_telefon")
    points: list[ControlPoint] = []
    pitches: list[float] = []
    for p in entry.get("puncte", []):
        pixel = (float(p["pixel"][0]), float(p["pixel"][1]))
        if "distanta_podea" in p:
            if phone_height is None:
                raise ValueError(f"{photo}: punctul '{p['nume']}' are nevoie de inaltime_telefon")
            args = (pixel, float(p["distanta_podea"]), float(p.get("inaltime", 0.0)), float(phone_height), camera)
            points.append(ControlPoint(p["nume"], pixel, axis_depth(*args)))
            pitches.append(implied_pitch_deg(*args))
        else:
            points.append(ControlPoint(p["nume"], pixel, float(p["adancime"]), approximate=True))
    return points, camera, pitches


def sample_depth(depth: np.ndarray, pixel: tuple[float, float], photo_size: tuple[int, int],
                 window: int = 4) -> float:
    """Median depth around a pixel given in photo coordinates; the map may have another resolution."""
    h, w = depth.shape
    x = int(round(pixel[0] * w / photo_size[0]))
    y = int(round(pixel[1] * h / photo_size[1]))
    patch = depth[max(0, y - window):y + window + 1, max(0, x - window):x + window + 1]
    good = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(good)) if good.size else float("nan")


@dataclass(frozen=True)
class Calibration:
    """Correction of a depth map from tape-measured points: true = scale * predicted + offset."""

    scale: float
    offset: float
    kind: str      # "afin" (scale and offset) or "scara" (scale only)
    n_points: int

    def apply(self, depth: np.ndarray) -> np.ndarray:
        return np.maximum(self.scale * np.asarray(depth, np.float32) + self.offset, 0.05).astype(np.float32)


def fit_calibration(predicted, true, min_spread: float = 0.15) -> Calibration:
    """Least-squares correction from pairs (depth in the map, depth measured with the tape).

    Scale and offset need at least two points that are clearly apart in depth, both in
    the map and in reality; with one point, or points at nearly the same depth, a line
    through them would be arbitrary, so only the scale is fitted.
    """
    p, t = np.asarray(predicted, np.float64), np.asarray(true, np.float64)
    ok = np.isfinite(p) & (p > 0) & np.isfinite(t) & (t > 0)
    p, t = p[ok], t[ok]
    if p.size == 0:
        raise ValueError("no usable control point")
    if p.size >= 2 and np.ptp(p) / p.mean() >= min_spread and np.ptp(t) / t.mean() >= min_spread:
        a, b = np.polyfit(p, t, 1)
        if a > 0:
            return Calibration(float(a), float(b), "afin", int(p.size))
    return Calibration(float((p * t).sum() / (p * p).sum()), 0.0, "scara", int(p.size))


def calibrate_depth(depth: np.ndarray, points: list[ControlPoint],
                    photo_size: tuple[int, int]) -> tuple[np.ndarray, Calibration]:
    """Correct one depth map with the photo's control points."""
    predicted = [sample_depth(depth, pt.pixel, photo_size) for pt in points]
    cal = fit_calibration(predicted, [pt.depth for pt in points])
    return cal.apply(depth), cal


def leave_one_out(depth: np.ndarray, points: list[ControlPoint], photo_size: tuple[int, int]) -> list[float] | None:
    """Honest error of the calibration: fit on all points but one, test on the one left out.

    Returns the relative error at each point, or None with fewer than three points
    (nothing would be left to test on).
    """
    if len(points) < 3:
        return None
    predicted = [sample_depth(depth, pt.pixel, photo_size) for pt in points]
    errors = []
    for i, pt in enumerate(points):
        rest_p = [p for j, p in enumerate(predicted) if j != i]
        rest_t = [q.depth for j, q in enumerate(points) if j != i]
        cal = fit_calibration(rest_p, rest_t)
        errors.append((cal.scale * predicted[i] + cal.offset - pt.depth) / pt.depth)
    return errors
