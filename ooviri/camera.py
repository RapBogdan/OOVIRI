"""Pinhole camera: how many pixels a metre covers at a given depth."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Camera:
    """Focal lengths in pixels plus the image size they refer to."""

    fx: float
    fy: float
    width: int
    height: int

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float = 60.0) -> "Camera":
        """Camera for a photo with unknown intrinsics, from an assumed horizontal field of view."""
        f = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
        return cls(fx=f, fy=f, width=width, height=height)

    @classmethod
    def from_focal35(cls, width: int, height: int, focal35_mm: float = 24.0) -> "Camera":
        """Camera of a phone photo, from the lens's 35 mm-equivalent focal length in the phone's spec sheet.

        The equivalence is defined on the image diagonal; a full-frame sensor has a 43.27 mm diagonal.
        """
        f = focal35_mm * math.hypot(width, height) / 43.2666
        return cls(fx=f, fy=f, width=width, height=height)

    def resized(self, width: int, height: int) -> "Camera":
        """Same camera after the image was resized."""
        return Camera(self.fx * width / self.width, self.fy * height / self.height, width, height)

    def pixels_per_metre(self, z: float) -> float:
        """An object 1 m tall at depth z covers this many pixels (perspective: farther = smaller)."""
        if z <= 0:
            raise ValueError("depth must be positive")
        return self.fy / z


# Kinect RGB intrinsics of NYU Depth V2 (from the dataset toolbox, camera_params.m).
NYU = Camera(fx=518.8579, fy=519.4696, width=640, height=480)
