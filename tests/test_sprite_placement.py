import numpy as np
import pytest

from ooviri.camera import NYU, Camera
from ooviri.placement import Placement, place, random_placements, sprite_pixel_size
from ooviri.sprite import PRIMITIVES, Sprite, make_box, make_sphere


@pytest.mark.parametrize("name", list(PRIMITIVES))
def test_primitives_are_valid_sprites(name):
    s = PRIMITIVES[name](px=64)
    alpha = s.rgba[..., 3]
    assert s.rgba.dtype == np.uint8 and s.rgba.shape[2] == 4
    assert s.relief.shape == s.rgba.shape[:2] and s.relief.dtype == np.float32
    assert alpha.max() == 255 and alpha[0, 0] == 0      # opaque somewhere, transparent border
    assert np.isfinite(s.relief).all() and np.abs(s.relief).max() < 1.0
    assert s.height_m > 0


def test_sphere_bulges_towards_camera():
    s = make_sphere(radius_m=0.2, px=64)
    h, w = s.relief.shape
    assert s.relief[h // 2, w // 2] == pytest.approx(-0.2, abs=0.01)   # centre is the closest point
    assert abs(s.relief[h // 2, 2]) < 0.1                               # the outline sits near the centre plane
    assert (s.relief[s.rgba[..., 3] > 0] <= 1e-6).all()                 # nothing behind the centre plane


def test_box_shows_three_differently_lit_faces():
    s = make_box(px=96)
    opaque = s.rgba[..., 3] == 255
    shades = np.unique(s.rgba[opaque][:, 1])
    assert len(shades) >= 3


def test_save_load_roundtrip(tmp_path):
    s = make_box(px=48)
    s.save(tmp_path)
    t = Sprite.load(tmp_path, "box")
    assert np.array_equal(s.rgba, t.rgba)
    assert np.allclose(s.relief, t.relief)
    assert t.height_m == pytest.approx(s.height_m)


def test_farther_means_smaller():
    s = make_sphere(px=64)
    _, h1 = sprite_pixel_size(s, Placement(320, 240, 1.0), NYU)
    _, h2 = sprite_pixel_size(s, Placement(320, 240, 2.0), NYU)
    assert h1 == pytest.approx(s.height_m * NYU.fy, abs=1)
    assert h1 == pytest.approx(2 * h2, abs=1)


def test_place_gives_colour_alpha_and_depth_in_the_image_frame():
    p = place(make_sphere(radius_m=0.2, px=64), Placement(320, 240, 2.0), NYU)
    assert p.rgb.shape == (480, 640, 3) and p.alpha.shape == (480, 640)
    assert p.footprint[240, 320] and not p.footprint[5, 5]
    assert p.depth[240, 320] == pytest.approx(1.8, abs=0.02)   # centre plane 2.0 m minus the 0.2 m bulge
    assert np.isinf(p.depth[5, 5])
    assert 0.0 <= p.alpha.min() and p.alpha.max() <= 1.0
    x0, y0, x1, y1 = p.bbox
    assert x0 < 320 < x1 and y0 < 240 < y1


def test_scale_multiplies_size_and_relief():
    s = make_sphere(radius_m=0.2, px=64)
    p1 = place(s, Placement(320, 240, 2.0), NYU)
    p2 = place(s, Placement(320, 240, 2.0, scale=2.0), NYU)
    assert p2.footprint.sum() == pytest.approx(4 * p1.footprint.sum(), rel=0.08)
    assert p2.depth[240, 320] == pytest.approx(1.6, abs=0.03)


def test_place_clips_at_the_border_and_survives_off_screen():
    p = place(make_sphere(px=64), Placement(2, 2, 1.5), NYU)
    assert p.footprint.any() and p.bbox[:2] == (0, 0)
    q = place(make_sphere(px=64), Placement(-500, -500, 1.5), NYU)
    assert not q.footprint.any()


def test_random_placements_are_reproducible_and_inside_the_image():
    depth = np.full((480, 640), 3.0, np.float32)
    depth[:, :320] = 1.5
    s = make_sphere(px=32)
    a = random_placements(depth, s, NYU, 5, seed=7)
    b = random_placements(depth, s, NYU, 5, seed=7)
    assert a == b and len(a) == 5
    assert all(0 < p.x < 640 and 0 < p.y < 480 and p.z > 0 for p in a)
    assert random_placements(depth, s, NYU, 5, seed=8) != a


def test_camera_helpers():
    assert Camera.from_fov(1000, 750, 90.0).fx == pytest.approx(500.0)
    assert NYU.resized(320, 240).fy == pytest.approx(NYU.fy / 2)
    with pytest.raises(ValueError):
        NYU.pixels_per_metre(0.0)


def test_phone_camera_from_equivalent_focal_length():
    """A 24 mm-equivalent lens on a 3:4 photo: about 57 degrees across, 72 degrees top to bottom."""
    import math
    cam = Camera.from_focal35(1500, 2000, 24.0)
    assert cam.fx == pytest.approx(1386.8, abs=1.0)
    assert math.degrees(2 * math.atan(750 / cam.fx)) == pytest.approx(56.8, abs=0.3)
    assert math.degrees(2 * math.atan(1000 / cam.fy)) == pytest.approx(71.6, abs=0.3)
    # the field of view does not depend on the resolution of the copy we work on
    small = Camera.from_focal35(1200, 1600, 24.0)
    assert small.fx / 1200 == pytest.approx(cam.fx / 1500)
