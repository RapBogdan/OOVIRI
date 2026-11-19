import json
import math

import numpy as np
import pytest

from ooviri.camera import Camera
from ooviri.cli import main
from ooviri.measure import (Calibration, ControlPoint, axis_depth, calibrate_depth, fit_calibration,
                            implied_pitch_deg, leave_one_out, load_control_points, sample_depth)

CAM = Camera.from_focal35(1500, 2000, 24.0)


def project(forward, lateral, height, phone_height, pitch_deg):
    """Pixel and true axis depth of a point, for a phone tilted down by pitch_deg (the reference geometry)."""
    t = math.radians(pitch_deg)
    down = phone_height - height                      # how far below the lens the point is
    z = forward * math.cos(t) + down * math.sin(t)    # along the viewing axis
    y = -forward * math.sin(t) + down * math.cos(t)   # downwards in the image
    return (CAM.width / 2 + CAM.fx * lateral / z, CAM.height / 2 + CAM.fy * y / z), z


@pytest.mark.parametrize("forward, height, phone, pitch", [
    (1.18, 0.0, 0.85, 17.5),    # a chair leg on the floor
    (2.60, 0.0, 1.30, 9.0),     # the base of a door across a hallway
    (1.60, 0.65, 0.85, 17.5),   # a point above the floor
])
def test_axis_depth_matches_the_projection_geometry(forward, height, phone, pitch):
    pixel, z = project(forward, 0.0, height, phone, pitch)
    assert axis_depth(pixel, forward, height, phone, CAM) == pytest.approx(z, rel=1e-6)
    assert implied_pitch_deg(pixel, forward, height, phone, CAM) == pytest.approx(pitch, abs=1e-6)


def test_axis_depth_is_close_for_points_a_little_to_the_side():
    pixel, z = project(1.5, 0.2, 0.0, 0.85, 17.5)
    floor_distance = math.hypot(1.5, 0.2)             # what the tape actually measures
    assert axis_depth(pixel, floor_distance, 0.0, 0.85, CAM) == pytest.approx(z, rel=0.01)


def test_floor_distance_is_not_the_depth():
    """The reason the phone height is needed: 1.18 m along the floor is 1.37 m along the camera axis."""
    pixel, z = project(1.18, 0.0, 0.0, 0.85, 17.5)
    assert z == pytest.approx(1.38, abs=0.02)


@pytest.fixture
def measurements(tmp_path):
    pixel, _ = project(1.18, 0.0, 0.0, 0.85, 17.5)
    data = {
        "telefon": {"model": "test", "focal35": 24},
        "poze": {
            "p": {"dimensiune": [1500, 2000], "inaltime_telefon": 0.85, "puncte": [
                {"nume": "pe podea", "pixel": list(pixel), "distanta_podea": 1.18, "inaltime": 0.0},
                {"nume": "direct", "pixel": [750, 1000], "adancime": 2.0},
            ]},
            "fara_inaltime": {"dimensiune": [1500, 2000], "inaltime_telefon": None, "puncte": [
                {"nume": "x", "pixel": [1, 1], "distanta_podea": 1.0}]},
        },
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_control_points(measurements):
    points, camera, pitches = load_control_points(measurements, "p")
    assert camera.fx == pytest.approx(CAM.fx)
    assert [p.approximate for p in points] == [False, True]
    assert points[0].depth == pytest.approx(1.38, abs=0.02) and points[1].depth == 2.0
    assert pitches == [pytest.approx(17.5, abs=1e-6)]
    with pytest.raises(ValueError, match="inaltime_telefon"):
        load_control_points(measurements, "fara_inaltime")
    with pytest.raises(KeyError):
        load_control_points(measurements, "nu_exista")


def test_sample_depth_scales_to_the_map_resolution_and_skips_holes():
    depth = np.full((200, 150), 3.0, np.float32)      # a map ten times smaller than the photo
    depth[100, 75] = 0.0
    assert sample_depth(depth, (750, 1000), (1500, 2000)) == pytest.approx(3.0)
    assert math.isnan(sample_depth(np.zeros((200, 150), np.float32), (750, 1000), (1500, 2000)))


def test_fit_calibration_recovers_scale_and_offset():
    true = np.array([1.37, 1.78, 2.41])
    cal = fit_calibration(0.96 * true + 0.93, true)             # a map that adds almost a metre
    assert cal.kind == "afin" and cal.n_points == 3
    assert cal.scale * (0.96 * 2.0 + 0.93) + cal.offset == pytest.approx(2.0, abs=1e-6)


def test_fit_calibration_falls_back_to_scale_only():
    one = fit_calibration([2.6], [1.45])
    assert one.kind == "scara" and one.offset == 0.0 and one.scale == pytest.approx(1.45 / 2.6)
    close = fit_calibration([1.80, 1.86], [1.28, 1.53])         # nearly the same depth in the map
    assert close.kind == "scara"
    with pytest.raises(ValueError):
        fit_calibration([float("nan")], [1.0])


def test_calibrate_depth_and_leave_one_out():
    size = (1500, 2000)
    truth = np.tile(np.linspace(1.0, 3.0, 2000, dtype=np.float32)[:, None], (1, 1500))   # depth grows downwards
    wrong = 0.9 * truth + 0.8
    pts = [ControlPoint(f"p{i}", (750.0, float(y)), float(truth[y, 750])) for i, y in enumerate((300, 1000, 1700))]
    fixed, cal = calibrate_depth(wrong, pts, size)
    assert cal.kind == "afin" and np.abs(fixed - truth).max() < 0.01
    errors = leave_one_out(wrong, pts, size)
    assert len(errors) == 3 and max(abs(e) for e in errors) < 0.01
    assert leave_one_out(wrong, pts[:2], size) is None
    assert Calibration(1.0, -5.0, "afin", 2).apply(np.ones((2, 2))).min() == pytest.approx(0.05)   # never below 5 cm


def test_cli_check(measurements, tmp_path, capsys):
    np.save(tmp_path / "d.npy", np.full((2000, 1500), 2.2, np.float32))
    assert main(["check", "--measurements", str(measurements), "--photo", "p",
                 "--depth", f"model={tmp_path / 'd.npy'}"]) == 0
    out = capsys.readouterr().out
    assert "17.5 grade" in out and "model 2.20 m (+10%)" in out
