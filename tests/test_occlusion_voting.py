import itertools

import numpy as np
import pytest

from ooviri.camera import NYU
from ooviri.metrics import f1, iou
from ooviri.occlusion import margin, occlusion_mask
from ooviri.placement import Placement, place
from ooviri.sprite import make_sphere
from ooviri.voting import majority_vote, same_as_majority, soft_weighted_vote, vote_count, weighted_vote


@pytest.fixture(scope="module")
def placed():
    """A 20 cm ball centred in the image, 2 m away: its surface spans 1.8 m (centre) to 2.0 m (outline)."""
    return place(make_sphere(radius_m=0.2, px=64), Placement(320, 240, 2.0), NYU)


def wall(z):
    return np.full((480, 640), z, np.float32)


# ---------------------------------------------------------------- occlusion test
def test_wall_in_front_hides_everything(placed):
    assert np.array_equal(occlusion_mask(wall(1.0), placed), placed.footprint)


def test_wall_behind_hides_nothing(placed):
    assert not occlusion_mask(wall(3.0), placed).any()


def test_half_wall_hides_half(placed):
    depth = wall(3.0)
    depth[:, :320] = 1.0
    m = occlusion_mask(depth, placed)
    assert m.sum() == pytest.approx(placed.footprint.sum() / 2, rel=0.06)
    assert not m[:, 320:].any()


def test_relief_lets_the_centre_poke_through(placed):
    """A wall at 1.85 m cuts the ball: the bulge (1.8 m) stays visible, the rim (about 2.0 m) is hidden."""
    m = occlusion_mask(wall(1.85), placed)
    x0 = placed.bbox[0]
    assert not m[240, 320]
    assert m[240, x0 + 8]
    assert 0 < m.sum() < placed.footprint.sum()


def test_pixels_without_depth_never_occlude(placed):
    depth = wall(1.0)
    depth[230:250, 310:330] = 0.0
    depth[200, 320] = np.nan
    m = occlusion_mask(depth, placed)
    assert not m[240, 320] and not m[200, 320]
    assert m[240, 300]


def test_eps_demands_a_clearer_gap(placed):
    assert occlusion_mask(wall(1.75), placed)[240, 320]
    assert not occlusion_mask(wall(1.75), placed, eps=0.10)[240, 320]


def test_margin_is_the_signed_gap(placed):
    assert margin(wall(1.0), placed)[240, 320] == pytest.approx(0.8, abs=0.02)
    assert margin(wall(3.0), placed)[240, 320] < 0
    assert np.isneginf(margin(wall(1.0), placed)[5, 5])


def test_wrong_size_depth_map_is_reported(placed):
    with pytest.raises(ValueError, match="resize"):
        occlusion_mask(np.ones((10, 10), np.float32), placed)


# ---------------------------------------------------------------- voting
def votes(*bits):
    return [np.array([[bool(b)]]) for b in bits]


def test_route_a_is_two_out_of_three():
    for bits in itertools.product([0, 1], repeat=3):
        assert majority_vote(votes(*bits))[0, 0] == (sum(bits) >= 2)


def test_route_b_with_accuracy_weights_never_differs_from_route_a():
    weights = [0.93, 0.88, 0.71]
    assert same_as_majority(weights)
    for bits in itertools.product([0, 1], repeat=3):
        assert weighted_vote(votes(*bits), weights)[0, 0] == majority_vote(votes(*bits))[0, 0]


def test_route_b_differs_only_when_one_model_dominates():
    weights = [5.0, 1.0, 1.0]
    assert not same_as_majority(weights)
    assert weighted_vote(votes(1, 0, 0), weights)[0, 0]
    assert not weighted_vote(votes(0, 1, 1), weights)[0, 0]


def test_weighted_vote_rejects_bad_weights():
    with pytest.raises(ValueError):
        weighted_vote(votes(1, 0, 1), [1.0, 1.0])
    with pytest.raises(ValueError):
        weighted_vote(votes(1, 0, 1), [1.0, -1.0, 1.0])


def test_soft_vote_lets_a_sure_model_outvote_two_unsure_ones():
    margins = [np.array([[0.50]]), np.array([[-0.01]]), np.array([[-0.01]])]
    assert not majority_vote([m > 0 for m in margins])[0, 0]          # route A: 1 against 2
    assert soft_weighted_vote(margins, [1, 1, 1], [0.1, 0.1, 0.1])[0, 0]


def test_soft_vote_agrees_with_everyone_when_all_are_sure():
    sure_yes = [np.array([[1.0]])] * 3
    sure_no = [np.array([[-1.0]])] * 3
    outside = [np.array([[-np.inf]])] * 3
    assert soft_weighted_vote(sure_yes, [1, 1, 1], [0.1] * 3)[0, 0]
    assert not soft_weighted_vote(sure_no, [1, 1, 1], [0.1] * 3)[0, 0]
    assert not soft_weighted_vote(outside, [1, 1, 1], [0.1] * 3)[0, 0]


def test_vote_count():
    assert vote_count(votes(1, 0, 1))[0, 0] == 2


# ---------------------------------------------------------------- scores
def test_iou_and_f1():
    a = np.array([[1, 1, 0, 0]], bool)
    b = np.array([[1, 0, 1, 0]], bool)
    assert iou(a, b) == pytest.approx(1 / 3)
    assert f1(a, b) == pytest.approx(0.5)
    empty = np.zeros((2, 2), bool)
    assert iou(empty, empty) == 1.0 and f1(empty, empty) == 1.0
    assert iou(a, b, region=np.array([[1, 0, 0, 1]], bool)) == 1.0
