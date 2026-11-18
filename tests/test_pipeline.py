import json

import numpy as np
import pytest
from PIL import Image

from ooviri.camera import NYU
from ooviri.cli import main
from ooviri.depthio import colorize_depth, load_depth
from ooviri.pipeline import ModelPrior, load_priors, run, save_outputs
from ooviri.placement import Placement
from ooviri.sprite import make_sphere
from ooviri.synthetic import make_scene, simulate_models


@pytest.fixture(scope="module")
def scene():
    rgb, gt = make_scene(0)
    return rgb, gt, simulate_models(gt, 0)


def test_end_to_end_on_the_synthetic_room(scene, tmp_path):
    rgb, gt, depths = scene
    res = run(rgb, depths, make_sphere(px=64), Placement(300, 300, 2.0), camera=NYU, gt_depth=gt)

    assert set(res.route_masks) == {"A", "B", "B_soft"}
    masks = list(res.model_masks.values())
    everyone, anyone = np.logical_and.reduce(masks), np.logical_or.reduce(masks)
    for m in res.route_masks.values():
        assert not (m & ~res.placed.footprint).any()                  # votes only on the object's pixels
    assert not (everyone & ~res.route_masks["A"]).any()               # unanimous "hidden" stays hidden
    assert not (res.route_masks["A"] & ~anyone).any()                 # nobody said "hidden" -> visible
    assert res.b_equals_a and np.array_equal(res.route_masks["A"], res.route_masks["B"])
    assert 0 < res.gt_mask.sum() < res.placed.footprint.sum()         # the case is a partial occlusion
    assert res.scores["ruta_A"]["iou"] > 0.9

    paths = save_outputs(res, rgb, tmp_path, "t", depths=depths, gt_depth=gt)
    assert all(p.exists() for p in paths.values())
    assert set(np.unique(np.array(Image.open(paths["mask_ruta_A"])))) <= {0, 255}   # binary PNG
    report = json.loads(paths["raport"].read_text(encoding="utf-8"))
    assert report["ruta_B_identica_cu_A"] is True
    assert report["placement"]["z"] == 2.0


def test_naive_composite_draws_the_object_and_full_occlusion_leaves_the_photo_untouched():
    rgb = np.full((480, 640, 3), 200, np.uint8)
    sprite, placement = make_sphere(px=64), Placement(320, 240, 2.0)
    far = run(rgb, {"m": np.full((480, 640), 5.0, np.float32)}, sprite, placement, camera=NYU)
    near = run(rgb, {"m": np.full((480, 640), 1.0, np.float32)}, sprite, placement, camera=NYU)
    assert not np.array_equal(far.composites["fara_ocluzie"], rgb)
    assert np.array_equal(far.composites["ruta_A"], far.composites["fara_ocluzie"])
    assert np.array_equal(near.composites["ruta_A"], rgb)


def test_depth_maps_of_another_size_are_resized(scene):
    rgb, gt, depths = scene
    small = {n: d[::2, ::2] for n, d in depths.items()}
    res = run(rgb, small, make_sphere(px=64), Placement(300, 300, 2.0), camera=NYU)
    assert res.route_masks["A"].shape == rgb.shape[:2] and res.route_masks["A"].any()


def test_priors_change_the_weighted_routes(scene, tmp_path):
    rgb, gt, depths = scene
    file = tmp_path / "priors.json"
    file.write_text(json.dumps({"sim_blur": {"weight": 9.0, "abs_rel": 0.02}}), encoding="utf-8")
    priors = load_priors(file)
    assert priors["sim_blur"] == ModelPrior(weight=9.0, abs_rel=0.02)
    res = run(rgb, depths, make_sphere(px=64), Placement(300, 300, 2.0), camera=NYU, priors=priors)
    assert not res.b_equals_a
    assert np.array_equal(res.route_masks["B"], res.model_masks["sim_blur"])   # the heavy model decides alone


def test_load_depth_reads_metres_from_npy_and_millimetres_from_png(tmp_path):
    np.save(tmp_path / "d.npy", np.full((4, 5), 2.5, np.float32))
    Image.fromarray(np.full((4, 5), 2500, np.uint16)).save(tmp_path / "d.png")
    assert load_depth(tmp_path / "d.npy")[0, 0] == pytest.approx(2.5)
    assert load_depth(tmp_path / "d.png")[0, 0] == pytest.approx(2.5)


def test_colorize_depth_marks_holes_black():
    depth = np.array([[1.0, 2.0, 0.0]], np.float32)
    img = colorize_depth(depth)
    assert img.shape == (1, 3, 3) and (img[0, 2] == 0).all()
    assert img[0, 0].sum() > img[0, 1].sum()        # near is brighter than far


def test_cli_demo(tmp_path):
    assert main(["demo", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "panouri" / "sfera_dupa_stalp__ar.png").exists()
    assert (tmp_path / "masti" / "sfera_prin_stalp__ruta_B_soft.png").exists()


def test_cli_run_on_files(scene, tmp_path):
    rgb, gt, depths = scene
    Image.fromarray(rgb).save(tmp_path / "img.png")
    args = ["run", "--rgb", str(tmp_path / "img.png"), "--z", "2.0", "--x", "300", "--y", "300",
            "--out", str(tmp_path / "out")]
    for name, d in depths.items():
        np.save(tmp_path / f"{name}.npy", d)
        args += ["--depth", f"{name}={tmp_path / (name + '.npy')}"]
    np.save(tmp_path / "gt.npy", gt)
    assert main(args + ["--gt", str(tmp_path / "gt.npy")]) == 0
    assert (tmp_path / "out" / "panouri" / "img__ar.png").exists()


def test_cli_preview_draws_the_object_without_occlusion(scene, tmp_path):
    rgb, _, _ = scene
    Image.fromarray(rgb).save(tmp_path / "img.png")
    out = tmp_path / "p" / "preview.png"
    assert main(["preview", "--rgb", str(tmp_path / "img.png"), "--z", "2.0", "--focal35", "24",
                 "--sprite", "box", "--out", str(out)]) == 0
    drawn = np.array(Image.open(out))
    assert drawn.shape == rgb.shape and not np.array_equal(drawn, rgb)


def test_cli_sprites_and_placements(scene, tmp_path):
    _, gt, _ = scene
    assert main(["sprites", "--out", str(tmp_path / "s")]) == 0
    assert (tmp_path / "s" / "sphere.png").exists() and (tmp_path / "s" / "box_relief.npy").exists()
    np.save(tmp_path / "gt.npy", gt)
    assert main(["placements", "--depth", str(tmp_path / "gt.npy"), "--n", "4", "--out", str(tmp_path / "p.json")]) == 0
    assert len(json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))) == 4
    assert main(["run", "--rgb", str(tmp_path / "s" / "sphere.png"), "--z", "2", "--sprite", f"{tmp_path / 's'}:box",
                 "--depth", f"m={tmp_path / 'gt.npy'}", "--out", str(tmp_path / "o")]) == 0
