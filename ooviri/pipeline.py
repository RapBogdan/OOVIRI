"""The whole P2 chain for one image: place, test occlusion per model, vote, compose, save."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .camera import Camera
from .compose import compose, crop, grid, overlay_mask, vote_overlay
from .depthio import colorize_depth, resize_depth, save_image
from .metrics import f1, iou
from .occlusion import margin, occlusion_mask, valid_depth
from .placement import PlacedSprite, Placement, place
from .sprite import Sprite
from .voting import majority_vote, same_as_majority, soft_weighted_vote, vote_count, weighted_vote

ROUTE_LABELS = {
    "A": "Ruta A · vot majoritar",
    "B": "Ruta B · vot ponderat",
    "B_soft": "Ruta B-soft · vot cu margine",
}


@dataclass
class ModelPrior:
    """What we know about a depth model from the validation subset (numbers supplied by P1)."""

    weight: float = 1.0    # its accuracy; the weight of its vote in routes B and B-soft
    abs_rel: float = 0.06  # its typical relative depth error (AbsRel); sets how sure a vote is in B-soft


def load_priors(path: str | Path) -> dict[str, ModelPrior]:
    """Read {"model": {"weight": 0.9, "abs_rel": 0.05}, ...} from a JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: ModelPrior(weight=float(v.get("weight", 1.0)), abs_rel=float(v.get("abs_rel", 0.06)))
            for name, v in raw.items()}


@dataclass
class RunResult:
    placed: PlacedSprite
    placement: Placement
    model_masks: dict[str, np.ndarray]
    route_masks: dict[str, np.ndarray]
    composites: dict[str, np.ndarray]
    counts: np.ndarray                      # number of models voting "occluded" per pixel
    priors: dict[str, ModelPrior]
    b_equals_a: bool                        # route B cannot differ from route A with these weights
    gt_mask: np.ndarray | None = None
    scores: dict[str, dict[str, float]] = field(default_factory=dict)


def run(rgb: np.ndarray, depths: dict[str, np.ndarray], sprite: Sprite, placement: Placement,
        camera: Camera | None = None, priors: dict[str, ModelPrior] | None = None,
        gt_depth: np.ndarray | None = None, eps: float = 0.0) -> RunResult:
    """Run every step for one image and one placement.

    depths maps a model name to its depth map in metres, aligned to the image.
    gt_depth, when given, produces the reference mask and the IoU / F1 self-check.
    """
    if not depths:
        raise ValueError("at least one depth map is needed")
    H, W = rgb.shape[:2]
    if camera is None:
        camera = Camera.from_fov(W, H)
    elif (camera.width, camera.height) != (W, H):
        camera = camera.resized(W, H)
    placed = place(sprite, placement, camera)

    names = list(depths)
    pri = {n: (priors or {}).get(n, ModelPrior()) for n in names}
    model_masks: dict[str, np.ndarray] = {}
    margins, taus = [], []
    for n in names:
        d = np.asarray(depths[n], dtype=np.float32)
        if d.shape != (H, W):
            d = resize_depth(d, (H, W))
        model_masks[n] = occlusion_mask(d, placed, eps)
        margins.append(margin(d, placed))
        # a model's error grows with distance: typical error = relative error x depth
        taus.append(np.maximum(pri[n].abs_rel * np.where(valid_depth(d), d, 1.0), 1e-3))

    weights = [pri[n].weight for n in names]
    masks = [model_masks[n] for n in names]
    route_masks = {
        "A": majority_vote(masks),
        "B": weighted_vote(masks, weights),
        "B_soft": soft_weighted_vote(margins, weights, taus) & placed.footprint,
    }

    composites = {"fara_ocluzie": compose(rgb, placed)}
    for n in names:
        composites[n] = compose(rgb, placed, model_masks[n])
    for r, m in route_masks.items():
        composites[f"ruta_{r}"] = compose(rgb, placed, m)

    result = RunResult(placed=placed, placement=placement, model_masks=model_masks, route_masks=route_masks,
                       composites=composites, counts=vote_count(masks), priors=pri,
                       b_equals_a=same_as_majority(weights))

    if gt_depth is not None:
        gt = np.asarray(gt_depth, dtype=np.float32)
        if gt.shape != (H, W):
            gt = resize_depth(gt, (H, W))
        result.gt_mask = occlusion_mask(gt, placed, eps)
        result.composites["referinta"] = compose(rgb, placed, result.gt_mask)
        region = placed.footprint & valid_depth(gt)   # score only where the reference depth exists
        for name, m in {**model_masks, **{f"ruta_{r}": v for r, v in route_masks.items()}}.items():
            result.scores[name] = {"iou": iou(m, result.gt_mask, region), "f1": f1(m, result.gt_mask, region)}
    return result


def _with_score(label: str, key: str, result: RunResult) -> str:
    s = result.scores.get(key)
    return f"{label} · IoU {s['iou']:.2f}" if s else label


def save_outputs(result: RunResult, rgb: np.ndarray, outdir: str | Path, stem: str,
                 depths: dict[str, np.ndarray] | None = None,
                 gt_depth: np.ndarray | None = None) -> dict[str, Path]:
    """Write the masks (binary PNG), the AR images, three overview panels and a JSON with the scores."""
    out = Path(outdir)
    paths: dict[str, Path] = {}
    names = list(result.model_masks)

    for n in names:
        paths[f"mask_{n}"] = save_image(out / "masti" / f"{stem}__{n}.png", result.model_masks[n])
    for r, m in result.route_masks.items():
        paths[f"mask_ruta_{r}"] = save_image(out / "masti" / f"{stem}__ruta_{r}.png", m)
    if result.gt_mask is not None:
        paths["mask_referinta"] = save_image(out / "masti" / f"{stem}__referinta.png", result.gt_mask)
    for key, img in result.composites.items():
        paths[f"ar_{key}"] = save_image(out / "ar" / f"{stem}__{key}.png", img)

    # Panel 1: the AR images side by side (no occlusion / each model / each route / reference).
    images = [result.composites["fara_ocluzie"]] + [result.composites[n] for n in names]
    labels = ["Fără ocluzie"] + [_with_score(n, n, result) for n in names]
    for r in result.route_masks:
        images.append(result.composites[f"ruta_{r}"])
        labels.append(_with_score(ROUTE_LABELS[r], f"ruta_{r}", result))
    if "referinta" in result.composites:
        images.append(result.composites["referinta"])
        labels.append("Referință · adâncime reală")
    paths["panel_ar"] = save_image(out / "panouri" / f"{stem}__ar.png", grid(images, labels, cols=4))

    # Panel 2: a close-up of the object, with the occluded pixels in red, plus the vote map.
    naive = result.composites["fara_ocluzie"]
    bbox = result.placed.bbox
    zoom = [crop(overlay_mask(naive, result.model_masks[n]), bbox) for n in names]
    zoom_labels = list(names)
    zoom.append(crop(vote_overlay(naive, result.counts, result.placed.footprint, len(names)), bbox))
    zoom_labels.append("Harta voturilor")
    for r, m in result.route_masks.items():
        zoom.append(crop(overlay_mask(naive, m), bbox))
        zoom_labels.append(ROUTE_LABELS[r].split(" · ")[0])
    if result.gt_mask is not None:
        zoom.append(crop(overlay_mask(naive, result.gt_mask), bbox))
        zoom_labels.append("Referință")
    if zoom[0].size:
        paths["panel_voturi"] = save_image(out / "panouri" / f"{stem}__voturi.png", grid(zoom, zoom_labels, cols=4))

    # Panel 3: the depth maps the decisions were based on, all on the same colour scale.
    if depths:
        maps = {n: np.asarray(depths[n], np.float32) for n in names}
        if gt_depth is not None:
            maps["referință (adâncime reală)"] = np.asarray(gt_depth, np.float32)
        good = np.concatenate([d[valid_depth(d)].ravel() for d in maps.values()])
        lo, hi = np.percentile(good, [1, 99])
        paths["panel_adancime"] = save_image(
            out / "panouri" / f"{stem}__adancime.png",
            grid([colorize_depth(d, lo, hi) for d in maps.values()], list(maps), cols=4))

    report = {
        "stem": stem,
        "placement": result.placement.to_dict(),
        "priors": {n: asdict(p) for n, p in result.priors.items()},
        "ruta_B_identica_cu_A": result.b_equals_a,
        "pixeli_obiect": int(result.placed.footprint.sum()),
        "pixeli_ocluzati": {**{n: int(m.sum()) for n, m in result.model_masks.items()},
                            **{f"ruta_{r}": int(m.sum()) for r, m in result.route_masks.items()}},
        "scoruri": result.scores,
    }
    out.mkdir(parents=True, exist_ok=True)
    paths["raport"] = out / f"{stem}__raport.json"
    paths["raport"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths
