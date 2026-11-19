"""Command line: python -m ooviri <sprites | demo | run | placements> ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .camera import NYU, Camera
from .compose import compose
from .depthio import load_depth, load_rgb, save_image
from .measure import calibrate_depth, leave_one_out, load_control_points, sample_depth
from .pipeline import ModelPrior, load_priors, run, save_outputs
from .placement import Placement, place, random_placements, sprite_pixel_size
from .sprite import PRIMITIVES, Sprite, make_box, make_cylinder, make_sphere
from .synthetic import make_scene, simulate_models


def _sprite(spec: str, color=None) -> Sprite:
    """'sphere' / 'box' / 'cylinder' (optionally in another colour), or 'folder:name' for a saved sprite."""
    if spec in PRIMITIVES:
        return PRIMITIVES[spec](color=tuple(color)) if color else PRIMITIVES[spec]()
    if ":" in spec:
        directory, name = spec.rsplit(":", 1)
        return Sprite.load(directory, name)
    raise SystemExit(f"sprite necunoscut: {spec} (alege din {', '.join(PRIMITIVES)} sau folder:nume)")


def _camera(args, w: int, h: int) -> Camera:
    """--focal35 wins when given; otherwise the NYU Kinect or an assumed field of view."""
    if getattr(args, "focal35", None):
        return Camera.from_focal35(w, h, args.focal35)
    return NYU.resized(w, h) if args.camera == "nyu" else Camera.from_fov(w, h, args.hfov)


def _flat_depth_maps(directory: str, image_id: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Read <dir>/<id>_<model>.npy (the layout written by depth.run); "ruta_*" maps do not vote."""
    voters: dict[str, np.ndarray] = {}
    extras: dict[str, np.ndarray] = {}
    prefix = f"{image_id}_"
    for f in sorted(Path(directory).glob(f"{image_id}_*.npy")):
        name = f.stem[len(prefix):]
        (extras if name.startswith("ruta_") else voters)[name] = load_depth(f)
    return voters, extras


def _print_scores(stem: str, result) -> None:
    footprint = int(result.placed.footprint.sum())
    print(f"\n{stem}: {footprint} pixeli de obiect")
    masks = {**result.model_masks, **{f"ruta_{r}": m for r, m in result.route_masks.items()}, **result.extra_masks}
    for name, mask in masks.items():
        line = f"  {name:<18} ocluzati {int(mask.sum()):>7} ({mask.sum() / max(footprint, 1):.0%})"
        if name in result.scores:
            line += f"   IoU {result.scores[name]['iou']:.3f}   F1 {result.scores[name]['f1']:.3f}"
        print(line)
    disagree = int(((result.counts > 0) & (result.counts < len(result.model_masks))).sum())
    print(f"  modelele se contrazic pe {disagree} pixeli ({disagree / max(footprint, 1):.1%} din obiect)")
    if result.b_equals_a:
        print("  nota: cu aceste ponderi ruta B da mereu aceeasi masca ca ruta A")
    if "ruta_C" in result.extra_masks:
        same = np.array_equal(result.extra_masks["ruta_C"], result.route_masks["A"])
        print(f"  ruta C {'da exact aceeasi masca ca' if same else 'difera de'} ruta A")


def cmd_sprites(args) -> int:
    for make in PRIMITIVES.values():
        print(make().save(args.out))
    return 0


def cmd_demo(args) -> int:
    """End-to-end run on the synthetic room with simulated depth models; needs no external data."""
    rgb, gt = make_scene(args.seed)
    depths = simulate_models(gt, args.seed)
    priors = {}
    for name, d in depths.items():   # stand-in for the numbers P1 measures on the validation subset
        rel = np.abs(d - gt) / gt
        priors[name] = ModelPrior(weight=float((rel < 0.05).mean()), abs_rel=float(rel.mean()))
    cases = [
        # clearly behind the pillar: every model should agree
        ("sfera_dupa_stalp", make_sphere(), Placement(x=300, y=300, z=2.0)),
        # the pillar's front plane cuts through the ball: the bulge pokes out, the models disagree
        ("sfera_prin_stalp", make_sphere(), Placement(x=262, y=300, z=1.72)),
        ("cub_dupa_masa", make_box(), Placement(x=360, y=335, z=3.2)),
        # in front of everything: nothing may be hidden
        ("cilindru_in_fata", make_cylinder(), Placement(x=190, y=340, z=1.25)),
    ]
    for stem, sprite, placement in cases:
        result = run(rgb, depths, sprite, placement, camera=NYU, priors=priors, gt_depth=gt)
        save_outputs(result, rgb, args.out, stem, depths=depths, gt_depth=gt)
        _print_scores(stem, result)
    print(f"\nrezultate in {Path(args.out).resolve()}")
    return 0


def cmd_run(args) -> int:
    rgb = load_rgb(args.rgb)
    depths: dict[str, np.ndarray] = {}
    for item in args.depth or []:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--depth asteapta nume=cale, am primit: {item}")
        depths[name] = load_depth(path)
    if args.depth_root:
        if not args.id:
            raise SystemExit("--depth-root are nevoie si de --id")
        for sub in sorted(Path(args.depth_root).iterdir()):
            candidate = sub / f"{args.id}.npy"
            if sub.is_dir() and candidate.exists():
                depths[sub.name] = load_depth(candidate)
    extras: dict[str, np.ndarray] = {}
    if args.depth_dir:
        if not args.id:
            raise SystemExit("--depth-dir are nevoie si de --id")
        voters, extras = _flat_depth_maps(args.depth_dir, args.id)
        depths.update(voters)
    for item in args.extra or []:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--extra asteapta nume=cale, am primit: {item}")
        extras[name] = load_depth(path)
    if not depths:
        raise SystemExit("nicio harta de adancime: foloseste --depth nume=cale, --depth-dir + --id "
                         "sau --depth-root + --id")
    h, w = rgb.shape[:2]
    camera = _camera(args, w, h)
    placement = Placement(x=args.x if args.x is not None else w / 2, y=args.y if args.y is not None else h / 2,
                          z=args.z, scale=args.scale)
    gt = load_depth(args.gt) if args.gt else None
    priors = load_priors(args.priors) if args.priors else None
    result = run(rgb, depths, _sprite(args.sprite), placement, camera=camera, priors=priors,
                 gt_depth=gt, eps=args.eps, extra_depths=extras)
    stem = args.stem or args.id or Path(args.rgb).stem
    paths = save_outputs(result, rgb, args.out, stem, depths=depths, gt_depth=gt, extra_depths=extras)
    _print_scores(stem, result)
    print(f"\npanou: {paths['panel_ar']}")
    return 0


def cmd_preview(args) -> int:
    """Draw the object over the photo with no occlusion: the naive AR picture, useful for choosing a placement."""
    rgb = load_rgb(args.rgb)
    h, w = rgb.shape[:2]
    camera = _camera(args, w, h)
    sprite = _sprite(args.sprite)
    placement = Placement(x=args.x if args.x is not None else w / 2, y=args.y if args.y is not None else h / 2,
                          z=args.z, scale=args.scale)
    tw, th = sprite_pixel_size(sprite, placement, camera)
    out = save_image(args.out, compose(rgb, place(sprite, placement, camera)))
    print(f"obiect de {tw}x{th} pixeli la {args.z} m -> {out}")
    return 0


def cmd_check(args) -> int:
    """Compare depth maps with the tape-measured control points of one of our own photos."""
    points, camera, pitches = load_control_points(args.measurements, args.photo)
    depths: dict[str, np.ndarray] = {}
    for item in args.depth or []:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--depth asteapta nume=cale, am primit: {item}")
        depths[name] = load_depth(path)
    if args.depth_dir:
        voters, extras = _flat_depth_maps(args.depth_dir, args.photo)
        depths.update(voters)
        depths.update(extras)
    size = (camera.width, camera.height)
    print(f"{args.photo}: {size[0]}x{size[1]} pixeli, distanta focala {camera.fx:.0f} px")
    if pitches:
        print("  inclinarea telefonului dedusa din fiecare punct masurat pe podea: "
              + ", ".join(f"{p:.1f}" for p in pitches) + " grade")
    for pt in points:
        line = f"  {pt.name:<44} asteptat {'~' if pt.approximate else ' '}{pt.depth:.2f} m"
        for name, d in depths.items():
            got = sample_depth(d, pt.pixel, size)
            line += f"   {name} {got:.2f} m ({(got - pt.depth) / pt.depth:+.0%})"
        print(line)
    if not points:
        print("  niciun punct de control masurat")
    elif depths:
        print("  dupa calibrarea pe punctele masurate:")
        for name, d in depths.items():
            cal = calibrate_depth(d, points, size)[1]
            line = f"    {name:<18} {cal.kind}: adevarat = {cal.scale:.2f} x harta {cal.offset:+.2f} m"
            held_out = leave_one_out(d, points, size)
            if held_out:
                line += "   eroare pe punctul lasat deoparte: " + ", ".join(f"{e:+.0%}" for e in held_out)
            else:
                line += "   (prea putine puncte pentru o verificare independenta)"
            print(line)
    return 0


def cmd_batch(args) -> int:
    """Run every placement listed in a JSON file: {"<image id>": [{"nume", "sprite", "x", "y", "z", "scale"}]}."""
    plan = json.loads(Path(args.placements).read_text(encoding="utf-8"))
    priors = load_priors(args.priors) if args.priors else None
    rows = []
    for image_id, cases in plan.items():
        if image_id.startswith("_"):
            continue
        found = [p for p in (Path(args.images) / f"{image_id}{ext}" for ext in (".jpg", ".jpeg", ".png")) if p.exists()]
        if not found:
            raise SystemExit(f"nu gasesc imaginea {image_id} in {args.images}")
        rgb = load_rgb(found[0])
        h, w = rgb.shape[:2]
        depths, extras = _flat_depth_maps(args.depth_dir, image_id)
        if not depths:
            print(f"\n{image_id}: nicio harta de adancime in {args.depth_dir}, sar peste")
            continue
        camera = _camera(args, w, h)
        calibrated = "nu"
        if args.measurements:
            points = load_control_points(args.measurements, image_id)[0]
            if points:
                # each model is corrected with its own fit on the measured points; route C is
                # then the median of the corrected maps, as its definition says
                fits = {}
                for name in depths:
                    depths[name], fits[name] = calibrate_depth(depths[name], points, (w, h))
                if "ruta_C" in extras:
                    extras["ruta_C"] = np.median(np.stack(list(depths.values())), axis=0).astype(np.float32)
                calibrated = next(iter(fits.values())).kind
                print(f"\n{image_id}: calibrat pe {len(points)} puncte masurate ("
                      + ", ".join(f"{n}: x{c.scale:.2f} {c.offset:+.2f} m" for n, c in fits.items()) + ")")
        reference = np.median(np.stack(list(depths.values())), axis=0)
        for case in cases:
            if "z_intre" in case:   # depth chosen between two things seen in the photo
                zi = case["z_intre"]
                near = sample_depth(reference, tuple(zi["fata"]), (w, h))
                far = sample_depth(reference, tuple(zi["spate"]), (w, h))
                z = near + float(zi.get("t", 0.5)) * (far - near)
            else:
                z = float(case["z"])
            placement = Placement(x=float(case["x"]), y=float(case["y"]), z=z, scale=float(case.get("scale", 1.0)))
            result = run(rgb, depths, _sprite(case.get("sprite", "sphere"), case.get("culoare")), placement,
                         camera=camera, priors=priors, extra_depths=extras)
            stem = f"{image_id}__{case['nume']}"
            save_outputs(result, rgb, args.out, stem, depths=depths, extra_depths=extras)
            _print_scores(stem, result)
            print(f"  obiect la {z:.2f} m")
            footprint = int(result.placed.footprint.sum())
            masks = {**result.model_masks, **{f"ruta_{r}": m for r, m in result.route_masks.items()},
                     **result.extra_masks}
            row = {"imagine": image_id, "caz": case["nume"], "calibrat": calibrated, "z_m": round(z, 3),
                   "pixeli_obiect": footprint,
                   "dezacord": int(((result.counts > 0) & (result.counts < len(depths))).sum())}
            row.update({f"ocluzat_{n}": int(m.sum()) for n, m in masks.items()})
            row["B_soft_difera_de_A"] = int((result.route_masks["B_soft"] != result.route_masks["A"]).sum())
            rows.append(row)
    if rows:
        import csv
        out = Path(args.out) / "rezumat.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n{len(rows)} cazuri; rezumat in {out}")
    return 0


def cmd_placements(args) -> int:
    depth = load_depth(args.depth)
    h, w = depth.shape
    camera = _camera(args, w, h)
    items = random_placements(depth, _sprite(args.sprite), camera, args.n, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([p.to_dict() for p in items], indent=2), encoding="utf-8")
    print(f"{len(items)} plasari scrise in {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ooviri", description="OOVIRI: compunere AR cu ocluzie si vot (partea P2)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sprites", help="genereaza sprite-urile primitive (PNG + relief)")
    p.add_argument("--out", default="assets/sprites")
    p.set_defaults(func=cmd_sprites)

    p = sub.add_parser("demo", help="demo complet pe o scena sintetica, fara date externe")
    p.add_argument("--out", default="outputs/demo_sintetic")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_demo)

    def camera_args(q):
        q.add_argument("--camera", choices=["nyu", "fov"], default="nyu",
                       help="nyu = camera Kinect din NYU Depth V2; fov = fotografie proprie")
        q.add_argument("--hfov", type=float, default=60.0, help="camp vizual orizontal in grade, pentru --camera fov")
        q.add_argument("--focal35", type=float,
                       help="distanta focala echivalenta 35 mm a telefonului (ex. 24); are prioritate fata de --camera")

    p = sub.add_parser("run", help="compune un obiect virtual intr-o imagine, cu hartile de adancime date")
    p.add_argument("--rgb", required=True)
    p.add_argument("--depth", action="append", metavar="NUME=CALE", help="o harta .npy in metri; se poate repeta")
    p.add_argument("--depth-root", help="folder cu <model>/<id>.npy")
    p.add_argument("--depth-dir", help="folder cu <id>_<model>.npy, cum scrie depth.run; hartile ruta_* nu voteaza")
    p.add_argument("--extra", action="append", metavar="NUME=CALE",
                   help="o harta care primeste masca proprie, dar nu voteaza (ex. ruta_C)")
    p.add_argument("--id", help="identificatorul imaginii, pentru --depth-root si --depth-dir")
    p.add_argument("--gt", help="adancimea reala (.npy in metri), optional")
    p.add_argument("--priors", help="JSON cu ponderile si erorile modelelor (de la P1)")
    p.add_argument("--sprite", default="sphere")
    p.add_argument("--x", type=float)
    p.add_argument("--y", type=float)
    p.add_argument("--z", type=float, required=True, help="adancimea obiectului, in metri")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=0.0)
    p.add_argument("--out", default="outputs/run")
    p.add_argument("--stem")
    camera_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("preview", help="deseneaza obiectul peste poza fara ocluzie, ca sa alegi plasarea")
    p.add_argument("--rgb", required=True)
    p.add_argument("--sprite", default="sphere")
    p.add_argument("--x", type=float)
    p.add_argument("--y", type=float)
    p.add_argument("--z", type=float, required=True, help="adancimea obiectului, in metri")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--out", default="outputs/previzualizare.png")
    camera_args(p)
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("batch", help="ruleaza toate plasarile dintr-un fisier JSON, pe mai multe imagini")
    p.add_argument("--placements", required=True, help="JSON: {id imagine: [{nume, sprite, x, y, z, scale}]}")
    p.add_argument("--images", required=True, help="folderul cu imagini <id>.jpg / .png")
    p.add_argument("--depth-dir", required=True, help="folder cu <id>_<model>.npy, cum scrie depth.run")
    p.add_argument("--priors", help="JSON cu ponderile si erorile modelelor (de la P1)")
    p.add_argument("--measurements", help="fisierul cu puncte masurate; hartile se calibreaza pe ele, per imagine")
    p.add_argument("--out", default="outputs/batch")
    camera_args(p)
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("check", help="compara hartile de adancime cu punctele masurate cu ruleta pe o poza proprie")
    p.add_argument("--measurements", default="data/photos/masuratori.json")
    p.add_argument("--photo", required=True, help="numele pozei din fisierul de masuratori")
    p.add_argument("--depth", action="append", metavar="NUME=CALE", help="o harta .npy in metri; se poate repeta")
    p.add_argument("--depth-dir", help="folder cu <poza>_<model>.npy, cum scrie depth.run")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("placements", help="plasari aleatoare reproductibile, dintr-o harta de referinta")
    p.add_argument("--depth", required=True)
    p.add_argument("--sprite", default="sphere")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/plasari.json")
    camera_args(p)
    p.set_defaults(func=cmd_placements)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
