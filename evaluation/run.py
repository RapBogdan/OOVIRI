"""Evaluarea adâncimii pe NYU Depth V2 test: fiecare model + ruta C, în două moduri de aliniere.

Rulare: python -m evaluation.run [--n 654] [--out outputs/eval_depth.csv]

Aliniere "gt": DA V2 și MiDaS aliniate pe adevărul NYU (protocolul din articole, comparabil cu SOTA).
Aliniere "zoedepth": aliniate pe ZoeDepth, fără adevăr (cazul realist, ca pe fotografii proprii).
ZoeDepth nu se aliniază niciodată. Timpul: secunde/imagine; ruta C costă cât cele 3 modele adunate.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

from depth.align import align_scale
from depth.models import MODELS, predict_raw
from depth.run import images_in, load_gt
from fusion import fuse_depth_median

from .metrics import depth_metrics

DATA = Path(__file__).resolve().parents[1] / "data" / "nyu_test"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=654, help="câte imagini de test (implicit toate)")
    parser.add_argument("--out", default="outputs/eval_depth.csv")
    args = parser.parse_args()

    paths = images_in(DATA)[: args.n]
    first = np.array(Image.open(paths[0]).convert("RGB"))
    for m in MODELS:  # încălzire: încărcarea modelelor nu intră în timp
        predict_raw(first, m)

    scores, seconds = {}, {m: 0.0 for m in MODELS}
    for path in paths:
        image, gt = np.array(Image.open(path).convert("RGB")), load_gt(path)
        raw = {}
        for m in MODELS:
            t = time.perf_counter()
            raw[m] = predict_raw(image, m)
            seconds[m] += time.perf_counter() - t
        for align, ref in (("gt", gt), ("zoedepth", raw["zoedepth"])):
            depths = {m: raw[m] if m == "zoedepth" else align_scale(raw[m], ref) for m in MODELS}
            depths["ruta_C"] = fuse_depth_median(list(depths.values()))
            for name, d in depths.items():
                scores.setdefault((name, align), []).append(depth_metrics(d, gt))

    seconds["ruta_C"] = sum(seconds.values())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metoda", "aliniere", "AbsRel", "RMSE", "delta1", "s_img"])
        print(f"{len(paths)} imagini NYU test, crop Eigen\n{'metoda':<20}{'aliniere':<10}{'AbsRel':>8}{'RMSE':>8}{'delta1':>8}{'s/img':>8}")
        for (name, align), s in scores.items():
            mean = {k: np.mean([x[k] for x in s]) for k in s[0]}
            t = seconds[name] / len(paths)
            w.writerow([name, align, f"{mean['AbsRel']:.4f}", f"{mean['RMSE']:.4f}", f"{mean['delta1']:.4f}", f"{t:.4f}"])
            print(f"{name:<20}{align:<10}{mean['AbsRel']:>8.3f}{mean['RMSE']:>8.3f}{mean['delta1']:>8.3f}{t:>8.3f}")
    print(f"tabel salvat în {out}")


if __name__ == "__main__":
    main()
