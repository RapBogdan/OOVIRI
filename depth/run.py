"""Sprint 1 (P1): cele 3 modele + ruta C pe un folder de imagini -> adâncimi aliniate în metri.

Rulare: python -m depth.run <folder> --out outputs

Lângă <id>.png poate sta <id>_depth.png (adevărul NYU, uint16 în mm), folosit ca referință
pentru aliniere; fără el, DA V2 și MiDaS se aliniază pe ZoeDepth.
Ieșire în --out: <id>_<model>.npy (metri), <id>_<model>.png și <id>_panel.png.
"""
import argparse
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.image import imsave
from PIL import Image

from evaluation.metrics import depth_metrics
from fusion import fuse_depth_median

from .models import MODELS, estimate_depth

CMAP = "turbo"


def images_in(folder) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir()
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and not p.stem.endswith("_depth"))


def load_gt(path: Path) -> np.ndarray | None:
    p = path.with_name(f"{path.stem}_depth.png")
    return np.array(Image.open(p), dtype=np.float32) / 1000 if p.exists() else None


def process(path: Path, out: Path) -> dict[str, float]:
    """Rulează cele 3 modele + ruta C pe o imagine, salvează rezultatele, întoarce AbsRel (dacă există adevăr)."""
    image = np.array(Image.open(path).convert("RGB"))
    gt = load_gt(path)
    zoe = estimate_depth(image, "zoedepth")
    ref = gt if gt is not None else zoe
    depths = {name: zoe if name == "zoedepth" else estimate_depth(image, name, ref) for name in MODELS}
    depths["ruta_C"] = fuse_depth_median(list(depths.values()))
    scores = {name: depth_metrics(d, gt)["AbsRel"] for name, d in depths.items()} if gt is not None else {}

    out.mkdir(parents=True, exist_ok=True)
    vmin, vmax = np.percentile(ref[ref > 0], [1, 99])
    for name, d in depths.items():
        np.save(out / f"{path.stem}_{name}.npy", d)
        imsave(out / f"{path.stem}_{name}.png", d, cmap=CMAP, vmin=vmin, vmax=vmax)

    panels = {"RGB": image, **({"adevăr NYU": gt} if gt is not None else {}), **depths}
    fig = Figure(figsize=(3.2 * len(panels), 2.8), layout="constrained")
    for ax, (title, im) in zip(fig.subplots(1, len(panels)), panels.items()):
        ax.imshow(im, cmap=CMAP, vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\nAbsRel {scores[title]:.3f}" if title in scores else title, fontsize=9)
        ax.axis("off")
    fig.savefig(out / f"{path.stem}_panel.png", dpi=100)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    for path in images_in(args.folder):
        scores = process(path, Path(args.out))
        print(path.name, "  ".join(f"{n}: {s:.3f}" for n, s in scores.items()))


if __name__ == "__main__":
    main()
