"""Hărțile de adâncime pentru un folder de fotografii proprii, cu modelele din depth/.

Aceiași pași ca depth.run.process() pentru imagini fără adevăr, fără panoul desenat cu
matplotlib (pe unul dintre calculatoarele noastre, DLL-ul Agg din matplotlib e blocat de
o politică Windows de control al aplicațiilor).

Rulare, din rădăcina proiectului:
    python scripts/harti_poze.py data/photos data/depth
Scrie <ieșire>/<poză>_<model>.npy (float32, metri) pentru cele 3 modele și ruta C.
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depth.models import DEVICE, MODELS, estimate_depth  # noqa: E402
from fusion import fuse_depth_median  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    print("device:", DEVICE)
    for path in sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
        image = np.array(Image.open(path).convert("RGB"))
        t = time.perf_counter()
        zoe = estimate_depth(image, "zoedepth")
        # fără adevăr pe fotografii proprii: modelele relative se aliniază pe ZoeDepth, ca în depth.run
        depths = {name: zoe if name == "zoedepth" else estimate_depth(image, name, zoe) for name in MODELS}
        depths["ruta_C"] = fuse_depth_median(list(depths.values()))
        for name, d in depths.items():
            np.save(out / f"{path.stem}_{name}.npy", d.astype(np.float32))
        span = "  ".join(f"{n} {float(d.min()):.2f}-{float(d.max()):.2f} m" for n, d in depths.items())
        print(f"{path.name} {image.shape[1]}x{image.shape[0]}  {time.perf_counter() - t:.1f}s  {span}")


if __name__ == "__main__":
    main()
