"""Depth maps for a folder of own photos, using P1's code unchanged (depth.models, depth.align, fusion).

Same steps as depth.run.process() for images without ground truth, minus the matplotlib
panel (matplotlib's Agg DLL is blocked by Windows application control on this PC).
Usage: python harti_poze.py <p1 worktree> <photo folder> <output folder>
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, sys.argv[1])
from depth.models import DEVICE, MODELS, estimate_depth  # noqa: E402
from fusion import fuse_depth_median  # noqa: E402

src, out = Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
print("device:", DEVICE)
for path in sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
    image = np.array(Image.open(path).convert("RGB"))
    t = time.perf_counter()
    zoe = estimate_depth(image, "zoedepth")
    # no ground truth for own photos: the relative models are aligned to ZoeDepth, as in depth.run
    depths = {name: zoe if name == "zoedepth" else estimate_depth(image, name, zoe) for name in MODELS}
    depths["ruta_C"] = fuse_depth_median(list(depths.values()))
    for name, d in depths.items():
        np.save(out / f"{path.stem}_{name}.npy", d.astype(np.float32))
    span = "  ".join(f"{n} {float(d.min()):.2f}-{float(d.max()):.2f} m" for n, d in depths.items())
    print(f"{path.name} {image.shape[1]}x{image.shape[0]}  {time.perf_counter() - t:.1f}s  {span}")
