"""Descarcă NYU Depth V2 test (654 imagini) în data/nyu_test/<id>.png + <id>_depth.png (uint16, mm).

Sursa: split-ul "validation" din https://huggingface.co/datasets/sayakpaul/nyu_depth_v2
(= cele 654 imagini de test NYU), 2 fișiere parquet, ~1 GB în data/raw/.
Rulare (din rădăcina proiectului): python scripts/download_nyu.py
"""
import io
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

URL = "https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/refs%2Fconvert%2Fparquet/default/partial-validation/{:04d}.parquet"
DATA = Path(__file__).resolve().parents[1] / "data"
RAW, OUT = DATA / "raw", DATA / "nyu_test"

RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
for k in range(2):
    f = RAW / f"val_{k:04d}.parquet"
    if not f.exists():
        print(f"descarc {f.name} ...")
        urllib.request.urlretrieve(URL.format(k), f.with_suffix(".part"))
        f.with_suffix(".part").rename(f)

if (OUT / "0653_depth.png").exists():
    print(f"setul există deja în {OUT}")
else:
    i = 0
    for f in sorted(RAW.glob("val_*.parquet")):
        for row in pq.read_table(f).to_pylist():
            Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB").save(OUT / f"{i:04d}.png")
            depth = np.array(Image.open(io.BytesIO(row["depth_map"]["bytes"])), dtype=np.float32)  # metri
            Image.fromarray(np.round(depth * 1000).clip(0, 65535).astype(np.uint16)).save(OUT / f"{i:04d}_depth.png")
            i += 1
    print(f"{i} imagini în {OUT}")
