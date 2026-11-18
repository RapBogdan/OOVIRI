"""Compositing and the pictures we show: the AR image, mask overlays, labelled grids."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .placement import PlacedSprite

_BAR_COLOR = (27, 31, 75)
_BAR_HEIGHT = 30


def _font(size: int = 16):
    for name in ("arial.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose(rgb: np.ndarray, placed: PlacedSprite, occluded: np.ndarray | None = None) -> np.ndarray:
    """Draw the virtual object over the photo, skipping the pixels marked as occluded.

    Without `occluded` this is the naive AR picture: the object always on top.
    """
    if rgb.shape[:2] != placed.alpha.shape:
        raise ValueError("the image and the placed sprite have different sizes")
    visible = placed.alpha.copy()
    if occluded is not None:
        visible[np.asarray(occluded, bool)] = 0.0
    base = rgb[..., :3].astype(np.float32) / 255.0
    out = base * (1.0 - visible[..., None]) + placed.rgb * visible[..., None]
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(230, 40, 40), strength: float = 0.6) -> np.ndarray:
    """Tint the masked pixels, to see a mask on top of the picture."""
    out = rgb[..., :3].astype(np.float32)
    m = np.asarray(mask, bool)
    out[m] = out[m] * (1.0 - strength) + np.asarray(color, np.float32) * strength
    return (out + 0.5).astype(np.uint8)


def vote_overlay(rgb: np.ndarray, counts: np.ndarray, footprint: np.ndarray, n_models: int) -> np.ndarray:
    """Colour the object's pixels by how many models voted "occluded".

    Green = nobody (clearly visible), red = everybody (clearly hidden), yellow and
    orange = the models disagree, which is where voting changes the result.
    """
    palette = {0: (60, 180, 90), n_models: (215, 40, 40)}
    for k in range(1, n_models):
        f = (k - 1) / max(n_models - 2, 1)
        palette[k] = (int(250 - 5 * f), int(215 - 80 * f), int(60 - 20 * f))
    out = rgb[..., :3].astype(np.float32)
    for k, col in palette.items():
        sel = np.asarray(footprint, bool) & (counts == k)
        out[sel] = out[sel] * 0.25 + np.asarray(col, np.float32) * 0.75
    return (out + 0.5).astype(np.uint8)


def crop(img: np.ndarray, bbox: tuple[int, int, int, int], margin: int = 16) -> np.ndarray:
    """Cut out the object's bounding box plus a margin."""
    x0, y0, x1, y1 = bbox
    h, w = img.shape[:2]
    return img[max(0, y0 - margin):min(h, y1 + margin), max(0, x0 - margin):min(w, x1 + margin)]


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    """Put a caption bar above the picture."""
    h, w = img.shape[:2]
    canvas = Image.new("RGB", (w, h + _BAR_HEIGHT), _BAR_COLOR)
    canvas.paste(Image.fromarray(img[..., :3] if img.ndim == 3 else np.dstack([img] * 3)), (0, _BAR_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    while font.getlength(text) > w - 12 and getattr(font, "size", 10) > 10:
        font = _font(font.size - 1)
    draw.text((8, _BAR_HEIGHT // 2), text, fill=(255, 255, 255), font=font, anchor="lm")
    return np.array(canvas)


def grid(images: Sequence[np.ndarray], labels: Sequence[str], cols: int = 4, pad: int = 6,
         background=(255, 255, 255)) -> np.ndarray:
    """Lay labelled pictures out in rows of `cols`; smaller ones are scaled up to the largest."""
    if len(images) != len(labels) or len(images) == 0:
        raise ValueError("one label per image is needed")
    th = max(im.shape[0] for im in images)
    tw = max(im.shape[1] for im in images)
    tiles = []
    for im, text in zip(images, labels):
        if im.dtype == bool:
            im = im.astype(np.uint8) * 255
        if im.ndim == 2:
            im = np.dstack([im] * 3)
        if im.shape[:2] != (th, tw):
            im = np.array(Image.fromarray(im).resize((tw, th), Image.Resampling.NEAREST))
        tiles.append(add_label(im, text))
    rows = (len(tiles) + cols - 1) // cols
    ch, cw = tiles[0].shape[:2]
    canvas = np.empty((rows * ch + (rows + 1) * pad, cols * cw + (cols + 1) * pad, 3), np.uint8)
    canvas[:] = np.asarray(background, np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = pad + r * (ch + pad), pad + c * (cw + pad)
        canvas[y:y + ch, x:x + cw] = tile
    return canvas
