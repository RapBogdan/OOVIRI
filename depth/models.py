"""Estimarea adâncimii cu cele 3 modele (Persoana 1)."""
from functools import cache
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from .align import align_scale

# Identificatori Hugging Face pentru cele 3 modele care votează.
# ZoeDepth dă metri; DA V2 și MiDaS dau disparitate relativă (se aliniază cu align_scale).
MODELS = {
    "depth_anything_v2": "depth-anything/Depth-Anything-V2-Small-hf",
    "midas": "Intel/dpt-large",
    "zoedepth": "Intel/zoedepth-nyu-kitti",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"  # se descarcă aici la prima rulare (ignorat de Git)


@cache
def _load(model_name: str):
    hf_id = MODELS[model_name]
    processor = AutoImageProcessor.from_pretrained(hf_id, cache_dir=MODELS_DIR)
    model = AutoModelForDepthEstimation.from_pretrained(hf_id, cache_dir=MODELS_DIR).to(DEVICE).eval()
    return processor, model


@torch.inference_mode()
def predict_raw(image: np.ndarray, model_name: str) -> np.ndarray:
    """Ieșirea brută a modelului, la rezoluția imaginii (H×W, float32)."""
    processor, model = _load(model_name)
    size = [image.shape[:2]]
    # ZoeDepth: fără padding-ul implicit din HF, care micșorează scena și supraestimează adâncimea cu ~25% pe NYU.
    zoe = model_name == "zoedepth"
    inputs = processor(images=image, return_tensors="pt", **({"do_pad": False} if zoe else {}))
    outputs = model(**inputs.to(DEVICE))
    extra = {"do_remove_padding": False} if zoe else {}
    depth = processor.post_process_depth_estimation(outputs, target_sizes=size, **extra)[0]["predicted_depth"]
    return depth.float().cpu().numpy()


def estimate_depth(image: np.ndarray, model_name: str, reference: np.ndarray | None = None) -> np.ndarray:
    """Interfața comună: imagine RGB (H×W×3, uint8) -> adâncime (H×W, float32, metri).

    ZoeDepth e metric. DA V2 și MiDaS se aliniază pe `reference` (adâncimea reală NYU),
    iar fără ea pe ieșirea ZoeDepth (fotografii proprii).
    """
    raw = predict_raw(image, model_name)
    if model_name == "zoedepth":
        return raw
    if reference is None:
        reference = predict_raw(image, "zoedepth")
    return align_scale(raw, reference)
