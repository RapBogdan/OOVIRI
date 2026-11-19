"""Rutele de vot / fuziune. Ruta C (Persoana 1); rutele A și B (Persoana 2) se adaugă aici."""
import numpy as np


def fuse_depth_median(depths: list[np.ndarray]) -> np.ndarray:
    """Ruta C: mediana per pixel a hărților de adâncime aliniate (metri) -> o singură hartă."""
    return np.median(np.stack(depths), axis=0)
