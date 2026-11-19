"""Rutele de vot / fuziune. Ruta C (Persoana 1); rutele A, B și B-soft (Persoana 2) vin din ooviri.voting."""
import numpy as np

from ooviri.voting import majority_vote, soft_weighted_vote, weighted_vote  # rutele A, B și B-soft


def fuse_depth_median(depths: list[np.ndarray]) -> np.ndarray:
    """Ruta C: mediana per pixel a hărților de adâncime aliniate (metri) -> o singură hartă."""
    return np.median(np.stack(depths), axis=0)


__all__ = ["fuse_depth_median", "majority_vote", "weighted_vote", "soft_weighted_vote"]
