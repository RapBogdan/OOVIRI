"""Voting routes that combine the occlusion decisions of several depth models.

Route A      majority vote on the binary masks (baseline).
Route B      weighted vote on the binary masks, weight = model accuracy (as written in the project plan).
Route B-soft weighted vote on confidences derived from the depth gap (our refinement, see the note below).

Note on route B. With exactly three voters and weights that are accuracies (all in
0.5..1), any two models always outweigh the third, so route B returns the same mask
as route A. `same_as_majority` checks this for a given set of weights. Route B-soft
lets a model that is very sure outvote two that are barely on the other side, which
is the behaviour "weighted" was meant to buy.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _stack(masks: Sequence[np.ndarray]) -> np.ndarray:
    if len(masks) == 0:
        raise ValueError("at least one mask is needed")
    stack = np.stack([np.asarray(m, dtype=bool) for m in masks])
    return stack


def vote_count(masks: Sequence[np.ndarray]) -> np.ndarray:
    """How many models say "occluded" at each pixel."""
    return _stack(masks).sum(axis=0).astype(np.uint8)


def majority_vote(masks: Sequence[np.ndarray]) -> np.ndarray:
    """Route A: occluded where more than half of the models say so (2 of 3)."""
    stack = _stack(masks)
    return stack.sum(axis=0) * 2 > stack.shape[0]


def weighted_vote(masks: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Route B: occluded where the models voting "occluded" hold more than half of the total weight."""
    stack = _stack(masks)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (stack.shape[0],) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("one non-negative weight per mask is needed, not all zero")
    score = np.tensordot(w, stack.astype(np.float64), axes=1)
    return score > 0.5 * w.sum()


def same_as_majority(weights: Sequence[float]) -> bool:
    """True when these weights can never change the outcome of a simple majority vote."""
    w = np.asarray(weights, dtype=np.float64)
    n = len(w)
    for bits in range(1 << n):
        chosen = np.array([(bits >> i) & 1 for i in range(n)], dtype=bool)
        if (w[chosen].sum() > 0.5 * w.sum()) != (chosen.sum() * 2 > n):
            return False
    return True


def soft_weighted_vote(margins: Sequence[np.ndarray], weights: Sequence[float],
                       taus: Sequence[float | np.ndarray]) -> np.ndarray:
    """Route B-soft: weighted vote on how sure each model is.

    margins  per model, the signed depth gap from `occlusion.margin` (positive = occluded).
    weights  per model, e.g. its accuracy on the validation subset.
    taus     per model, the depth error it typically makes (metres; a number or a per-pixel
             array). A gap much larger than tau is a confident vote, a gap around zero is
             close to "don't know" (0.5).
    """
    if not (len(margins) == len(weights) == len(taus)) or len(margins) == 0:
        raise ValueError("margins, weights and taus must have the same non-zero length")
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative and not all zero")
    score = np.zeros(np.asarray(margins[0]).shape, np.float64)
    for m, wi, tau in zip(margins, w, taus):
        t = np.maximum(np.asarray(tau, dtype=np.float64), 1e-6)
        x = np.clip(np.asarray(m, dtype=np.float64) / t, -60.0, 60.0)
        score += wi / (1.0 + np.exp(-x))
    return score > 0.5 * w.sum()
