"""Distance, overlap, and experiment metrics for MHA-PM outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(sorted_weights)
    threshold = quantile * float(sorted_weights.sum())
    return float(sorted_values[np.searchsorted(cumulative, threshold, side="left")])


def assignment_metrics(assignments: pd.DataFrame) -> dict[str, float]:
    weights = assignments["weight"].to_numpy(dtype=float)
    distances = assignments["assigned_distance"].to_numpy(dtype=float)
    return {
        "objective": float(np.dot(weights, distances)),
        "weighted_mean_distance": float(np.dot(weights, distances)),
        "weighted_p90_distance": weighted_quantile(distances, weights, 0.90),
        "max_distance": float(np.max(distances)),
    }


def shelter_set_metrics(first: set[str], second: set[str]) -> dict[str, float | int]:
    intersection = len(first & second)
    union = len(first | second)
    jaccard = intersection / union if union else 1.0
    return {
        "overlap_count": intersection,
        "jaccard_similarity": jaccard,
        "temporal_configuration_change": 1.0 - jaccard,
    }
