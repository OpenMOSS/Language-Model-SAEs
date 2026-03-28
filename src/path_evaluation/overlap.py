"""Pairwise Jaccard overlap between top-feature CSVs in one folder."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

FeatureKey = tuple[int, int, int, str]
"""One row in CSV: (position_idx, layer, feature_id, feature_type)."""

WithinLayerKey = tuple[int, int, str]
"""Feature identity inside a fixed layer: (position_idx, feature_id, feature_type)."""


def _get_feature_set(csv_path: Path) -> set[FeatureKey]:
    """Read a feature CSV and convert rows into a set of comparable feature keys."""
    df = pd.read_csv(csv_path)
    features: set[FeatureKey] = set()
    for _, row in df.iterrows():
        features.add(
            (
                int(row["position_idx"]),
                int(row["layer"]),
                int(row["feature_id"]),
                str(row["feature_type"]),
            )
        )
    return features


def _subset_for_layer(feature_set: set[FeatureKey], layer: int) -> set[WithinLayerKey]:
    """Project features to within-layer keys (position_idx, feature_id, feature_type)."""
    return {(pos, fid, ftype) for pos, lyr, fid, ftype in feature_set if lyr == layer}


def _compute_pair_overlap(a: set[FeatureKey], b: set[FeatureKey]) -> float:
    """Compute Jaccard overlap between two feature sets."""
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def _compute_pair_overlap_for_layer(
    a: set[FeatureKey], b: set[FeatureKey], layer: int
) -> float:
    """Jaccard overlap restricted to rows with the given ``layer`` index."""
    sa = _subset_for_layer(a, layer)
    sb = _subset_for_layer(b, layer)
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return inter / union


def _load_folder_feature_sets(folder_path: str) -> dict[str, set[FeatureKey]] | None:
    """Load all ``*_top*_features.csv`` in a folder into name -> feature set. At least 2 CSVs."""
    folder = Path(folder_path)
    csv_files: list[Path] = sorted(
        folder.glob("*_top*_features.csv"), key=lambda p: p.name
    )
    if len(csv_files) < 2:
        return None
    return {f.name: _get_feature_set(f) for f in csv_files}


def _all_layers_in_sets(feature_sets: dict[str, set[FeatureKey]]) -> set[int]:
    layers: set[int] = set()
    for feats in feature_sets.values():
        for _, lyr, _, _ in feats:
            layers.add(lyr)
    return layers


def compute_folder_overlap(folder_path: str) -> float:
    """Return the mean pairwise overlap among all '*_top*_features.csv' in a folder."""
    _, mean_overlap = compute_folder_overlap_details(folder_path)
    return mean_overlap


def compute_folder_overlap_details(
    folder_path: str,
) -> tuple[list[tuple[str, str, float]], float]:
    """Return all pairwise overlaps and their mean for CSVs in a folder.

    A feature is considered identical iff (position_idx, layer, feature_id, feature_type) are equal.
    """
    feature_sets = _load_folder_feature_sets(folder_path)
    if feature_sets is None:
        return [], float("nan")

    pair_overlaps: list[tuple[str, str, float]] = []
    for (name_a, set_a), (name_b, set_b) in combinations(feature_sets.items(), 2):
        pair_overlaps.append((name_a, name_b, _compute_pair_overlap(set_a, set_b)))

    mean_overlap = (
        float(sum(v for _, _, v in pair_overlaps) / len(pair_overlaps))
        if pair_overlaps
        else float("nan")
    )
    return pair_overlaps, mean_overlap


def compute_folder_overlap_per_layer_details(
    folder_path: str,
) -> tuple[
    dict[int, list[tuple[str, str, float]]],
    dict[int, float],
]:
    """Per-layer pairwise overlaps and per-layer mean (same definition as :func:`compute_folder_overlap_per_layer`)."""
    feature_sets = _load_folder_feature_sets(folder_path)
    if feature_sets is None:
        return {}, {}

    layers = sorted(_all_layers_in_sets(feature_sets))
    by_layer: dict[int, list[tuple[str, str, float]]] = {lyr: [] for lyr in layers}
    items = list(feature_sets.items())

    for layer in layers:
        for (name_a, set_a), (name_b, set_b) in combinations(items, 2):
            o = _compute_pair_overlap_for_layer(set_a, set_b, layer)
            by_layer[layer].append((name_a, name_b, o))

    means: dict[int, float] = {
        lyr: float(sum(t[2] for t in pairs) / len(pairs))
        for lyr, pairs in by_layer.items()
        if pairs
    }
    return by_layer, means


def compute_folder_overlap_per_layer(folder_path: str) -> dict[int, float]:
    """Mean pairwise Jaccard overlap **within each layer** (see :func:`compute_folder_overlap_per_layer_details`)."""
    _, means = compute_folder_overlap_per_layer_details(folder_path)
    return means
