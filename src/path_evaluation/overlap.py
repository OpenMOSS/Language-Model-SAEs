import pandas as pd
from pathlib import Path
from itertools import combinations
from typing import Iterable, Tuple, Set, Dict, List


FeatureKey = Tuple[int, int, int, str]


def _get_feature_set(csv_path: Path) -> Set[FeatureKey]:
    """Read a feature CSV and convert rows into a set of comparable feature keys."""
    df = pd.read_csv(csv_path)
    features: Set[FeatureKey] = set()
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


def _compute_pair_overlap(a: Set[FeatureKey], b: Set[FeatureKey]) -> float:
    """Compute Jaccard overlap between two feature sets."""
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def compute_folder_overlap(folder_path: str) -> float:
    """Return the mean pairwise overlap among all '*_top*_features.csv' in a folder."""
    _, mean_overlap = compute_folder_overlap_details(folder_path)
    return mean_overlap


def compute_folder_overlap_details(folder_path: str) -> tuple[list[tuple[str, str, float]], float]:
    """Return all pairwise overlaps and their mean for CSVs in a folder.

    A feature is considered identical iff (position_idx, layer, feature_id, feature_type) are equal.
    """
    folder = Path(folder_path)
    csv_files: Iterable[Path] = folder.glob("*_top*_features.csv")
    csv_files = sorted(list(csv_files), key=lambda p: p.name)
    if len(csv_files) < 2:
        return [], float("nan")

    feature_sets: Dict[str, Set[FeatureKey]] = {f.name: _get_feature_set(f) for f in csv_files}

    pair_overlaps: List[Tuple[str, str, float]] = []
    for (name_a, set_a), (name_b, set_b) in combinations(feature_sets.items(), 2):
        pair_overlaps.append((name_a, name_b, _compute_pair_overlap(set_a, set_b)))

    mean_overlap = float(sum(v for _, _, v in pair_overlaps) / len(pair_overlaps)) if pair_overlaps else float("nan")
    return pair_overlaps, mean_overlap

import pandas as pd