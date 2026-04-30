from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .types import ThresholdSpec


def _to_flat_numpy(activations: torch.Tensor) -> np.ndarray:
    return activations.detach().to(torch.float64).reshape(-1).cpu().numpy()


def resolve_thresholds(
    activations_per_fen: Sequence[torch.Tensor],
    threshold: ThresholdSpec,
) -> list[float]:
    """Resolve one scalar threshold per sample.

    Notes
    -----
    ``absolute`` is trivial: the same threshold is reused for every sample.

    ``ratio_to_max`` supports two use cases:
    - ``scope="sample"``: each FEN uses its own ``max_activation``.
    - ``scope="dataset"``: one global ``max_activation`` is computed across all FENs.

    ``percentile`` is handled in the same way: per-sample or global percentile.
    """

    if not activations_per_fen:
        return []

    if threshold.mode == "absolute":
        return [float(threshold.value)] * len(activations_per_fen)

    if threshold.scope == "dataset":
        dataset_values = np.concatenate([_to_flat_numpy(acts) for acts in activations_per_fen])
        if threshold.mode == "ratio_to_max":
            global_max = float(dataset_values.max()) if dataset_values.size else 0.0
            resolved = float(threshold.value) * global_max
        elif threshold.mode == "percentile":
            resolved = float(np.percentile(dataset_values, threshold.value)) if dataset_values.size else 0.0
        else:
            raise ValueError(f"Unsupported threshold mode: {threshold.mode}")
        return [resolved] * len(activations_per_fen)

    per_sample_thresholds: list[float] = []
    for activations in activations_per_fen:
        values = _to_flat_numpy(activations)
        if threshold.mode == "ratio_to_max":
            local_max = float(values.max()) if values.size else 0.0
            per_sample_thresholds.append(float(threshold.value) * local_max)
        elif threshold.mode == "percentile":
            per_sample_thresholds.append(float(np.percentile(values, threshold.value)) if values.size else 0.0)
        else:
            raise ValueError(f"Unsupported threshold mode: {threshold.mode}")
    return per_sample_thresholds
