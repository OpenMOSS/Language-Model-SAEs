"""Utilities for comparing feature activations between two chess FEN positions.

This module is intentionally lightweight: it only relies on
`ReplacementModel.setup_attribution(...)` to obtain the sparse activation matrices
for Lorsa (attention) and TC (MLP transcoder) features, then performs a set-based
diff at a single board position (token position 0..63).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FeatureKey:
    """A stable identifier for a feature at a specific layer.

    - type: "lorsa" (attention) or "tc" (mlp transcoder)
    - layer: transformer block index
    - feature_idx: head index for Lorsa, feature index for TC
    """

    type: str
    layer: int
    feature_idx: int


def _extract_pos_activations_from_sparse(
    activation_matrix: torch.Tensor,
    *,
    position: int,
    layers: set[int],
) -> dict[int, dict[int, float]]:
    """Extract sparse activations at a fixed position.

    Args:
        activation_matrix: Sparse COO tensor with shape `(n_layers, n_pos, n_features)`.
        position: Token position to filter on (0..n_pos-1).
        layers: Only return activations for these layers.

    Returns:
        Mapping: layer -> {feature_idx -> activation_value}.
    """

    if activation_matrix.layout != torch.sparse_coo:
        raise TypeError(
            f"Expected sparse COO activation matrix, got layout={activation_matrix.layout}"
        )

    coalesced = activation_matrix.coalesce()
    idx = coalesced.indices()  # (3, nnz): layer, pos, feat
    vals = coalesced.values()

    if idx.numel() == 0:
        return {layer: {} for layer in layers}

    # Filter by position first; much cheaper than filtering by both for each layer.
    pos_mask = idx[1] == int(position)
    if not bool(pos_mask.any()):
        return {layer: {} for layer in layers}

    layers_sel = idx[0][pos_mask].to(torch.int64).tolist()
    feats_sel = idx[2][pos_mask].to(torch.int64).tolist()
    vals_sel = vals[pos_mask].to(torch.float32).tolist()

    out: dict[int, dict[int, float]] = {layer: {} for layer in layers}
    for l, f, v in zip(layers_sel, feats_sel, vals_sel, strict=False):
        if l in layers:
            # coalesced => no duplicate (l,pos,f), last write is fine
            out[l][f] = float(v)
    return out


def _diff_feature_maps(
    a: dict[int, float],
    b: dict[int, float],
    *,
    top_k_deltas: int,
) -> dict[str, Any]:
    """Diff two feature->value maps for the same (type, layer, position)."""

    keys_a = set(a.keys())
    keys_b = set(b.keys())
    common = keys_a & keys_b
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a

    common_rows = [
        {"feature_idx": k, "a": float(a[k]), "b": float(b[k]), "delta": float(b[k] - a[k])}
        for k in common
    ]
    only_a_rows = [{"feature_idx": k, "a": float(a[k])} for k in only_a]
    only_b_rows = [{"feature_idx": k, "b": float(b[k])} for k in only_b]

    # Rank by absolute delta for common features.
    common_rows_sorted = sorted(common_rows, key=lambda r: abs(r["delta"]), reverse=True)
    top_deltas = common_rows_sorted[: max(0, int(top_k_deltas))]

    return {
        "n_common": len(common_rows),
        "n_only_a": len(only_a_rows),
        "n_only_b": len(only_b_rows),
        "top_deltas": top_deltas,
        "only_a": sorted(only_a_rows, key=lambda r: abs(r["a"]), reverse=True),
        "only_b": sorted(only_b_rows, key=lambda r: abs(r["b"]), reverse=True),
    }


def compare_fens_position_activations(
    *,
    model: Any,
    fen_a: str,
    fen_b: str,
    position: int,
    layers: list[int] | None = None,
    include_lorsa: bool = True,
    include_tc: bool = True,
    top_k_deltas: int = 50,
) -> dict[str, Any]:
    """Compare sparse feature activations at the same token position for two FENs.

    This is designed for chess models where the prompt is the FEN string and the
    sequence length is 64 (8x8 board tokens). We compute activations via
    `ReplacementModel.setup_attribution(fen, sparse=True)` for each FEN, then
    diff the active features at `position` for each layer.

    Args:
        model: A `ReplacementModel` (or compatible) instance.
        fen_a: First FEN.
        fen_b: Second FEN.
        position: Token position to compare (0..63).
        layers: Which layers to compare. Defaults to `range(model.cfg.n_layers)`.
        include_lorsa: Whether to compare Lorsa (attention) activations.
        include_tc: Whether to compare TC (MLP transcoder) activations.
        top_k_deltas: For common features, return only the top-K by |delta|.

    Returns:
        JSON-serializable dict containing per-layer diffs for Lorsa and/or TC.
    """

    if position < 0:
        raise ValueError(f"position must be >=0, got {position}")

    n_layers = int(getattr(getattr(model, "cfg", None), "n_layers", 15))
    layer_list = layers if layers is not None else list(range(n_layers))
    layer_set = set(int(x) for x in layer_list)

    # Compute sparse activations for both FENs.
    with torch.no_grad():
        _, lorsa_a, _, tc_a, _, _ = model.setup_attribution(fen_a, sparse=True)
        _, lorsa_b, _, tc_b, _, _ = model.setup_attribution(fen_b, sparse=True)

    result_layers: list[dict[str, Any]] = []

    lorsa_pos_a: dict[int, dict[int, float]] = {}
    lorsa_pos_b: dict[int, dict[int, float]] = {}
    tc_pos_a: dict[int, dict[int, float]] = {}
    tc_pos_b: dict[int, dict[int, float]] = {}

    if include_lorsa:
        lorsa_pos_a = _extract_pos_activations_from_sparse(
            lorsa_a, position=position, layers=layer_set
        )
        lorsa_pos_b = _extract_pos_activations_from_sparse(
            lorsa_b, position=position, layers=layer_set
        )
    if include_tc:
        tc_pos_a = _extract_pos_activations_from_sparse(
            tc_a, position=position, layers=layer_set
        )
        tc_pos_b = _extract_pos_activations_from_sparse(
            tc_b, position=position, layers=layer_set
        )

    for layer in layer_list:
        row: dict[str, Any] = {"layer": int(layer)}
        if include_lorsa:
            row["lorsa"] = _diff_feature_maps(
                lorsa_pos_a.get(layer, {}),
                lorsa_pos_b.get(layer, {}),
                top_k_deltas=top_k_deltas,
            )
        if include_tc:
            row["tc"] = _diff_feature_maps(
                tc_pos_a.get(layer, {}),
                tc_pos_b.get(layer, {}),
                top_k_deltas=top_k_deltas,
            )
        result_layers.append(row)

    return {
        "fen_a": fen_a,
        "fen_b": fen_b,
        "position": int(position),
        "layers": result_layers,
        "meta": {
            "n_layers": n_layers,
            "include_lorsa": bool(include_lorsa),
            "include_tc": bool(include_tc),
            "top_k_deltas": int(top_k_deltas),
        },
    }






