from __future__ import annotations

from typing import Literal, Optional

import torch


try:
    from transformer_lens import HookedTransformer
except Exception:
    HookedTransformer = object  # type: ignore[misc,assignment]


_GLOBAL_MODEL: Optional[HookedTransformer] = None

FeatureType = Literal["transcoder", "lorsa"]


def set_layernorm_model(model: HookedTransformer) -> None:
    """Set global model for LayerNorm path propagation.

    The model should be a `HookedTransformer` in the style of Leela Chess (Lc0),
    and each block should contain `ln1`, `ln2`, `alpha_input`, `alpha_out1` attributes.
    The subsequent path propagation functions will use these attributes to propagate vectors between layers.
    """
    global _GLOBAL_MODEL
    _GLOBAL_MODEL = model


def apply_layernorm_path_with_feature_types(
    vec: torch.Tensor,
    src_layer: int,
    src_feature_type: FeatureType,
    tgt_layer: int,
    tgt_feature_type: FeatureType,
) -> torch.Tensor:
    """Use the model's LayerNorm to propagate a vector from the source feature to the target feature.

    The model should be a `HookedTransformer` in the style of Leela Chess (Lc0),
    and each block should contain `ln1`, `ln2`, `alpha_input`, `alpha_out1` attributes.
    The subsequent path propagation functions will use these attributes to propagate vectors between layers.
    """
    if _GLOBAL_MODEL is None:
        raise RuntimeError(
            "Global model for apply_layernorm_path_with_feature_types is not set. "
            "Please call set_layernorm_model(model) beforehand."
        )

    blocks = getattr(_GLOBAL_MODEL, "blocks", None)
    if blocks is None:
        raise RuntimeError("Global model does not have attribute 'blocks'.")

    num_layers = len(blocks)
    if not (0 <= src_layer < num_layers):
        raise ValueError(
            f"src_layer={src_layer} out of range for num_layers={num_layers}."
        )
    if not (src_layer <= tgt_layer <= num_layers):
        raise ValueError(
            f"tgt_layer={tgt_layer} must satisfy src_layer <= tgt_layer <= num_layers={num_layers}."
        )

    # abstract the position as "before the nth LayerNorm in the layer"
    # for layer L:
    #   - Lorsa position: before ln1, denoted as time point 2L
    #   - Transcoder position: before ln2, denoted as time point 2L+1
    # cross-layer propagation is applying ln1 / ln2 sequentially between these time points.
    def _pos(layer: int, feature_type: FeatureType) -> int:
        return 2 * layer + (1 if feature_type == "transcoder" else 0)

    src_pos = _pos(src_layer, src_feature_type)
    tgt_pos = _pos(tgt_layer, tgt_feature_type)

    if src_pos > tgt_pos:
        raise ValueError(
            "apply_layernorm_path_with_feature_types only supports forward propagation "
            f"but got src position {src_pos} > tgt position {tgt_pos}."
        )

    x = vec

    for t in range(src_pos, tgt_pos):
        layer_idx = t // 2
        is_ln1 = (t % 2 == 0)
        layer = blocks[layer_idx]
        # print(f'{layer_idx = }, {is_ln1 = }')
        if is_ln1:
            ln = getattr(layer, "ln1", None)
            ln_name = "ln1"
        else:
            ln = getattr(layer, "ln2", None)
            ln_name = "ln2"

        if ln is None:
            raise RuntimeError(
                f"Layer {layer_idx} is missing {ln_name} attribute "
                "required for apply_layernorm_path_with_feature_types."
            )
        # print(f'{ln = }, {ln.w = }, {ln.b = }')
        x = ln(x)

    return x
