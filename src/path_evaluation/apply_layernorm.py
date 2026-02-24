from __future__ import annotations

from typing import Literal, Optional

import torch


try:  # 仅用于类型标注，避免在运行时强依赖
    from transformer_lens import HookedTransformer
except Exception:  # pragma: no cover - 在没有 transformer_lens 时静默退化
    HookedTransformer = object  # type: ignore[misc,assignment]


_GLOBAL_MODEL: Optional[HookedTransformer] = None

FeatureType = Literal["transcoder", "lorsa"]


def set_layernorm_model(model: HookedTransformer) -> None:
    """设置用于路径 LayerNorm 传播的全局模型。

    该模型应当是 Leela Chess (Lc0) 风格的 `HookedTransformer`，
    并且每一层块包含 `ln1`, `ln2`, `alpha_input`, `alpha_out1` 等属性。
    后续路径传播函数会使用这些属性在层之间传播向量。
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
    """使用模型的 LayerNorm，将向量从 source feature 传播到 target feature。

    该函数考虑了 transcoder 和 Lorsa 在模型中的不同位置：
    - Lorsa 特征位于对应层 ``ln1`` 之前（即该层 attention 的输入残差处）；
    - Transcoder 特征位于对应层 ``ln2`` 之前（即该层 MLP 的输入残差处）。

    我们只沿着各层的 ``ln1`` / ``ln2`` 轨迹传播向量，不显式建模注意力或 MLP 的非线性部分，
    也不使用 ``alpha_input`` / ``alpha_out1`` 等缩放系数。

    参数
    ----
    vec:
        需要在残差流中沿层传播的向量，形状应与模型残差维度一致。
    src_layer:
        起始层索引，应满足 ``0 <= src_layer < num_layers``。
    src_feature_type:
        起始 feature 类型，'transcoder' 或 'lorsa'。
    tgt_layer:
        目标层索引，应满足 ``src_layer <= tgt_layer <= num_layers``。
    tgt_feature_type:
        目标 feature 类型，'transcoder' 或 'lorsa'。

    返回
    ----
    torch.Tensor
        经过正确的 LayerNorm 路径传播后的向量。

    说明
    ----
    - 本函数假设全局模型已通过 `set_layernorm_model` 设置；
    - 仅支持向前传播（即 ``src_layer <= tgt_layer``）。
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

    # 将位置抽象为「在第几层的第几个 LayerNorm 之前」
    # 对于第 L 层：
    #   - Lorsa 位置：在 ln1 之前，记为时间点 2L
    #   - Transcoder 位置：在 ln2 之前，记为时间点 2L+1
    # 跨层传播就是在这些时间点之间依次应用 ln1 / ln2。
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

    # 依次跨过中间的 LayerNorm 边界
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
