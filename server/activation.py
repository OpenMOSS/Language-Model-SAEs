from lm_saes import SparseAutoEncoder, LowRankSparseAttention
from transformer_lens import HookedTransformer
from typing import Dict, List, Tuple, Optional
import torch


def get_activated_features_at_position(
    model: HookedTransformer,
    transcoders: Dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    fen: str,
    layer: int,
    pos: int,
    component_type: str  # "attn" or "mlp"
) -> Dict[str, List[Dict[str, float]]]:
    """
    获取指定层和位置激活的所有 features
    
    Args:
        model: HookedTransformer 模型
        transcoders: 字典，层号 -> Transcoder SAE
        lorsas: 列表，Lorsa SAE（按层索引）
        fen: FEN 字符串
        layer: 层号（0-14）
        pos: 位置索引（0-63）
        component_type: 组件类型，"attn" 或 "mlp"
    
    Returns:
        字典，包含：
        - "attn_features": 如果是 attn，返回激活的 Lorsa features
        - "mlp_features": 如果是 mlp，返回激活的 Transcoder features
        每个 feature 包含：
        - "feature_index": feature 索引
        - "activation_value": 激活值
    """
    if layer < 0 or layer >= 15:
        raise ValueError(f"层号必须在 0-14 之间，当前值: {layer}")
    
    if pos < 0 or pos >= 64:
        raise ValueError(f"位置索引必须在 0-63 之间，当前值: {pos}")
    
    if component_type not in ["attn", "mlp"]:
        raise ValueError(f"component_type 必须是 'attn' 或 'mlp'，当前值: {component_type}")
    
    # 运行模型获取激活值
    with torch.no_grad():
        _, cache = model.run_with_cache(fen, prepend_bos=False)
    
    result = {}
    
    # 获取 Attention (Lorsa) 的激活 features
    if component_type == "attn":
        if layer >= len(lorsas):
            raise ValueError(f"层 {layer} 超出 Lorsa 范围（共 {len(lorsas)} 层）")
        
        # 获取 attention 的输入激活值
        hook_name = f"blocks.{layer}.hook_attn_in"
        if hook_name not in cache:
            available_hooks = [k for k in cache.keys() if f"blocks.{layer}" in str(k)]
            raise ValueError(
                f"无法找到 hook 点 {hook_name}。"
                f"可用的 hook 点: {available_hooks[:10]}"
            )
        
        attn_input = cache[hook_name]  # [batch, seq_len, d_model]
        
        # 确保有 batch 维度
        if attn_input.dim() == 2:
            attn_input = attn_input.unsqueeze(0)  # [1, seq_len, d_model]
        
        # 使用 Lorsa 编码
        lorsa = lorsas[layer]
        feature_acts = lorsa.encode(attn_input)  # [batch, seq_len, d_sae]
        
        # 获取指定位置的激活值
        if feature_acts.dim() == 3:
            pos_activations = feature_acts[0, pos, :]  # [d_sae]
        else:
            pos_activations = feature_acts[pos, :]  # [d_sae]
        
        # 找出非零激活的 features
        nonzero_mask = pos_activations != 0
        nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False).squeeze(-1)
        nonzero_values = pos_activations[nonzero_mask]
        
        # 构建结果列表
        attn_features = []
        for idx, val in zip(nonzero_indices.cpu().numpy(), nonzero_values.cpu().numpy()):
            attn_features.append({
                "feature_index": int(idx),
                "activation_value": float(val)
            })
        
        # 按激活值绝对值排序（从大到小）
        attn_features.sort(key=lambda x: abs(x["activation_value"]), reverse=True)
        
        result["attn_features"] = attn_features
    
    # 获取 MLP (Transcoder) 的激活 features
    if component_type == "mlp":
        if layer not in transcoders:
            raise ValueError(f"层 {layer} 不在 transcoders 中")
        
        # 获取 MLP 的输入激活值（resid_mid_after_ln）
        hook_name = f"blocks.{layer}.resid_mid_after_ln"
        if hook_name not in cache:
            # 尝试备用 hook 点
            alt_hook_name = f"blocks.{layer}.ln2.hook_normalized"
            if alt_hook_name in cache:
                hook_name = alt_hook_name
            else:
                available_hooks = [k for k in cache.keys() if f"blocks.{layer}" in str(k)]
                raise ValueError(
                    f"无法找到 hook 点 {hook_name} 或 {alt_hook_name}。"
                    f"可用的 hook 点: {available_hooks[:10]}"
                )
        
        mlp_input = cache[hook_name]  # [batch, seq_len, d_model]
        
        # 确保有 batch 维度
        if mlp_input.dim() == 2:
            mlp_input = mlp_input.unsqueeze(0)  # [1, seq_len, d_model]
        
        # 使用 Transcoder 编码
        transcoder = transcoders[layer]
        feature_acts = transcoder.encode(mlp_input)  # [batch, seq_len, d_sae]
        
        # 获取指定位置的激活值
        if feature_acts.dim() == 3:
            pos_activations = feature_acts[0, pos, :]  # [d_sae]
        else:
            pos_activations = feature_acts[pos, :]  # [d_sae]
        
        # 找出非零激活的 features
        nonzero_mask = pos_activations != 0
        nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False).squeeze(-1)
        nonzero_values = pos_activations[nonzero_mask]
        
        # 构建结果列表
        mlp_features = []
        for idx, val in zip(nonzero_indices.cpu().numpy(), nonzero_values.cpu().numpy()):
            mlp_features.append({
                "feature_index": int(idx),
                "activation_value": float(val)
            })
        
        # 按激活值绝对值排序（从大到小）
        mlp_features.sort(key=lambda x: abs(x["activation_value"]), reverse=True)
        
        result["mlp_features"] = mlp_features
    
    return result
