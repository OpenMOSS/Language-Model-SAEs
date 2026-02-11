"""Global weight calculation functions for feature analysis."""
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import einops

from lm_saes.sae import SparseAutoEncoder
from lm_saes import LowRankSparseAttention


# Global cache for max and mean activations
_max_activations_cache: Dict[str, Dict[str, torch.Tensor]] = {}  # combo_key -> {"tc_max": tensor, "lorsa_max": tensor, "tc_mean": tensor, "lorsa_mean": tensor}


def load_max_activations(
    sae_combo_id: str,
    device: str = "cuda",
    get_bt4_sae_combo=None,
    activation_type: str = "max",  # "max" or "mean"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    加载max或mean activations数据
    
    Args:
        sae_combo_id: SAE组合ID
        device: 设备（cuda或cpu）
        get_bt4_sae_combo: 获取SAE组合配置的函数
        activation_type: 激活类型，"max" 或 "mean"
    
    Returns:
        (tc_activations, lorsa_activations) 元组
    """
    global _max_activations_cache
    
    if get_bt4_sae_combo is None:
        raise ValueError("get_bt4_sae_combo function is required")
    
    if activation_type not in ["max", "mean"]:
        raise ValueError(f"activation_type必须是'max'或'mean'，当前值: {activation_type}")
    
    combo_cfg = get_bt4_sae_combo(sae_combo_id)
    normalized_combo_id = combo_cfg["id"]
    cache_key = f"BT4_{normalized_combo_id}"
    
    # 检查缓存
    cache_key_tc = f"tc_{activation_type}"
    cache_key_lorsa = f"lorsa_{activation_type}"
    
    if cache_key in _max_activations_cache:
        cached = _max_activations_cache[cache_key]
        if cache_key_tc in cached and cache_key_lorsa in cached:
            return cached[cache_key_tc], cached[cache_key_lorsa]
    
    # 尝试从文件加载
    try:
        exp_dir = Path(__file__).parent.parent / "exp" / "44global_weight"
        
        if activation_type == "max":
            tc_path = exp_dir / "tc_feature_max_activations.pt"
            lorsa_path = exp_dir / "lorsa_feature_max_activations.pt"
        else:  # mean
            tc_path = exp_dir / "tc_feature_mean_activations.pt"
            lorsa_path = exp_dir / "lorsa_feature_mean_activations.pt"
        
        if tc_path.exists() and lorsa_path.exists():
            tc_acts = torch.load(tc_path, map_location=device)
            lorsa_acts = torch.load(lorsa_path, map_location=device)
            
            if activation_type == "max":
                # 对于max activations，使用平均值（根据notebook中的逻辑）
                # max activations 文件通常是3维的 [n_layers, n_samples, n_features]
                # 需要取平均值得到 [n_layers, n_features]，然后扩展为 [n_layers, 16384]
                if tc_acts.dim() == 3:
                    tc_mean = torch.mean(tc_acts, dim=1)  # [n_layers, n_features]
                    tc_acts = tc_mean[:, None].repeat(1, 16384)  # [n_layers, 16384]
                elif tc_acts.dim() == 2:
                    # 如果已经是2维的，直接扩展
                    tc_acts = tc_acts[:, None].repeat(1, 16384)
                
                if lorsa_acts.dim() == 3:
                    lorsa_mean = torch.mean(lorsa_acts, dim=1)  # [n_layers, n_features]
                    lorsa_acts = lorsa_mean[:, None].repeat(1, 16384)  # [n_layers, 16384]
                elif lorsa_acts.dim() == 2:
                    # 如果已经是2维的，直接扩展
                    lorsa_acts = lorsa_acts[:, None].repeat(1, 16384)
            else:  # mean
                # 对于mean activations，根据notebook，文件加载出来应该是 [15, 16384] 的形状
                # 直接使用，不需要再取平均值或扩展
                if tc_acts.dim() == 2:
                    # 如果已经是 [n_layers, n_features] 的形状，直接使用
                    # 根据notebook，应该是 [15, 16384]，不需要额外处理
                    pass
                elif tc_acts.dim() == 3:
                    # 如果是3维的，取平均值（这种情况不应该发生，但为了兼容性保留）
                    print(f"⚠️ 警告: mean activations 文件是3维的，将取平均值")
                    tc_mean = torch.mean(tc_acts, dim=1)  # [n_layers, n_features]
                    tc_acts = tc_mean
                    if tc_acts.shape[1] != 16384:
                        # 如果特征数不是16384，需要扩展
                        tc_acts = tc_acts[:, None].repeat(1, 16384)
                
                if lorsa_acts.dim() == 2:
                    # 如果已经是 [n_layers, n_features] 的形状，直接使用
                    # 根据notebook，应该是 [15, 16384]，不需要额外处理
                    pass
                elif lorsa_acts.dim() == 3:
                    # 如果是3维的，取平均值（这种情况不应该发生，但为了兼容性保留）
                    print(f"⚠️ 警告: mean activations 文件是3维的，将取平均值")
                    lorsa_mean = torch.mean(lorsa_acts, dim=1)  # [n_layers, n_features]
                    lorsa_acts = lorsa_mean
                    if lorsa_acts.shape[1] != 16384:
                        # 如果特征数不是16384，需要扩展
                        lorsa_acts = lorsa_acts[:, None].repeat(1, 16384)
            
            # 更新缓存
            if cache_key not in _max_activations_cache:
                _max_activations_cache[cache_key] = {}
            _max_activations_cache[cache_key][cache_key_tc] = tc_acts
            _max_activations_cache[cache_key][cache_key_lorsa] = lorsa_acts
            
            print(f"✅ 加载{activation_type} activations数据: {cache_key}")
            return tc_acts, lorsa_acts
    except Exception as e:
        print(f"⚠️ 加载{activation_type} activations失败，将使用默认值: {e}")
    
    # 如果加载失败，创建默认值（使用1.0作为占位符）
    n_layers = 15
    n_features = 16384
    tc_default = torch.ones(n_layers, n_features, device=device, dtype=torch.float32)
    lorsa_default = torch.ones(n_layers, n_features, device=device, dtype=torch.float32)
    
    # 更新缓存
    if cache_key not in _max_activations_cache:
        _max_activations_cache[cache_key] = {}
    _max_activations_cache[cache_key][cache_key_tc] = tc_default
    _max_activations_cache[cache_key][cache_key_lorsa] = lorsa_default
    
    return tc_default, lorsa_default


def construct_topk(feature_list: List[Tuple[str, float]], k: int) -> List[Tuple[str, float]]:
    """选择top k特征"""
    sorted_features = sorted(feature_list, key=lambda x: x[1], reverse=True)
    return sorted_features[:k]


def construct_name(global_weights: torch.Tensor, prefix: str, k: int) -> List[Tuple[str, float]]:
    """构建特征名称列表"""
    n_features = global_weights.shape[0]
    feature_list = [(prefix.format(i), global_weights[i].item()) for i in range(n_features)]
    return construct_topk(feature_list, k)


def tc_global_weight_in(
    transcoders: Dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    layer_idx: int,
    feature_idx: int,
    tc_activations: torch.Tensor,
    lorsa_activations: torch.Tensor,
    k: int = 100,
    layer_filter: List[int] | None = None,
) -> List[Tuple[str, float]]:
    """计算TC feature的输入全局权重"""
    if layer_filter is not None:
        print(f"🔍 tc_global_weight_in: layer_filter={layer_filter}")
    # 获取encoder向量
    f_enc = transcoders[layer_idx].W_E[:, feature_idx]  # [d_model]

    feature_list_tc = []
    # TC 0 ~ layer_idx-1
    for i in range(layer_idx):
        # 如果有层过滤器，只包含指定层
        if layer_filter is not None and i not in layer_filter:
            continue
        if layer_filter is not None:  # 调试信息
            print(f"✅ tc_global_weight_in: 处理TC层{i} (过滤器: {layer_filter})")
        # 获取decoder矩阵（带activations权重）
        f_dec = transcoders[i].W_D * tc_activations[i, :, None]  # [d_sae, d_model]
        V = einops.einsum(f_enc, f_dec, "d_model, d_sae d_model -> d_sae")  # [d_sae]
        feature_list_tc.extend(construct_name(V, f"BT4_tc_L{i}M_k30_e16#{{}}", k=k))

    feature_list_lorsa = []
    # LoRSA 0 ~ layer_idx
    for i in range(layer_idx + 1):
        # 如果有层过滤器，只包含指定层
        if layer_filter is not None and i not in layer_filter:
            continue
        if layer_filter is not None:  # 调试信息
            print(f"✅ tc_global_weight_in: 处理LoRSA层{i} (过滤器: {layer_filter})")
        # 获取decoder矩阵（带activations权重）
        f_dec = lorsas[i].W_O * lorsa_activations[i, :, None]  # [d_sae, d_model]
        V = einops.einsum(f_enc, f_dec, "d_model, d_sae d_model -> d_sae")  # [d_sae]
        feature_list_lorsa.extend(construct_name(V, f"BT4_lorsa_L{i}A_k30_e16#{{}}", k=k))

    feature_list_tc = construct_topk(feature_list_tc, k)
    feature_list_lorsa = construct_topk(feature_list_lorsa, k)
    return feature_list_tc + feature_list_lorsa


def lorsa_global_weight_in(
    transcoders: Dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    layer_idx: int,
    feature_idx: int,
    tc_activations: torch.Tensor,
    lorsa_activations: torch.Tensor,
    k: int = 100,
    layer_filter: List[int] | None = None,
) -> List[Tuple[str, float]]:
    """计算LoRSA feature的输入全局权重"""
    if layer_filter is not None:
        print(f"🔍 lorsa_global_weight_in: layer_filter={layer_filter}")
    # 获取encoder向量
    f_enc = lorsas[layer_idx].W_V[feature_idx, :]  # [d_model]

    feature_list_tc = []
    # TC 0 ~ layer_idx-1
    for i in range(layer_idx):
        # 如果有层过滤器，只包含指定层
        if layer_filter is not None and i not in layer_filter:
            continue
        if layer_filter is not None:  # 调试信息
            print(f"✅ lorsa_global_weight_in: 处理TC层{i} (过滤器: {layer_filter})")
        f_dec = transcoders[i].W_D * tc_activations[i, :, None]  # [d_sae, d_model]
        V = einops.einsum(f_enc, f_dec, "d_model, d_sae d_model -> d_sae")  # [d_sae]
        feature_list_tc.extend(construct_name(V, f"BT4_tc_L{i}M_k30_e16#{{}}", k=k))

    feature_list_lorsa = []
    # LoRSA 0 ~ layer_idx-1
    for i in range(layer_idx):
        # 如果有层过滤器，只包含指定层
        if layer_filter is not None and i not in layer_filter:
            continue
        if layer_filter is not None:  # 调试信息
            print(f"✅ lorsa_global_weight_in: 处理LoRSA层{i} (过滤器: {layer_filter})")
        f_dec = lorsas[i].W_O * lorsa_activations[i, :, None]  # [d_sae, d_model]
        V = einops.einsum(f_enc, f_dec, "d_model, d_sae d_model -> d_sae")  # [d_sae]
        feature_list_lorsa.extend(construct_name(V, f"BT4_lorsa_L{i}A_k30_e16#{{}}", k=k))

    feature_list_tc = construct_topk(feature_list_tc, k)
    feature_list_lorsa = construct_topk(feature_list_lorsa, k)
    return feature_list_tc + feature_list_lorsa


def tc_global_weight_out(
    transcoders: Dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    layer_idx: int,
    feature_idx: int,
    tc_activations: torch.Tensor,
    lorsa_activations: torch.Tensor,
    k: int = 100,
    layer_filter: List[int] | None = None,
) -> List[Tuple[str, float]]:
    """计算TC feature的输出全局权重"""
    if layer_filter is not None:
        print(f"🔍 tc_global_weight_out: layer_filter={layer_filter}")
    # 获取decoder向量（带activations权重）
    f_dec = transcoders[layer_idx].W_D[feature_idx, :] * tc_activations[layer_idx, feature_idx]  # [d_model]

    feature_list_tc = []
    for i in range(layer_idx + 1, len(transcoders)):
        # 如果有层过滤器，只包含指定层
        if layer_filter is not None and i not in layer_filter:
            continue
        if layer_filter is not None:  # 调试信息
            print(f"✅ tc_global_weight_out: 处理TC层{i} (过滤器: {layer_filter})")
        f_enc = transcoders[i].W_D * tc_activations[i, :, None]  # [d_sae, d_model]
        V = einops.einsum(f_dec, f_enc, "d_model, d_sae d_model -> d_sae")  # [d_sae]
        feature_list_tc.extend(construct_name(V, f"BT4_tc_L{i}M_k30_e16#{{}}", k=k))

    feature_list_lorsa = []
    for i in range(layer_idx + 1, len(transcoders)):
        # 如果有层过滤器，只包含指定层
        if layer_filter is not None and i not in layer_filter:
            continue
        if layer_filter is not None:  # 调试信息
            print(f"✅ tc_global_weight_out: 处理LoRSA层{i} (过滤器: {layer_filter})")
        f_enc = lorsas[i].W_V  # [d_sae, d_model]
        V = einops.einsum(f_dec, f_enc, "d_model, d_sae d_model -> d_sae")  # [d_sae]
        feature_list_lorsa.extend(construct_name(V, f"BT4_lorsa_L{i}A_k30_e16#{{}}", k=k))

    feature_list_tc = construct_topk(feature_list_tc, k)
    feature_list_lorsa = construct_topk(feature_list_lorsa, k)
    return feature_list_tc + feature_list_lorsa


def lorsa_global_weight_out(
    transcoders: Dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    layer_idx: int,
    feature_idx: int,
    tc_activations: torch.Tensor,
    lorsa_activations: torch.Tensor,
    k: int = 100,
    layer_filter: List[int] | None = None,
) -> List[Tuple[str, float]]:
    """计算LoRSA feature的输出全局权重"""
    if layer_filter is not None:
        print(f"🔍 lorsa_global_weight_out: layer_filter={layer_filter}")
    # 获取decoder向量（带activations权重）
    f_dec = lorsas[layer_idx].W_O[feature_idx, :] * lorsa_activations[layer_idx, feature_idx]  # [d_model]

    feature_list_tc = []
    # TC layer_idx ~ n_layers-1
    for i in range(layer_idx, len(transcoders)):
        # 如果有层过滤器，只包含指定层
        if layer_filter is not None and i not in layer_filter:
            continue
        if layer_filter is not None:  # 调试信息
            print(f"✅ lorsa_global_weight_out: 处理TC层{i} (过滤器: {layer_filter})")
        f_enc = transcoders[i].W_D * tc_activations[i, :, None]  # [d_sae, d_model]
        V = einops.einsum(f_dec, f_enc, "d_model, d_sae d_model -> d_sae")  # [d_sae]
        feature_list_tc.extend(construct_name(V, f"BT4_tc_L{i}M_k30_e16#{{}}", k=k))

    feature_list_lorsa = []
    # LoRSA layer_idx+1 ~ n_layers-1
    for i in range(layer_idx + 1, len(transcoders)):
        # 如果有层过滤器，只包含指定层
        if layer_filter is not None and i not in layer_filter:
            continue
        if layer_filter is not None:  # 调试信息
            print(f"✅ lorsa_global_weight_out: 处理LoRSA层{i} (过滤器: {layer_filter})")
        f_enc = lorsas[i].W_V  # [d_sae, d_model]
        V = einops.einsum(f_dec, f_enc, "d_model, d_sae d_model -> d_sae")  # [d_sae]
        feature_list_lorsa.extend(construct_name(V, f"BT4_lorsa_L{i}A_k30_e16#{{}}", k=k))

    feature_list_tc = construct_topk(feature_list_tc, k)
    feature_list_lorsa = construct_topk(feature_list_lorsa, k)
    return feature_list_tc + feature_list_lorsa




