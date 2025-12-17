"""Global weight calculation functions for feature analysis."""
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import einops

from lm_saes.sae import SparseAutoEncoder
from lm_saes import LowRankSparseAttention


# Global cache for max activations
_max_activations_cache: Dict[str, Dict[str, torch.Tensor]] = {}  # combo_key -> {"tc": tensor, "lorsa": tensor}


def load_max_activations(
    sae_combo_id: str,
    device: str = "cuda",
    get_bt4_sae_combo=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    加载max activations数据
    
    Args:
        sae_combo_id: SAE组合ID
        device: 设备（cuda或cpu）
        get_bt4_sae_combo: 获取SAE组合配置的函数
    
    Returns:
        (tc_max_activations, lorsa_max_activations) 元组
    """
    global _max_activations_cache
    
    if get_bt4_sae_combo is None:
        raise ValueError("get_bt4_sae_combo function is required")
    
    combo_cfg = get_bt4_sae_combo(sae_combo_id)
    normalized_combo_id = combo_cfg["id"]
    cache_key = f"BT4_{normalized_combo_id}"
    
    if cache_key in _max_activations_cache:
        cached = _max_activations_cache[cache_key]
        return cached["tc"], cached["lorsa"]
    
    # 尝试从文件加载
    try:
        exp_dir = Path(__file__).parent.parent / "exp" / "44global_weight"
        tc_path = exp_dir / "tc_feature_max_activations.pt"
        lorsa_path = exp_dir / "lorsa_feature_max_activations.pt"
        
        if tc_path.exists() and lorsa_path.exists():
            tc_max_acts = torch.load(tc_path, map_location=device)
            lorsa_max_acts = torch.load(lorsa_path, map_location=device)
            
            # 使用平均值（根据notebook中的逻辑）
            tc_mean = torch.mean(tc_max_acts, dim=1)
            lorsa_mean = torch.mean(lorsa_max_acts, dim=1)
            tc_max_acts = tc_mean[:, None].repeat(1, 16384)
            lorsa_max_acts = lorsa_mean[:, None].repeat(1, 16384)
            
            _max_activations_cache[cache_key] = {
                "tc": tc_max_acts,
                "lorsa": lorsa_max_acts
            }
            print(f"✅ 加载max activations数据: {cache_key}")
            return tc_max_acts, lorsa_max_acts
    except Exception as e:
        print(f"⚠️ 加载max activations失败，将使用默认值: {e}")
    
    # 如果加载失败，创建默认值（使用1.0作为占位符）
    n_layers = 15
    n_features = 16384
    tc_default = torch.ones(n_layers, n_features, device=device, dtype=torch.float32)
    lorsa_default = torch.ones(n_layers, n_features, device=device, dtype=torch.float32)
    
    _max_activations_cache[cache_key] = {
        "tc": tc_default,
        "lorsa": lorsa_default
    }
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
    tc_max_activations: torch.Tensor,
    lorsa_max_activations: torch.Tensor,
    k: int = 100,
) -> List[Tuple[str, float]]:
    """计算TC feature的输入全局权重"""
    # 获取encoder向量
    f_enc = transcoders[layer_idx].W_E[:, feature_idx]  # [d_model]
    
    feature_list_tc = []
    # TC 0 ~ layer_idx-1
    for i in range(layer_idx):
        # 获取decoder矩阵（带max activations权重）
        f_dec = transcoders[i].W_D * tc_max_activations[i, :, None]  # [d_sae, d_model]
        V = einops.einsum(f_enc, f_dec, "d_model, d_sae d_model -> d_sae")  # [d_sae]
        feature_list_tc.extend(construct_name(V, f"BT4_tc_L{i}M_k30_e16#{{}}", k=k))
    
    feature_list_lorsa = []
    # LoRSA 0 ~ layer_idx
    for i in range(layer_idx + 1):
        # 获取decoder矩阵（带max activations权重）
        f_dec = lorsas[i].W_O * lorsa_max_activations[i, :, None]  # [d_sae, d_model]
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
    tc_max_activations: torch.Tensor,
    lorsa_max_activations: torch.Tensor,
    k: int = 100,
) -> List[Tuple[str, float]]:
    """计算LoRSA feature的输入全局权重"""
    # 获取encoder向量
    f_enc = lorsas[layer_idx].W_V[feature_idx, :]  # [d_model]
    
    feature_list_tc = []
    # TC 0 ~ layer_idx-1
    for i in range(layer_idx):
        f_dec = transcoders[i].W_D * tc_max_activations[i, :, None]  # [d_sae, d_model]
        V = einops.einsum(f_enc, f_dec, "d_model, d_sae d_model -> d_sae")  # [d_sae]
        feature_list_tc.extend(construct_name(V, f"BT4_tc_L{i}M_k30_e16#{{}}", k=k))
    
    feature_list_lorsa = []
    # LoRSA 0 ~ layer_idx-1
    for i in range(layer_idx):
        f_dec = lorsas[i].W_O * lorsa_max_activations[i, :, None]  # [d_sae, d_model]
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
    tc_max_activations: torch.Tensor,
    lorsa_max_activations: torch.Tensor,
    k: int = 100,
) -> List[Tuple[str, float]]:
    """计算TC feature的输出全局权重"""
    # 获取decoder向量（带max activations权重）
    f_dec = transcoders[layer_idx].W_D[feature_idx, :] * tc_max_activations[layer_idx, feature_idx]  # [d_model]
    
    feature_list_tc = []
    for i in range(layer_idx + 1, len(transcoders)):
        f_enc = transcoders[i].W_D * tc_max_activations[i, :, None]  # [d_sae, d_model]
        V = einops.einsum(f_enc, f_dec, "d_sae d_model, d_model -> d_sae")  # [d_sae]
        feature_list_tc.extend(construct_name(V, f"BT4_tc_L{i}M_k30_e16#{{}}", k=k))
    
    feature_list_lorsa = []
    for i in range(layer_idx + 1, len(transcoders)):
        f_enc = lorsas[i].W_V  # [d_sae, d_model]
        V = einops.einsum(f_enc, f_dec, "d_sae d_model, d_model -> d_sae")  # [d_sae]
        feature_list_lorsa.extend(construct_name(V, f"BT4_lorsa_L{i}A_k30_e16#{{}}", k=k))
    
    feature_list_tc = construct_topk(feature_list_tc, k)
    feature_list_lorsa = construct_topk(feature_list_lorsa, k)
    return feature_list_tc + feature_list_lorsa


def lorsa_global_weight_out(
    transcoders: Dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    layer_idx: int,
    feature_idx: int,
    tc_max_activations: torch.Tensor,
    lorsa_max_activations: torch.Tensor,
    k: int = 100,
) -> List[Tuple[str, float]]:
    """计算LoRSA feature的输出全局权重"""
    # 获取decoder向量（带max activations权重）
    f_dec = lorsas[layer_idx].W_O[feature_idx, :] * lorsa_max_activations[layer_idx, feature_idx]  # [d_model]
    
    feature_list_tc = []
    # TC layer_idx ~ n_layers-1
    for i in range(layer_idx, len(transcoders)):
        f_enc = transcoders[i].W_D * tc_max_activations[i, :, None]  # [d_sae, d_model]
        V = einops.einsum(f_enc, f_dec, "d_sae d_model, d_model -> d_sae")  # [d_sae]
        feature_list_tc.extend(construct_name(V, f"BT4_tc_L{i}M_k30_e16#{{}}", k=k))
    
    feature_list_lorsa = []
    # LoRSA layer_idx+1 ~ n_layers-1
    for i in range(layer_idx + 1, len(transcoders)):
        f_enc = lorsas[i].W_V  # [d_sae, d_model]
        V = einops.einsum(f_enc, f_dec, "d_sae d_model, d_model -> d_sae")  # [d_sae]
        feature_list_lorsa.extend(construct_name(V, f"BT4_lorsa_L{i}A_k30_e16#{{}}", k=k))
    
    feature_list_tc = construct_topk(feature_list_tc, k)
    feature_list_lorsa = construct_topk(feature_list_lorsa, k)
    return feature_list_tc + feature_list_lorsa
