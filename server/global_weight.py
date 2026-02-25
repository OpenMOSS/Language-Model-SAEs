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
    global _max_activations_cache
    
    if get_bt4_sae_combo is None:
        raise ValueError("get_bt4_sae_combo function is required")
    
    if activation_type not in ["max", "mean"]:
        raise ValueError(f"activation_type must be 'max' or 'mean', current value: {activation_type}")
    
    combo_cfg = get_bt4_sae_combo(sae_combo_id)
    normalized_combo_id = combo_cfg["id"]
    cache_key = f"BT4_{normalized_combo_id}"
    
    # check cache
    cache_key_tc = f"tc_{activation_type}"
    cache_key_lorsa = f"lorsa_{activation_type}"
    
    if cache_key in _max_activations_cache:
        cached = _max_activations_cache[cache_key]
        if cache_key_tc in cached and cache_key_lorsa in cached:
            return cached[cache_key_tc], cached[cache_key_lorsa]
    
    # try to load from file
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
                # for max activations, use average value (according to notebook logic)
                # max activations file is usually 3D [n_layers, n_samples, n_features]
                # need to take average to get [n_layers, n_features], then extend to [n_layers, 16384]
                if tc_acts.dim() == 3:
                    tc_mean = torch.mean(tc_acts, dim=1)  # [n_layers, n_features]
                    tc_acts = tc_mean[:, None].repeat(1, 16384)  # [n_layers, 16384]
                elif tc_acts.dim() == 2:
                    # if already 2D, extend directly
                    tc_acts = tc_acts[:, None].repeat(1, 16384)
                
                if lorsa_acts.dim() == 3:
                    lorsa_mean = torch.mean(lorsa_acts, dim=1)  # [n_layers, n_features]
                    lorsa_acts = lorsa_mean[:, None].repeat(1, 16384)  # [n_layers, 16384]
                elif lorsa_acts.dim() == 2:
                    # if already 2D, extend directly
                    lorsa_acts = lorsa_acts[:, None].repeat(1, 16384)
            else:  # mean
                # for mean activations, according to notebook, the file should be loaded as [15, 16384] shape
                # use directly, no need to take average or extend
                if tc_acts.dim() == 2:
                    # if already [n_layers, n_features] shape, use directly
                    # according to notebook, should be [15, 16384], no need to extra processing
                    pass
                elif tc_acts.dim() == 3:
                    # if 3D, take average (should not happen, kept for backward compatibility)
                    print("Warning: mean activations file is 3D, will take average")
                    tc_mean = torch.mean(tc_acts, dim=1)  # [n_layers, n_features]
                    tc_acts = tc_mean
                    if tc_acts.shape[1] != 16384:
                        # if feature number is not 16384, need to extend
                        tc_acts = tc_acts[:, None].repeat(1, 16384)
                
                if lorsa_acts.dim() == 2:
                    # if already [n_layers, n_features] shape, use directly
                    # according to notebook, should be [15, 16384], no need to extra processing
                    pass
                elif lorsa_acts.dim() == 3:
                    # If 3D, take average (should not happen, kept for backward compatibility)
                    print("Warning: mean activations file is 3D, will take average")
                    lorsa_mean = torch.mean(lorsa_acts, dim=1)  # [n_layers, n_features]
                    lorsa_acts = lorsa_mean
                    if lorsa_acts.shape[1] != 16384:
                        # if feature number is not 16384, need to extend
                        lorsa_acts = lorsa_acts[:, None].repeat(1, 16384)
            
            # update cache
            if cache_key not in _max_activations_cache:
                _max_activations_cache[cache_key] = {}
            _max_activations_cache[cache_key][cache_key_tc] = tc_acts
            _max_activations_cache[cache_key][cache_key_lorsa] = lorsa_acts
            
            print(f"Load {activation_type} activations data: {cache_key}")
            return tc_acts, lorsa_acts
    except Exception as e:
        print(f"Load {activation_type} activations failed, will use default value: {e}")
    
    # if load failed, create default value (use 1.0 as placeholder)
    n_layers = 15
    n_features = 16384
    tc_default = torch.ones(n_layers, n_features, device=device, dtype=torch.float32)
    lorsa_default = torch.ones(n_layers, n_features, device=device, dtype=torch.float32)
    
    # update cache
    if cache_key not in _max_activations_cache:
        _max_activations_cache[cache_key] = {}
    _max_activations_cache[cache_key][cache_key_tc] = tc_default
    _max_activations_cache[cache_key][cache_key_lorsa] = lorsa_default
    
    return tc_default, lorsa_default


def construct_topk(feature_list: List[Tuple[str, float]], k: int) -> List[Tuple[str, float]]:
    """select top k features"""
    sorted_features = sorted(feature_list, key=lambda x: x[1], reverse=True)
    return sorted_features[:k]


def construct_name(global_weights: torch.Tensor, prefix: str, k: int) -> List[Tuple[str, float]]:
    """construct feature name list"""
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
    """calculate the input global weight of TC feature"""
    if layer_filter is not None:
        print(f"🔍 tc_global_weight_in: layer_filter={layer_filter}")
    # get encoder vector
    f_enc = transcoders[layer_idx].W_E[:, feature_idx]  # [d_model]

    feature_list_tc = []
    # TC 0 to layer_idx-1
    for i in range(layer_idx):
        # if there is layer filter, only include specified layers
        if layer_filter is not None and i not in layer_filter:
            continue
        if layer_filter is not None:  # debug information
            print(f"tc_global_weight_in: process TC layer {i} (filter: {layer_filter})")
        # get decoder matrix (with activations weight)
        f_dec = transcoders[i].W_D * tc_activations[i, :, None]  # [d_sae, d_model]
        V = einops.einsum(f_enc, f_dec, "d_model, d_sae d_model -> d_sae")  # [d_sae]
        feature_list_tc.extend(construct_name(V, f"BT4_tc_L{i}M_k30_e16#{{}}", k=k))

    feature_list_lorsa = []
    # Lorsa 0 to layer_idx
    for i in range(layer_idx + 1):
        # if there is layer filter, only include specified layers
        if layer_filter is not None and i not in layer_filter:
            continue
        if layer_filter is not None:  # debug information
            print(f"tc_global_weight_in: process Lorsa layer {i} (filter: {layer_filter})")
        # get decoder matrix (with activations weight)
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
    """calculate the input global weight of Lorsa feature"""
    if layer_filter is not None:
        print(f"🔍 lorsa_global_weight_in: layer_filter={layer_filter}")
    # get encoder vector
    f_enc = lorsas[layer_idx].W_V[feature_idx, :]  # [d_model]

    feature_list_tc = []
    # TC 0 to layer_idx-1
    for i in range(layer_idx):
        # if there is layer filter, only include specified layers
        if layer_filter is not None and i not in layer_filter:
            continue
        if layer_filter is not None:  # debug information
            print(f"lorsa_global_weight_in: process TC layer {i} (filter: {layer_filter})")
        f_dec = transcoders[i].W_D * tc_activations[i, :, None]  # [d_sae, d_model]
        V = einops.einsum(f_enc, f_dec, "d_model, d_sae d_model -> d_sae")  # [d_sae]
        feature_list_tc.extend(construct_name(V, f"BT4_tc_L{i}M_k30_e16#{{}}", k=k))

    feature_list_lorsa = []
    # Lorsa 0 to layer_idx-1
    for i in range(layer_idx):
        # if there is layer filter, only include specified layers
        if layer_filter is not None and i not in layer_filter:
            continue
        if layer_filter is not None:  # debug information
            print(f"lorsa_global_weight_in: process Lorsa layer {i} (filter: {layer_filter})")
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
    """calculate the output global weight of TC feature"""
    if layer_filter is not None:
        print(f"🔍 tc_global_weight_out: layer_filter={layer_filter}")
    # get decoder vector (with activations weight)
    f_dec = transcoders[layer_idx].W_D[feature_idx, :] * tc_activations[layer_idx, feature_idx]  # [d_model]

    feature_list_tc = []
    for i in range(layer_idx + 1, len(transcoders)):
        # if there is layer filter, only include specified layers
        if layer_filter is not None and i not in layer_filter:
            continue
        if layer_filter is not None:  # debug information
            print(f"tc_global_weight_out: process TC layer {i} (filter: {layer_filter})")
        f_enc = transcoders[i].W_D * tc_activations[i, :, None]  # [d_sae, d_model]
        V = einops.einsum(f_dec, f_enc, "d_model, d_sae d_model -> d_sae")  # [d_sae]
        feature_list_tc.extend(construct_name(V, f"BT4_tc_L{i}M_k30_e16#{{}}", k=k))

    feature_list_lorsa = []
    for i in range(layer_idx + 1, len(transcoders)):
        # if there is layer filter, only include specified layers
        if layer_filter is not None and i not in layer_filter:
            continue
        if layer_filter is not None:  # debug information
            print(f"tc_global_weight_out: process Lorsa layer {i} (filter: {layer_filter})")
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
    """calculate the output global weight of Lorsa feature"""
    if layer_filter is not None:
        print(f"🔍 lorsa_global_weight_out: layer_filter={layer_filter}")
    # get decoder vector (with activations weight)
    f_dec = lorsas[layer_idx].W_O[feature_idx, :] * lorsa_activations[layer_idx, feature_idx]  # [d_model]

    feature_list_tc = []
    # TC layer_idx to n_layers-1
    for i in range(layer_idx, len(transcoders)):
        # if there is layer filter, only include specified layers
        if layer_filter is not None and i not in layer_filter:
            continue
        if layer_filter is not None:  # debug information
            print(f"lorsa_global_weight_out: process TC layer {i} (filter: {layer_filter})")
        f_enc = transcoders[i].W_D * tc_activations[i, :, None]  # [d_sae, d_model]
        V = einops.einsum(f_dec, f_enc, "d_model, d_sae d_model -> d_sae")  # [d_sae]
        feature_list_tc.extend(construct_name(V, f"BT4_tc_L{i}M_k30_e16#{{}}", k=k))

    feature_list_lorsa = []
    # Lorsa layer_idx+1 to n_layers-1
    for i in range(layer_idx + 1, len(transcoders)):
        # if there is layer filter, only include specified layers
        if layer_filter is not None and i not in layer_filter:
            continue
        if layer_filter is not None:  # debug information
            print(f"lorsa_global_weight_out: process Lorsa layer {i} (filter: {layer_filter})")
        f_enc = lorsas[i].W_V  # [d_sae, d_model]
        V = einops.einsum(f_dec, f_enc, "d_model, d_sae d_model -> d_sae")  # [d_sae]
        feature_list_lorsa.extend(construct_name(V, f"BT4_lorsa_L{i}A_k30_e16#{{}}", k=k))

    feature_list_tc = construct_topk(feature_list_tc, k)
    feature_list_lorsa = construct_topk(feature_list_lorsa, k)
    return feature_list_tc + feature_list_lorsa




