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
    get all features activated at the specified layer and position
    
    Args:
        model: HookedTransformer model
        transcoders: dictionary, layer number -> Transcoder SAE
        lorsas: list, Lorsa SAE (indexed by layer)
        fen: FEN string
        layer: layer number (0-14)
        pos: position index (0-63)
        component_type: component type, "attn" or "mlp"
    
    Returns:
        dictionary, containing:
        - "attn_features": if attn, return activated Lorsa features
        - "mlp_features": if mlp, return activated Transcoder features
        each feature contains:
        - "feature_index": feature index
        - "activation_value": activation value
    """
    if layer < 0 or layer >= 15:
        raise ValueError(f"layer number must be between 0-14, current value: {layer}")
    
    if pos < 0 or pos >= 64:
        raise ValueError(f"position index must be between 0-63, current value: {pos}")
    
    if component_type not in ["attn", "mlp"]:
        raise ValueError(f"component_type must be 'attn' or 'mlp', current value: {component_type}")
    
    # run model to get activation values
    with torch.no_grad():
        _, cache = model.run_with_cache(fen, prepend_bos=False)
    
    result = {}
    
    # get activated features of Attention (Lorsa)
    if component_type == "attn":
        if layer >= len(lorsas):
            raise ValueError(f"layer {layer} out of Lorsa range (total {len(lorsas)} layers)")
        
        # get input activation value of attention
        hook_name = f"blocks.{layer}.hook_attn_in"
        if hook_name not in cache:
            available_hooks = [k for k in cache.keys() if f"blocks.{layer}" in str(k)]
            raise ValueError(
                f"cannot find hook point {hook_name}."
                f"available hook points: {available_hooks[:10]}"
            )
        
        attn_input = cache[hook_name]  # [batch, seq_len, d_model]
        
        # ensure batch dimension
        if attn_input.dim() == 2:
            attn_input = attn_input.unsqueeze(0)  # [1, seq_len, d_model]
        
        # use Lorsa to encode
        lorsa = lorsas[layer]
        feature_acts = lorsa.encode(attn_input)  # [batch, seq_len, d_sae]
        
        # get activation value at the specified position
        if feature_acts.dim() == 3:
            pos_activations = feature_acts[0, pos, :]  # [d_sae]
        else:
            pos_activations = feature_acts[pos, :]  # [d_sae]
        
        # find non-zero activated features
        nonzero_mask = pos_activations != 0
        nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False).squeeze(-1)
        nonzero_values = pos_activations[nonzero_mask]
        
        # build result list
        attn_features = []
        for idx, val in zip(nonzero_indices.cpu().numpy(), nonzero_values.cpu().numpy()):
            attn_features.append({
                "feature_index": int(idx),
                "activation_value": float(val)
            })
        
        # sort by activation value absolute value (from largest to smallest)
        attn_features.sort(key=lambda x: abs(x["activation_value"]), reverse=True)
        
        result["attn_features"] = attn_features
    
    # get activated features of MLP (Transcoder)
    if component_type == "mlp":
        if layer not in transcoders:
            raise ValueError(f"layer {layer} not in transcoders")
        
        # get input activation value of MLP (resid_mid_after_ln)
        hook_name = f"blocks.{layer}.resid_mid_after_ln"
        if hook_name not in cache:
            # try alternative hook points
            alt_hook_name = f"blocks.{layer}.ln2.hook_normalized"
            if alt_hook_name in cache:
                hook_name = alt_hook_name
            else:
                available_hooks = [k for k in cache.keys() if f"blocks.{layer}" in str(k)]
                raise ValueError(
                    f"cannot find hook point {hook_name} or {alt_hook_name}."
                    f"available hook points: {available_hooks[:10]}"
                )
        
        mlp_input = cache[hook_name]  # [batch, seq_len, d_model]
        
        # ensure batch dimension
        if mlp_input.dim() == 2:
            mlp_input = mlp_input.unsqueeze(0)  # [1, seq_len, d_model]
        
        # use Transcoder to encode
        transcoder = transcoders[layer]
        feature_acts = transcoder.encode(mlp_input)  # [batch, seq_len, d_sae]
        
        # get activation value at the specified position
        if feature_acts.dim() == 3:
            pos_activations = feature_acts[0, pos, :]  # [d_sae]
        else:
            pos_activations = feature_acts[pos, :]  # [d_sae]
        
        # find non-zero activated features
        nonzero_mask = pos_activations != 0
        nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False).squeeze(-1)
        nonzero_values = pos_activations[nonzero_mask]
        
        # build result list
        mlp_features = []
        for idx, val in zip(nonzero_indices.cpu().numpy(), nonzero_values.cpu().numpy()):
            mlp_features.append({
                "feature_index": int(idx),
                "activation_value": float(val)
            })
        
        # sort by activation value absolute value (from largest to smallest)
        mlp_features.sort(key=lambda x: abs(x["activation_value"]), reverse=True)
        
        result["mlp_features"] = mlp_features
    
    return result
