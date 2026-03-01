
import chess
import torch
from transformer_lens import HookedTransformer
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from tqdm.auto import tqdm
from src.chess_utils.sae import get_feature_vector
from src.chess_utils import get_move_from_policy_output_with_prob

def collect_activated_features_at_position(
    pos_dict: Dict[str, int],
    position_name: str,
    model,
    transcoders,
    lorsas,
    fen: str,
    feature_types: List[str] = ["transcoder", "lorsa"],
    activation_threshold: float = 0.1,
    max_features_per_type: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Collect all SAE features activated at a given sequence position.

    Returns a list of dicts with keys:
        layer, feature_id, activation_value, feature_type,
        position_name, position_idx
    """
    if position_name not in pos_dict:
        raise ValueError(f"Unknown position_name: {position_name}")

    position_idx = pos_dict[position_name]
    results: List[Dict[str, Any]] = []

    _, cache = model.run_with_cache(fen, prepend_bos=False)

    n_layers = max(len(transcoders), len(lorsas))

    for layer in range(n_layers):
        for feature_type in feature_types:
            if feature_type == "transcoder":
                if layer >= len(transcoders):
                    continue
                sae = transcoders[layer]
                input_hook = f"blocks.{layer}.resid_mid_after_ln"
            elif feature_type == "lorsa":
                if layer >= len(lorsas):
                    continue
                sae = lorsas[layer]
                input_hook = f"blocks.{layer}.hook_attn_in"
            else:
                raise ValueError(f"Unknown feature_type: {feature_type}")

            if input_hook not in cache:
                continue

            activations = sae.encode(cache[input_hook])
            pos_acts = activations[0, position_idx]  # [n_features]

            mask = pos_acts > activation_threshold
            if not mask.any():
                continue

            feature_ids = torch.where(mask)[0]
            values = pos_acts[mask]

            pairs = sorted(
                zip(feature_ids.tolist(), values.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )

            if max_features_per_type is not None:
                pairs = pairs[:max_features_per_type]

            for feature_id, activation_value in pairs:
                results.append(
                    {
                        "layer": layer,
                        "feature_id": feature_id,
                        "activation_value": float(activation_value),
                        "feature_type": feature_type,
                        "position_name": position_name,
                        "position_idx": position_idx,
                    }
                )

    results.sort(key=lambda x: x["activation_value"], reverse=True)
    return results


def activation_steering_effect(
    model,
    transcoders,
    lorsas,
    feature_type: str,
    layer: int,
    pos: int,
    feature_id: int,
    steering_scale: float,
    fen: str,
    get_value: bool = True,
):
    model.reset_hooks()

    def _get_logits_and_value(output):
        logits = output[0]
        if logits.ndim == 2:
            logits = logits[0]
        value = (
            float(output[1][0][0] - output[1][0][2])
            if get_value else 0.0
        )
        return logits, value

    # original forward
    original_output, original_cache = model.run_with_cache(fen, prepend_bos=False)
    logits_original, original_value = _get_logits_and_value(original_output)

    # sparse activations
    if feature_type == "transcoder":
        input_hook = f"blocks.{layer}.resid_mid_after_ln"
        activations = transcoders[layer].encode(original_cache[input_hook])
        hook_point = f"blocks.{layer}.hook_mlp_out"
    elif feature_type == "lorsa":
        input_hook = f"blocks.{layer}.hook_attn_in"
        activations = lorsas[layer].encode(original_cache[input_hook])
        hook_point = f"blocks.{layer}.hook_attn_out"
    else:
        raise ValueError("feature_type must be 'transcoder' or 'lorsa'")

    activation_value = activations[0, pos, feature_id]
    if activation_value is None:
        return None

    # activation-weighted steering delta
    dec = get_feature_vector(lorsas, transcoders, feature_type, layer, feature_id)
    feature_contribution = activation_value * dec
    inject_val = (steering_scale - 1.0) * feature_contribution

    def _steer(act, hook):
        out = act.clone()
        delta = inject_val.to(out.device)
        out[(slice(None), pos) if out.dim() == 3 else (pos,)] += delta
        return out

    model.add_hook(hook_point, _steer)

    # steered forward
    modified_output, modified_cache = model.run_with_cache(fen, prepend_bos=False)
    model.reset_hooks()

    logits_modified, modified_value = _get_logits_and_value(modified_output)

    return {
        "feature_type": feature_type,
        "layer": layer,
        "pos": pos,
        "feature_id": feature_id,
        "activation_value": float(activation_value),
        "steering_scale": steering_scale,
        "hook_point": hook_point,

        "logits_original": logits_original.detach().cpu(),
        "logits_modified": logits_modified.detach().cpu(),
        "logits_diff": (logits_modified - logits_original).detach().cpu(),

        "original_value": original_value,
        "modified_value": modified_value,
        "value_diff": modified_value - original_value,

        "original_cache": original_cache,
        "modified_cache": modified_cache,
        "original_output": original_output,
        "modified_output": modified_output,
    }


def run_steering_for_position_features(
    position_name: str,
    features: List[Dict[str, Any]],
    model,
    transcoders,
    lorsas,
    fen: str,
    moves_tracing: Dict[str, str],
    steering_scale: float = 2.0,
    max_features: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run steering analysis for all activated features at a given position.

    Args:
        position_name: Name of the position
        features: List of activated features at this position
        model: HookedTransformer model
        transcoders: Dictionary of transcoder SAEs by layer
        lorsas: List of LORSA models by layer
        fen: FEN string of the board position
        moves_tracing: Dictionary mapping move names to UCI strings
        steering_scale: Steering scale factor
        max_features: Maximum number of features to analyze (for testing)

    Returns:
        List of steering results for each feature
    """
    if not moves_tracing:
        raise ValueError("moves_tracing cannot be empty")

    if max_features is not None:
        features = features[:max_features]

    results = []

    for feature_info in tqdm(features, desc=f"Steering {position_name}", unit="feature"):
        feature_id = feature_info['feature_id']
        layer = feature_info['layer']
        feature_type = feature_info['feature_type']
        position_idx = feature_info['position_idx']
        activation_value = feature_info['activation_value']

        try:
            # Run steering analysis
            steering_result = activation_steering_effect(
                model=model,
                transcoders=transcoders,
                lorsas=lorsas,
                feature_type=feature_type,
                layer=layer,
                pos=position_idx,
                feature_id=feature_id,
                steering_scale=steering_scale,
                fen=fen
            )

            if steering_result:
                # Calculate move probability changes
                move_probabilities = {}

                logits_original = torch.tensor(steering_result['logits_original'])
                logits_modified = torch.tensor(steering_result['logits_modified'])

                for move_name, move_uci in moves_tracing.items():
                    try:
                        original_prob = get_move_from_policy_output_with_prob(
                            logits_original, fen, move_uci=move_uci
                        )
                        modified_prob = get_move_from_policy_output_with_prob(
                            logits_modified, fen, move_uci=move_uci
                        )

                        prob_diff = None
                        if original_prob is not None and modified_prob is not None:
                            # Ensure both are float values before subtraction
                            orig_val = original_prob if isinstance(original_prob, (int, float)) else 0.0
                            mod_val = modified_prob if isinstance(modified_prob, (int, float)) else 0.0
                            prob_diff = float(mod_val) - float(orig_val)

                        move_probabilities[move_name] = {
                            'uci': move_uci,
                            'original_prob': original_prob,
                            'modified_prob': modified_prob,
                            'prob_diff': prob_diff
                        }
                    except Exception as e:
                        move_probabilities[move_name] = {
                            'uci': move_uci,
                            'original_prob': None,
                            'modified_prob': None,
                            'prob_diff': None
                        }

                # Build result
                result = {
                    'position_name': position_name,
                    'position_idx': position_idx,
                    'feature_type': feature_type,
                    'layer': layer,
                    'feature_id': feature_id,
                    'activation_value': steering_result['activation_value'],
                    'steering_scale': steering_scale,
                    'move_probabilities': move_probabilities,
                    'original_value': steering_result['original_value'],
                    'modified_value': steering_result['modified_value'],
                    'value_diff': steering_result['value_diff']
                }

                results.append(result)

            else:
                # Skip if no activation value
                pass

        except Exception as e:
            # Skip failed features
            pass

    return results


def nested_activation_steering_effect(
    model,
    transcoders,
    lorsas,
    first_steering: Dict[str, Any],  # first steering parameters: feature_type, layer, pos, feature_id, steering_scale
    second_steering: Dict[str, Any],  # second steering parameters: feature_type, layer, pos, feature_id, steering_scale
    fen: str,
    get_value: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    nested steering: on the basis of first steering, perform second steering, analyze the impact of second steering.
    
    Args:
        model: HookedTransformer model
        transcoders: dictionary, layer -> Transcoder SAE
        lorsas: list of Lorsa models (in layer order)
        first_steering: first steering parameters dictionary, contains:
            - feature_type: str ('transcoder' or 'lorsa')
            - layer: int
            - pos: int
            - feature_id: int
            - steering_scale: float
        second_steering: second steering parameters dictionary, contains:
            - feature_type: str ('transcoder' or 'lorsa')
            - layer: int
            - pos: int
            - feature_id: int
            - steering_scale: float
        fen: FEN string
        get_value: whether to get value output
    
    Returns:
        dictionary containing nested steering results, or None (if failed)
    """
    model.reset_hooks()
    
    def _get_logits_and_value(output):
        logits = output[0]
        if logits.ndim == 2:
            logits = logits[0]
        value = (
            float(output[1][0][0] - output[1][0][2])
            if get_value else 0.0
        )
        return logits, value
    
    # first step: original forward, get the basic data of first steering
    original_output, cache_original = model.run_with_cache(fen, prepend_bos=False)
    logits_original, original_value = _get_logits_and_value(original_output)
    
    # prepare first steering
    ft1 = first_steering['feature_type']
    layer1 = first_steering['layer']
    pos1 = first_steering['pos']
    feature_id1 = first_steering['feature_id']
    steering_scale1 = first_steering['steering_scale']
    
    if ft1 == "transcoder":
        input_hook1 = f"blocks.{layer1}.resid_mid_after_ln"
        activations1 = transcoders[layer1].encode(cache_original[input_hook1])
        hook_point1 = f"blocks.{layer1}.hook_mlp_out"
    elif ft1 == "lorsa":
        input_hook1 = f"blocks.{layer1}.hook_attn_in"
        activations1 = lorsas[layer1].encode(cache_original[input_hook1])
        hook_point1 = f"blocks.{layer1}.hook_attn_out"
    else:
        raise ValueError("first_steering feature_type must be 'transcoder' or 'lorsa'")
    
    activation_value1 = activations1[0, pos1, feature_id1]
    if activation_value1 is None:
        return None
    
    # calculate the injection value of first steering
    dec1 = get_feature_vector(lorsas, transcoders, ft1, layer1, feature_id1)
    feature_contribution1 = activation_value1 * dec1
    inject_val1 = (steering_scale1 - 1.0) * feature_contribution1
    
    def _steer1(act, hook):
        out = act.clone()
        delta = inject_val1.to(out.device)
        out[(slice(None), pos1) if out.dim() == 3 else (pos1,)] += delta
        return out
    
    # add the hook of first steering
    model.add_hook(hook_point1, _steer1)
    
    # second step: forward on the basis of first steering, get the basic data of second steering
    first_steered_output, cache_first_steered = model.run_with_cache(fen, prepend_bos=False)
    logits_after_first, value_after_first = _get_logits_and_value(first_steered_output)
    
    # prepare second steering (based on the cache after first steering)
    ft2 = second_steering['feature_type']
    layer2 = second_steering['layer']
    pos2 = second_steering['pos']
    feature_id2 = second_steering['feature_id']
    steering_scale2 = second_steering['steering_scale']
    
    if ft2 == "transcoder":
        input_hook2 = f"blocks.{layer2}.resid_mid_after_ln"
        activations2 = transcoders[layer2].encode(cache_first_steered[input_hook2])
        hook_point2 = f"blocks.{layer2}.hook_mlp_out"
    elif ft2 == "lorsa":
        input_hook2 = f"blocks.{layer2}.hook_attn_in"
        activations2 = lorsas[layer2].encode(cache_first_steered[input_hook2])
        hook_point2 = f"blocks.{layer2}.hook_attn_out"
    else:
        model.reset_hooks()
        raise ValueError("second_steering feature_type must be 'transcoder' or 'lorsa'")
    
    activation_value2 = activations2[0, pos2, feature_id2]
    if activation_value2 is None:
        model.reset_hooks()
        return None
    
    # calculate the injection value of second steering
    dec2 = get_feature_vector(lorsas, transcoders, ft2, layer2, feature_id2)
    feature_contribution2 = activation_value2 * dec2
    inject_val2 = (steering_scale2 - 1.0) * feature_contribution2
    
    def _steer2(act, hook):
        out = act.clone()
        delta = inject_val2.to(out.device)
        out[(slice(None), pos2) if out.dim() == 3 else (pos2,)] += delta
        return out
    
    # add the hook of second steering
    model.add_hook(hook_point2, _steer2)
    
    # third step: forward on the basis of two steerings
    second_steered_output, _ = model.run_with_cache(fen, prepend_bos=False)
    model.reset_hooks()
    
    logits_after_second, value_after_second = _get_logits_and_value(second_steered_output)
    
    return {
        # first steering information
        "first_steering": {
            "feature_type": ft1,
            "layer": layer1,
            "pos": pos1,
            "feature_id": feature_id1,
            "steering_scale": steering_scale1,
            "activation_value": float(activation_value1),
            "hook_point": hook_point1,
        },
        # second steering information
        "second_steering": {
            "feature_type": ft2,
            "layer": layer2,
            "pos": pos2,
            "feature_id": feature_id2,
            "steering_scale": steering_scale2,
            "activation_value": float(activation_value2),  # based on the activation value after first steering
            "hook_point": hook_point2,
        },
        # logits of three forward
        "logits_original": logits_original.detach().cpu(),
        "logits_after_first_steering": logits_after_first.detach().cpu(),
        "logits_after_second_steering": logits_after_second.detach().cpu(),
        # logits difference
        "logits_diff_first": (logits_after_first - logits_original).detach().cpu(),
        "logits_diff_second": (logits_after_second - logits_after_first).detach().cpu(),
        "logits_diff_total": (logits_after_second - logits_original).detach().cpu(),
        # value output
        "original_value": original_value,
        "value_after_first": value_after_first,
        "value_after_second": value_after_second,
        "value_diff_first": value_after_first - original_value,
        "value_diff_second": value_after_second - value_after_first,
        "value_diff_total": value_after_second - original_value,
    }


def analyze_features_after_first_steering(
    model,
    transcoders,
    lorsas,
    first_steering: Dict[str, Any],
    second_position_name: str,
    pos_dict: Dict[str, int],
    fen: str,
    moves_tracing: Dict[str, str],
    feature_types: List[str] = ["transcoder", "lorsa"],
    activation_threshold: float = 0.1,
    steering_scale_second: float = 2.0,
    max_features_per_type: Optional[int] = None,
    max_steering_features: Optional[int] = None,
) -> Dict[str, Any]:
    """
    analyze the features at a position that become important after first steering (measured by second steering).
    
    Args:
        model: HookedTransformer model
        transcoders: dictionary, layer -> Transcoder SAE
        lorsas: list of Lorsa models (in layer order)
        first_steering: first steering parameters dictionary
        second_position_name: name of the second position to analyze
        pos_dict: dictionary of position names to indices
        fen: FEN string
        moves_tracing: dictionary of moves to track {move_name: move_uci}
        feature_types: list of feature types to analyze
        activation_threshold: activation threshold
        steering_scale_second: steering scale for second steering
        max_features_per_type: maximum number of features per type
        max_steering_features: maximum number of steering features to analyze
    
    Returns:
        dictionary containing analysis results
    """
    if second_position_name not in pos_dict:
        raise ValueError(f"Unknown position_name: {second_position_name}")
    
    second_pos_idx = pos_dict[second_position_name]
    
    # first step: collect the activated features at the second position on the basis of first steering
    model.reset_hooks()
    
    # prepare the hook of first steering
    ft1 = first_steering['feature_type']
    layer1 = first_steering['layer']
    pos1 = first_steering['pos']
    feature_id1 = first_steering['feature_id']
    steering_scale1 = first_steering['steering_scale']
    
    # get the original cache
    _, cache_original = model.run_with_cache(fen, prepend_bos=False)
    
    if ft1 == "transcoder":
        input_hook1 = f"blocks.{layer1}.resid_mid_after_ln"
        activations1 = transcoders[layer1].encode(cache_original[input_hook1])
        hook_point1 = f"blocks.{layer1}.hook_mlp_out"
    elif ft1 == "lorsa":
        input_hook1 = f"blocks.{layer1}.hook_attn_in"
        activations1 = lorsas[layer1].encode(cache_original[input_hook1])
        hook_point1 = f"blocks.{layer1}.hook_attn_out"
    else:
        raise ValueError("first_steering feature_type must be 'transcoder' or 'lorsa'")
    
    activation_value1 = activations1[0, pos1, feature_id1]
    if activation_value1 is None:
        return {
            "error": "First steering feature not found",
            "results": []
        }
    
    dec1 = get_feature_vector(lorsas, transcoders, ft1, layer1, feature_id1)
    feature_contribution1 = activation_value1 * dec1
    inject_val1 = (steering_scale1 - 1.0) * feature_contribution1
    
    def _steer1(act, hook):
        out = act.clone()
        delta = inject_val1.to(out.device)
        out[(slice(None), pos1) if out.dim() == 3 else (pos1,)] += delta
        return out
    
    model.add_hook(hook_point1, _steer1)
    
    # get the cache on the basis of first steering
    _, cache_first_steered = model.run_with_cache(fen, prepend_bos=False)
    
    # collect the activated features at the second position (based on the cache after first steering)
    second_position_features = []
    n_layers = max(len(transcoders), len(lorsas))
    
    for layer in range(n_layers):
        for feature_type in feature_types:
            if feature_type == "transcoder":
                if layer >= len(transcoders):
                    continue
                sae = transcoders[layer]
                input_hook = f"blocks.{layer}.resid_mid_after_ln"
            elif feature_type == "lorsa":
                if layer >= len(lorsas):
                    continue
                sae = lorsas[layer]
                input_hook = f"blocks.{layer}.hook_attn_in"
            else:
                continue
            
            if input_hook not in cache_first_steered:
                continue
            
            activations = sae.encode(cache_first_steered[input_hook])
            pos_acts = activations[0, second_pos_idx]  # [n_features]
            
            mask = pos_acts > activation_threshold
            if not mask.any():
                continue
            
            feature_ids = torch.where(mask)[0]
            values = pos_acts[mask]
            
            pairs = sorted(
                zip(feature_ids.tolist(), values.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
            
            if max_features_per_type is not None:
                pairs = pairs[:max_features_per_type]
            
            for feature_id, activation_value in pairs:
                second_position_features.append({
                    "layer": layer,
                    "feature_id": feature_id,
                    "activation_value": float(activation_value),
                    "feature_type": feature_type,
                    "position_name": second_position_name,
                    "position_idx": second_pos_idx,
                })
    
    model.reset_hooks()
    
    second_position_features.sort(key=lambda x: x["activation_value"], reverse=True)
    
    # limit the number of features to analyze
    if max_steering_features is not None:
        second_position_features = second_position_features[:max_steering_features]
    
    # second step: analyze each feature at the second position with nested steering
    results = []
    
    for feature_info in tqdm(second_position_features, desc=f"Nested steering {second_position_name}", unit="feature"):
        second_steering = {
            'feature_type': feature_info['feature_type'],
            'layer': feature_info['layer'],
            'pos': feature_info['position_idx'],
            'feature_id': feature_info['feature_id'],
            'steering_scale': steering_scale_second,
        }
        
        try:
            nested_result = nested_activation_steering_effect(
                model=model,
                transcoders=transcoders,
                lorsas=lorsas,
                first_steering=first_steering,
                second_steering=second_steering,
                fen=fen,
                get_value=True
            )
            
            if nested_result:
                # calculate the change of move probabilities (based on the logits before and after second steering)
                move_probabilities = {}
                
                logits_after_first = torch.tensor(nested_result['logits_after_first_steering'])
                logits_after_second = torch.tensor(nested_result['logits_after_second_steering'])
                
                for move_name, move_uci in moves_tracing.items():
                    try:
                        prob_after_first = get_move_from_policy_output_with_prob(
                            logits_after_first, fen, move_uci=move_uci
                        )
                        prob_after_second = get_move_from_policy_output_with_prob(
                            logits_after_second, fen, move_uci=move_uci
                        )
                        
                        prob_diff = None
                        if prob_after_first is not None and prob_after_second is not None:
                            orig_val = prob_after_first if isinstance(prob_after_first, (int, float)) else 0.0
                            mod_val = prob_after_second if isinstance(prob_after_second, (int, float)) else 0.0
                            prob_diff = float(mod_val) - float(orig_val)
                        
                        move_probabilities[move_name] = {
                            'uci': move_uci,
                            'prob_after_first_steering': prob_after_first,
                            'prob_after_second_steering': prob_after_second,
                            'prob_diff': prob_diff
                        }
                    except Exception:
                        move_probabilities[move_name] = {
                            'uci': move_uci,
                            'prob_after_first_steering': None,
                            'prob_after_second_steering': None,
                            'prob_diff': None
                        }
                
                result = {
                    'position_name': second_position_name,
                    'position_idx': second_pos_idx,
                    'feature_type': feature_info['feature_type'],
                    'layer': feature_info['layer'],
                    'feature_id': feature_info['feature_id'],
                    'activation_value': feature_info['activation_value'],
                    'steering_scale_second': steering_scale_second,
                    'move_probabilities': move_probabilities,
                    'value_after_first': nested_result['value_after_first'],
                    'value_after_second': nested_result['value_after_second'],
                    'value_diff_second': nested_result['value_diff_second'],
                }
                
                results.append(result)
        
        except Exception:
            # Skip failed features
            pass
    
    return {
        'first_steering': first_steering,
        'second_position_name': second_position_name,
        'total_features_found': len(second_position_features),
        'total_features_analyzed': len(results),
        'steering_scale_second': steering_scale_second,
        'activation_threshold': activation_threshold,
        'moves_tracing': moves_tracing,
        'results': results
    }


def analyze_position_features_comprehensive(
    pos_dict: Dict[str, int],
    position_name: str,
    model,
    transcoders,
    lorsas,
    fen: str,
    moves_tracing: Dict[str, str],
    feature_types: Optional[List[str]] = None,
    steering_scale: float = 2.0,
    activation_threshold: float = 0.1,
    max_features_per_type: Optional[int] = None,
    max_steering_features: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Comprehensive analysis of all activated features at a given position with steering effects.

    Args:
        pos_dict: Dictionary mapping position names to position indices
        position_name: Name of the position to analyze
        model: HookedTransformer model
        transcoders: Dictionary of transcoder SAEs by layer
        lorsas: List of LORSA models by layer
        fen: FEN string of the board position
        moves_tracing: Dictionary mapping move names to UCI strings
        feature_types: List of feature types to analyze
        steering_scale: Steering scale factor
        activation_threshold: Activation threshold for feature detection
        max_features_per_type: Maximum features per type to collect
        max_steering_features: Maximum features to analyze with steering

    Returns:
        Dictionary containing comprehensive analysis results
    """
    if feature_types is None:
        feature_types = ["transcoder", "lorsa"]

    # Collect all activated features at this position
    all_features = collect_activated_features_at_position(
        pos_dict=pos_dict,
        position_name=position_name,
        model=model,
        transcoders=transcoders,
        lorsas=lorsas,
        fen=fen,
        feature_types=feature_types,
        activation_threshold=activation_threshold,
        max_features_per_type=max_features_per_type
    )

    # Group by feature type for statistics
    type_counts = defaultdict(int)
    for feature in all_features:
        type_counts[feature['feature_type']] += 1

    # Run steering analysis
    steering_results = run_steering_for_position_features(
        position_name=position_name,
        features=all_features,
        model=model,
        transcoders=transcoders,
        lorsas=lorsas,
        fen=fen,
        moves_tracing=moves_tracing,
        steering_scale=steering_scale,
        max_features=max_steering_features
    )

    analysis_result = {
        'position_name': position_name,
        'total_features_found': len(all_features),
        'total_features_analyzed': len(steering_results),
        'feature_types_analyzed': feature_types,
        'feature_type_counts': dict(type_counts),
        'steering_scale': steering_scale,
        'activation_threshold': activation_threshold,
        'moves_tracing': moves_tracing,
        'results': steering_results
    }

    return analysis_result


def multi_feature_steering_effect(
    model,
    transcoders,
    lorsas,
    steering_configs: List[Tuple[int, int, int, str, float]],
    fen: str,
    get_value: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    steering multiple features simultaneously, each feature can have a different steering_scale.
    
    Args:
        model: HookedTransformer model
        transcoders: dictionary, layer -> Transcoder SAE
        lorsas: list of Lorsa models (in layer order)
        steering_configs: list of steering configurations, each configuration is a tuple:
            (layer: int, pos: int, feature_id: int, feature_type: str, steering_scale: float)
            feature_type must be 'transcoder' or 'lorsa'
        fen: FEN string
        get_value: whether to get value output
    
    Returns:
        dictionary containing multi-feature steering results, or None (if failed)
    """
    if not steering_configs:
        raise ValueError("steering_configs cannot be empty")
    
    model.reset_hooks()
    
    def _get_logits_and_value(output):
        logits = output[0]
        if logits.ndim == 2:
            logits = logits[0]
        value = (
            float(output[1][0][0] - output[1][0][2])
            if get_value else 0.0
        )
        return logits, value
    
    original_output, original_cache = model.run_with_cache(fen, prepend_bos=False)
    logits_original, original_value = _get_logits_and_value(original_output)
    
    steering_info_list = []
    hook_functions = {}
    
    for i, config in enumerate(steering_configs):
        if len(config) != 5:
            raise ValueError(f"Each steering config must be a tuple of 5 elements: (layer, pos, feature_id, feature_type, steering_scale), got {config}")
        
        layer, pos, feature_id, feature_type, steering_scale = config
        
        if feature_type == "transcoder":
            input_hook = f"blocks.{layer}.resid_mid_after_ln"
            activations = transcoders[layer].encode(original_cache[input_hook])
            hook_point = f"blocks.{layer}.hook_mlp_out"
        elif feature_type == "lorsa":
            input_hook = f"blocks.{layer}.hook_attn_in"
            activations = lorsas[layer].encode(original_cache[input_hook])
            hook_point = f"blocks.{layer}.hook_attn_out"
        else:
            raise ValueError(f"feature_type must be 'transcoder' or 'lorsa', got {feature_type}")
        
        activation_value = activations[0, pos, feature_id]
        if activation_value is None:
            model.reset_hooks()
            return None
        
        dec = get_feature_vector(lorsas, transcoders, feature_type, layer, feature_id)
        feature_contribution = activation_value * dec
        inject_val = (steering_scale - 1.0) * feature_contribution
        
        steering_info_list.append({
            "index": i,
            "feature_type": feature_type,
            "layer": layer,
            "pos": pos,
            "feature_id": feature_id,
            "steering_scale": steering_scale,
            "activation_value": float(activation_value),
            "hook_point": hook_point,
            "inject_val": inject_val,
        })
        
        if hook_point not in hook_functions:
            hook_functions[hook_point] = []
        
        hook_functions[hook_point].append({
            "pos": pos,
            "inject_val": inject_val,
        })
    
    def _create_steer_function(hook_point, inject_configs):
        def _steer(act, hook):
            out = act.clone()
            for config in inject_configs:
                pos = config["pos"]
                delta = config["inject_val"].to(out.device)
                out[(slice(None), pos) if out.dim() == 3 else (pos,)] += delta
            return out
        return _steer
    
    for hook_point, inject_configs in hook_functions.items():
        steer_func = _create_steer_function(hook_point, inject_configs)
        model.add_hook(hook_point, steer_func)
    
    # steered forward
    modified_output, modified_cache = model.run_with_cache(fen, prepend_bos=False)
    model.reset_hooks()
    
    logits_modified, modified_value = _get_logits_and_value(modified_output)
    
    return {
        "steering_configs": steering_info_list,
        "num_features": len(steering_configs),
        
        "logits_original": logits_original.detach().cpu(),
        "logits_modified": logits_modified.detach().cpu(),
        "logits_diff": (logits_modified - logits_original).detach().cpu(),
        
        "original_value": original_value,
        "modified_value": modified_value,
        "value_diff": modified_value - original_value,
        
        "original_cache": original_cache,
        "modified_cache": modified_cache,
        "original_output": original_output,
        "modified_output": modified_output,
    }
