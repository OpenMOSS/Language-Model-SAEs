
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
    first_steering: Dict[str, Any],  # 第一次steering的参数: feature_type, layer, pos, feature_id, steering_scale
    second_steering: Dict[str, Any],  # 第二次steering的参数: feature_type, layer, pos, feature_id, steering_scale
    fen: str,
    get_value: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    嵌套steering：在第一次steering的基础上，进行第二次steering，分析第二次steering的影响。
    
    Args:
        model: HookedTransformer模型
        transcoders: 字典，layer -> Transcoder SAE
        lorsas: Lorsa模型列表（按layer顺序）
        first_steering: 第一次steering的参数字典，包含:
            - feature_type: str ('transcoder' 或 'lorsa')
            - layer: int
            - pos: int
            - feature_id: int
            - steering_scale: float
        second_steering: 第二次steering的参数字典，包含:
            - feature_type: str ('transcoder' 或 'lorsa')
            - layer: int
            - pos: int
            - feature_id: int
            - steering_scale: float
        fen: FEN字符串
        get_value: 是否获取value输出
    
    Returns:
        包含嵌套steering结果的字典，或None（如果失败）
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
    
    # 第一步：原始forward，获取第一次steering的基础数据
    original_output, cache_original = model.run_with_cache(fen, prepend_bos=False)
    logits_original, original_value = _get_logits_and_value(original_output)
    
    # 准备第一次steering
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
    
    # 计算第一次steering的注入值
    dec1 = get_feature_vector(lorsas, transcoders, ft1, layer1, feature_id1)
    feature_contribution1 = activation_value1 * dec1
    inject_val1 = (steering_scale1 - 1.0) * feature_contribution1
    
    def _steer1(act, hook):
        out = act.clone()
        delta = inject_val1.to(out.device)
        out[(slice(None), pos1) if out.dim() == 3 else (pos1,)] += delta
        return out
    
    # 添加第一次steering的hook
    model.add_hook(hook_point1, _steer1)
    
    # 第二步：在第一次steering的基础上进行forward，获取第二次steering的基础数据
    first_steered_output, cache_first_steered = model.run_with_cache(fen, prepend_bos=False)
    logits_after_first, value_after_first = _get_logits_and_value(first_steered_output)
    
    # 准备第二次steering（基于第一次steering后的cache）
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
    
    # 计算第二次steering的注入值（基于第一次steering后的激活值）
    dec2 = get_feature_vector(lorsas, transcoders, ft2, layer2, feature_id2)
    feature_contribution2 = activation_value2 * dec2
    inject_val2 = (steering_scale2 - 1.0) * feature_contribution2
    
    def _steer2(act, hook):
        out = act.clone()
        delta = inject_val2.to(out.device)
        out[(slice(None), pos2) if out.dim() == 3 else (pos2,)] += delta
        return out
    
    # 添加第二次steering的hook（在第一次steering的hook之后）
    model.add_hook(hook_point2, _steer2)
    
    # 第三步：在两次steering的基础上进行forward
    second_steered_output, _ = model.run_with_cache(fen, prepend_bos=False)
    model.reset_hooks()
    
    logits_after_second, value_after_second = _get_logits_and_value(second_steered_output)
    
    return {
        # 第一次steering信息
        "first_steering": {
            "feature_type": ft1,
            "layer": layer1,
            "pos": pos1,
            "feature_id": feature_id1,
            "steering_scale": steering_scale1,
            "activation_value": float(activation_value1),
            "hook_point": hook_point1,
        },
        # 第二次steering信息
        "second_steering": {
            "feature_type": ft2,
            "layer": layer2,
            "pos": pos2,
            "feature_id": feature_id2,
            "steering_scale": steering_scale2,
            "activation_value": float(activation_value2),  # 基于第一次steering后的激活值
            "hook_point": hook_point2,
        },
        # 三次forward的logits
        "logits_original": logits_original.detach().cpu(),
        "logits_after_first_steering": logits_after_first.detach().cpu(),
        "logits_after_second_steering": logits_after_second.detach().cpu(),
        # logits差异
        "logits_diff_first": (logits_after_first - logits_original).detach().cpu(),
        "logits_diff_second": (logits_after_second - logits_after_first).detach().cpu(),
        "logits_diff_total": (logits_after_second - logits_original).detach().cpu(),
        # value输出
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
    分析在第一次steering后，某个位置的features变得重要（通过第二次steering测量）。
    
    Args:
        model: HookedTransformer模型
        transcoders: 字典，layer -> Transcoder SAE
        lorsas: Lorsa模型列表（按layer顺序）
        first_steering: 第一次steering的参数字典
        second_position_name: 要分析的第二个位置名称
        pos_dict: 位置名称到索引的字典
        fen: FEN字符串
        moves_tracing: 要追踪的moves字典 {move_name: move_uci}
        feature_types: 要分析的feature类型列表
        activation_threshold: 激活阈值
        steering_scale_second: 第二次steering的放大系数
        max_features_per_type: 每种类型最多收集的features数量
        max_steering_features: 最多分析的steering features数量
    
    Returns:
        包含分析结果的字典
    """
    if second_position_name not in pos_dict:
        raise ValueError(f"Unknown position_name: {second_position_name}")
    
    second_pos_idx = pos_dict[second_position_name]
    
    # 第一步：在第一次steering的基础上，收集第二个位置的激活features
    model.reset_hooks()
    
    # 准备第一次steering的hook
    ft1 = first_steering['feature_type']
    layer1 = first_steering['layer']
    pos1 = first_steering['pos']
    feature_id1 = first_steering['feature_id']
    steering_scale1 = first_steering['steering_scale']
    
    # 获取原始cache
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
    
    # 在第一次steering的基础上获取cache
    _, cache_first_steered = model.run_with_cache(fen, prepend_bos=False)
    
    # 收集第二个位置的激活features（基于第一次steering后的cache）
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
    
    # 限制要分析的features数量
    if max_steering_features is not None:
        second_position_features = second_position_features[:max_steering_features]
    
    # 第二步：对第二个位置的每个feature进行嵌套steering分析
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
                # 计算move概率变化（基于第二次steering前后的logits）
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
                    'activation_value': feature_info['activation_value'],  # 基于第一次steering后的激活值
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
    steering_configs: List[Tuple[int, int, int, str, float]],  # 每个配置是 (layer, pos, feature_id, feature_type, steering_scale)
    fen: str,
    get_value: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    同时对多个feature进行steering，每个feature可以有不同的steering_scale。
    
    Args:
        model: HookedTransformer模型
        transcoders: 字典，layer -> Transcoder SAE
        lorsas: Lorsa模型列表（按layer顺序）
        steering_configs: steering配置列表，每个配置是tuple:
            (layer: int, pos: int, feature_id: int, feature_type: str, steering_scale: float)
            feature_type 必须是 'transcoder' 或 'lorsa'
        fen: FEN字符串
        get_value: 是否获取value输出
    
    Returns:
        包含多feature steering结果的字典，或None（如果失败）
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
    
    # 原始forward
    original_output, original_cache = model.run_with_cache(fen, prepend_bos=False)
    logits_original, original_value = _get_logits_and_value(original_output)
    
    # 准备所有steering的hook
    steering_info_list = []
    hook_functions = {}
    
    for i, config in enumerate(steering_configs):
        # 解析tuple: (layer, pos, feature_id, feature_type, steering_scale)
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
        
        # 计算steering的注入值
        dec = get_feature_vector(lorsas, transcoders, feature_type, layer, feature_id)
        feature_contribution = activation_value * dec
        inject_val = (steering_scale - 1.0) * feature_contribution
        
        # 存储steering信息
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
        
        # 为每个hook_point创建或更新hook函数
        if hook_point not in hook_functions:
            hook_functions[hook_point] = []
        
        hook_functions[hook_point].append({
            "pos": pos,
            "inject_val": inject_val,
        })
    
    # 创建hook函数：对于每个hook_point，累加所有在该位置的steering delta
    def _create_steer_function(hook_point, inject_configs):
        def _steer(act, hook):
            out = act.clone()
            for config in inject_configs:
                pos = config["pos"]
                delta = config["inject_val"].to(out.device)
                out[(slice(None), pos) if out.dim() == 3 else (pos,)] += delta
            return out
        return _steer
    
    # 添加所有hooks
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
