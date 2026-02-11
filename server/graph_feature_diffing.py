"""
Graph Feature Diffing Service
比较两个FEN的激活差异，找出在perturbed FEN中未激活的节点
"""

import json
import torch
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

def parse_node_id(node_id: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[str]]:
    """
    解析 node_id 格式: {encoded_layer}_{feature_idx}_{pos}
    
    注意：layer 是编码后的：
    - lorsa: encoded_layer = 2 * layer + 0 (偶数，如 28 -> layer 14)
    - tc: encoded_layer = 2 * layer + 1 (奇数，如 29 -> layer 14)
    
    返回: (original_layer, feature_idx, position, feature_type)
        feature_type: 'lorsa' 或 'tc' 或 None
    """
    parts = node_id.split('_')
    if len(parts) >= 3:
        try:
            encoded_layer = int(parts[0])
            feature_idx = int(parts[1])
            position = int(parts[2])
            
            # 解码 layer: layer = encoded_layer // 2
            original_layer = encoded_layer // 2
            is_lorsa = (encoded_layer % 2 == 0)
            feature_type = 'lorsa' if is_lorsa else 'tc'
            
            return original_layer, feature_idx, position, feature_type
        except ValueError:
            pass
    return None, None, None, None


def find_inactive_nodes_in_perturbed(
    graph_json: Dict[str, Any],
    lorsa_feature_acts_perturbed: List[torch.Tensor],
    tc_feature_acts_perturbed: List[torch.Tensor],
    activation_threshold: float = 0.0,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    找出在 perturbed FEN 中没有激活的节点
    
    Args:
        graph_json: 加载的 JSON 图数据
        lorsa_feature_acts_perturbed: list of tensors, shape [batch, seq_len, n_features]
        tc_feature_acts_perturbed: list of tensors, shape [batch, seq_len, n_features]
        activation_threshold: 激活阈值，低于此值视为未激活
        debug: 是否打印调试信息
    
    Returns:
        未激活的节点列表，每个节点包含原始节点信息和未激活原因
    """
    nodes = graph_json.get('nodes', [])
    inactive_nodes = []
    skipped_nodes = []
    
    for node in nodes:
        node_id = node.get('node_id', '')
        feature_type_raw = node.get('feature_type', '')
        feature_type = feature_type_raw.lower()
        
        # 跳过 logit 和 error 节点
        if 'logit' in feature_type or 'error' in feature_type:
            if debug:
                skipped_nodes.append({'node_id': node_id, 'reason': 'logit_or_error'})
            continue
        
        # 解析 node_id 获取 layer, feature_idx, position, feature_type
        parsed_layer, parsed_feat_idx, parsed_pos, parsed_feat_type = parse_node_id(node_id)
        
        # 必须能够解析 node_id，否则跳过
        if parsed_layer is None:
            if debug:
                skipped_nodes.append({'node_id': node_id, 'reason': 'cannot_parse_node_id'})
            continue
        
        # 使用解析出的值
        layer = parsed_layer
        feature_idx = parsed_feat_idx
        position = parsed_pos
        feat_type = parsed_feat_type
        
        # 验证 feature_type 是否匹配
        if feat_type == 'lorsa' and 'lorsa' not in feature_type:
            if debug:
                skipped_nodes.append({
                    'node_id': node_id, 
                    'reason': f'type_mismatch: parsed={feat_type}, node_type={feature_type}'
                })
            continue
        if feat_type == 'tc' and 'transcoder' not in feature_type and 'cross layer' not in feature_type:
            if debug:
                skipped_nodes.append({
                    'node_id': node_id, 
                    'reason': f'type_mismatch: parsed={feat_type}, node_type={feature_type}'
                })
            continue
        
        # 检查在 perturbed FEN 中是否激活
        try:
            if feat_type == 'lorsa':
                if layer < len(lorsa_feature_acts_perturbed):
                    activation_value = lorsa_feature_acts_perturbed[layer][0][position][feature_idx].item()
                    if activation_value <= activation_threshold:
                        inactive_nodes.append({
                            'node_id': node_id,
                            'layer': layer,
                            'feature_idx': feature_idx,
                            'position': position,
                            'feature_type': feature_type_raw,
                            'parsed_feature_type': feat_type,
                            'reason': f'activation={activation_value:.6f} <= threshold={activation_threshold}',
                            'perturbed_activation': activation_value,
                            'key': f'({layer}, {feature_idx}, {position}, {feat_type})',
                            'original_node': node
                        })
            elif feat_type == 'tc':
                if layer < len(tc_feature_acts_perturbed):
                    activation_value = tc_feature_acts_perturbed[layer][0][position][feature_idx].item()
                    if activation_value <= activation_threshold:
                        inactive_nodes.append({
                            'node_id': node_id,
                            'layer': layer,
                            'feature_idx': feature_idx,
                            'position': position,
                            'feature_type': feature_type_raw,
                            'parsed_feature_type': feat_type,
                            'reason': f'activation={activation_value:.6f} <= threshold={activation_threshold}',
                            'perturbed_activation': activation_value,
                            'key': f'({layer}, {feature_idx}, {position}, {feat_type})',
                            'original_node': node
                        })
        except (IndexError, RuntimeError) as e:
            if debug:
                skipped_nodes.append({
                    'node_id': node_id,
                    'reason': f'index_error: {str(e)}'
                })
            continue
    
    if debug and skipped_nodes:
        print(f"\n跳过的节点数量: {len(skipped_nodes)}")
        for skip in skipped_nodes[:10]:
            print(f"  - {skip['node_id']}: {skip['reason']}")
    
    return inactive_nodes


def compare_fen_activations(
    graph_json: Dict[str, Any],
    original_fen: str,
    perturbed_fen: str,
    model_name: str,
    transcoders: Dict[int, Any],
    lorsas: List[Any],
    replacement_model: Any,
    activation_threshold: float = 0.0,
    n_layers: int = 15
) -> Dict[str, Any]:
    """
    比较两个FEN的激活差异
    
    Args:
        graph_json: 原始图的JSON数据
        original_fen: 原始FEN字符串
        perturbed_fen: 扰动后的FEN字符串
        model_name: 模型名称
        transcoders: transcoders字典
        lorsas: lorsas列表
        replacement_model: replacement model实例
        activation_threshold: 激活阈值
        n_layers: 层数
    
    Returns:
        包含inactive nodes的字典
    """
    # 运行perturbed FEN并获取激活
    output_perturbed, cache_perturbed = replacement_model.run_with_cache(perturbed_fen, prepend_bos=False)
    
    # 获取激活张量
    lorsa_feature_acts_perturbed = []
    tc_feature_acts_perturbed = []
    
    for layer in range(n_layers):
        lorsa_input = cache_perturbed[f'blocks.{layer}.hook_attn_in']
        lorsa_feature_act = lorsas[layer].encode(lorsa_input)
        lorsa_feature_acts_perturbed.append(lorsa_feature_act)
        
        tc_input = cache_perturbed[f'blocks.{layer}.resid_mid_after_ln']
        tc_feature_act = transcoders[layer].encode(tc_input)
        tc_feature_acts_perturbed.append(tc_feature_act)
    
    # 找出未激活的节点
    inactive_nodes = find_inactive_nodes_in_perturbed(
        graph_json,
        lorsa_feature_acts_perturbed,
        tc_feature_acts_perturbed,
        activation_threshold=activation_threshold,
        debug=True
    )
    
    # 按 layer 和 feature_type 分组统计
    by_layer = defaultdict(int)
    by_type = defaultdict(int)
    for node in inactive_nodes:
        by_layer[node['layer']] += 1
        by_type[node['feature_type']] += 1
    
    return {
        'original_fen': original_fen,
        'perturbed_fen': perturbed_fen,
        'total_nodes': len(graph_json.get('nodes', [])),
        'inactive_nodes_count': len(inactive_nodes),
        'inactive_nodes': inactive_nodes,
        'statistics': {
            'by_layer': dict(by_layer),
            'by_type': dict(by_type)
        }
    }
