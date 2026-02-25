"""
Graph Feature Diffing Service
compare the activation differences between two FENs, find the inactive nodes in the perturbed FEN
"""

import json
import torch
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

def parse_node_id(node_id: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[str]]:
    parts = node_id.split('_')
    if len(parts) >= 3:
        try:
            encoded_layer = int(parts[0])
            feature_idx = int(parts[1])
            position = int(parts[2])
            
            # decode layer: layer = encoded_layer // 2
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
    find the inactive nodes in the perturbed FEN
    
    Args:
        graph_json: loaded JSON graph data
        lorsa_feature_acts_perturbed: list of tensors, shape [batch, seq_len, n_features]
        tc_feature_acts_perturbed: list of tensors, shape [batch, seq_len, n_features]
        activation_threshold: activation threshold, below which is considered inactive
        debug: whether to print debug information
    
    Returns:
        list of inactive nodes, each node contains the original node information and the reason for inactivity
    """
    nodes = graph_json.get('nodes', [])
    inactive_nodes = []
    skipped_nodes = []
    
    for node in nodes:
        node_id = node.get('node_id', '')
        feature_type_raw = node.get('feature_type', '')
        feature_type = feature_type_raw.lower()
        
        # skip logit and error nodes
        if 'logit' in feature_type or 'error' in feature_type:
            if debug:
                skipped_nodes.append({'node_id': node_id, 'reason': 'logit_or_error'})
            continue
        
        # parse node_id to get layer, feature_idx, position, feature_type
        parsed_layer, parsed_feat_idx, parsed_pos, parsed_feat_type = parse_node_id(node_id)
        
        # must be able to parse node_id, otherwise skip
        if parsed_layer is None:
            if debug:
                skipped_nodes.append({'node_id': node_id, 'reason': 'cannot_parse_node_id'})
            continue
        
        # use the parsed values
        layer = parsed_layer
        feature_idx = parsed_feat_idx
        position = parsed_pos
        feat_type = parsed_feat_type
        
        # verify if feature_type matches
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
        
        # check if the node is activated in the perturbed FEN
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
        print(f"\nnumber of skipped nodes: {len(skipped_nodes)}")
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
    compare the activation differences between two FENs
    
    Args:
        graph_json: original graph JSON data
        original_fen: original FEN string
        perturbed_fen: perturbed FEN string
        model_name: model name
        transcoders: transcoders dictionary
        lorsas: lorsas list
        replacement_model: replacement model instance
        activation_threshold: activation threshold
        n_layers: number of layers
    
    Returns:
        dictionary containing inactive nodes
    """
    # run perturbed FEN and get activations
    output_perturbed, cache_perturbed = replacement_model.run_with_cache(perturbed_fen, prepend_bos=False)
    
    # get activation tensors
    lorsa_feature_acts_perturbed = []
    tc_feature_acts_perturbed = []
    
    for layer in range(n_layers):
        lorsa_input = cache_perturbed[f'blocks.{layer}.hook_attn_in']
        lorsa_feature_act = lorsas[layer].encode(lorsa_input)
        lorsa_feature_acts_perturbed.append(lorsa_feature_act)
        
        tc_input = cache_perturbed[f'blocks.{layer}.resid_mid_after_ln']
        tc_feature_act = transcoders[layer].encode(tc_input)
        tc_feature_acts_perturbed.append(tc_feature_act)
    
    # find inactive nodes
    inactive_nodes = find_inactive_nodes_in_perturbed(
        graph_json,
        lorsa_feature_acts_perturbed,
        tc_feature_acts_perturbed,
        activation_threshold=activation_threshold,
        debug=True
    )
    
    # group by layer and feature_type
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
