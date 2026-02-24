from typing import Optional, Dict, Any, Tuple, List, Set, Literal, Union
import torch
from dataclasses import dataclass


@dataclass
class Node:
    """Represents a feature node in the model."""
    raw_id: str
    first: int
    feature: int
    pos: int
    feature_type: Literal["lorsa", "transcoder"]
    layer: int

    @classmethod
    def from_id(cls, id_str: str):
        """Create Node from ID string format: '<first>_<feature>_<pos>'."""
        parts = id_str.split("_")
        if len(parts) != 3:
            raise ValueError(f"Invalid id format: {id_str}, expected '<num>_<feature>_<pos>'")

        first, feature, pos = map(int, parts)
        feature_type = "lorsa" if first % 2 == 0 else "transcoder"
        layer = first // 2

        return cls(
            raw_id=id_str,
            first=first,
            feature=feature,
            pos=pos,
            feature_type=feature_type,
            layer=layer,
        )
        
    def default_hook_name(self) -> str:
        """Return default hook name based on node type."""
        if self.feature_type == "lorsa":
            return f"blocks.{self.layer}.hook_attn_out"
        elif self.feature_type == "transcoder":
            return f"blocks.{self.layer}.hook_mlp_out"
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

    def __repr__(self):
        return (f"Node(id={self.raw_id}, type={self.feature_type}, "
                f"layer={self.layer}, feature={self.feature}, pos={self.pos})")


def parse_node_list(id_list: List[str]) -> List[Node]:
    """Parse list of ID strings into Node instances."""
    return [Node.from_id(id_str) for id_str in id_list]


def _normalize_nodes(nodes: List[Union[str, dict, "Node", Tuple[int, int, int, str]]]) -> List["Node"]:
    """Normalize different node input formats into Node objects."""
    norm: List[Node] = []
    for n in nodes:
        if isinstance(n, Node):
            norm.append(n)
        elif isinstance(n, str):
            norm.append(Node.from_id(n))
        elif isinstance(n, tuple) and len(n) == 4:
            # Support tuple format: (layer, pos, feature, feature_type)
            layer, pos, feature, feature_type = n
            ft_lower = feature_type.lower()
            if ft_lower not in ("lorsa", "transcoder"):
                raise ValueError(f"Invalid feature_type: {feature_type!r}. Expected 'lorsa' or 'transcoder'.")
            first = layer * 2 + (0 if ft_lower == "lorsa" else 1)
            raw_id = f"{first}_{feature}_{pos}"
            norm.append(Node(
                raw_id=raw_id,
                first=first,
                feature=feature,
                pos=pos,
                feature_type=ft_lower,  # type: ignore
                layer=layer,
            ))
        elif isinstance(n, dict):
            # Support raw_id directly
            if "raw_id" in n:
                norm.append(Node.from_id(n["raw_id"]))
            # Support feature_type + layer + feature + pos
            elif {"feature", "pos", "feature_type", "layer"}.issubset(n.keys()):
                ft = str(n["feature_type"]).lower()
                if ft not in ("lorsa", "transcoder"):
                    raise ValueError(f"Invalid feature_type: {n['feature_type']!r}. Expected 'lorsa' or 'transcoder'.")
                layer = int(n["layer"])
                feature = int(n["feature"])
                pos = int(n["pos"])
                first = layer * 2 + (0 if ft == "lorsa" else 1)
                norm.append(Node.from_id(f"{first}_{feature}_{pos}"))
            # Support first + feature + pos
            elif {"feature", "pos", "first"}.issubset(n.keys()):
                first = int(n["first"])
                feature = int(n["feature"])
                pos = int(n["pos"])
                norm.append(Node.from_id(f"{first}_{feature}_{pos}"))
            else:
                raise ValueError(
                    f"Unsupported node dict format: {n!r}. "
                    "Expected one of: {'raw_id'}, or {'feature_type','layer','feature','pos'}, or {'first','feature','pos'}"
                )
        else:
            raise ValueError(f"Unsupported node type: {type(n)}")
    return norm


def analyze_node_activation_impact(
    steering_nodes: List[Union[str, dict, "Node", Tuple[int, int, int, str]]],
    target_node: Union[str, dict, "Node", Tuple[int, int, int, str]],
    steering_scale: float,
    cache,
    model,
    tc_activations: List[torch.Tensor],
    lorsa_activations: List[torch.Tensor],
    tc_WDs: Optional[List[Optional[torch.Tensor]]] = None,
    lorsa_WDs: Optional[List[Optional[torch.Tensor]]] = None,
    transcoders: Optional[Any] = None,
    lorsas: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Analyze how steering source nodes affects a target node's activation.
    
    This function implements a causal intervention: it steers the activation of source nodes
    and measures how this affects the activation of a target node, revealing feature
    interactions in the model's latent space.

    Args:
        steering_nodes: List of nodes to steer (can be Node objects, strings, dicts, or tuples)
        target_node: Target node to measure activation change for (can be Node object, string, dict, or tuple)
        steering_scale: Steering scale factor (typically -1 or positive values)
        cache: Model cache containing intermediate activations
        model: HookedTransformer model instance
        tc_activations: List of transcoder activations per layer (sparse COO tensors)
        lorsa_activations: List of LORSA activations per layer (sparse COO tensors)
        tc_WDs: Optional transcoder decoder weights per layer
        lorsa_WDs: Optional LORSA decoder weights per layer

    Returns:
        Dictionary containing:
        - target_node: String representation of target node
        - original_activation: Original activation value of target node
        - modified_activation: Activation value after steering
        - activation_ratio: Ratio of modified/original activation
        - activation_change: Absolute change in activation
        - steering_scale: Steering scale used
        - steering_nodes_count: Number of steering nodes applied
        - steering_details: Details about each steering node application
    """
    # Get the device from the model
    model_device = next(model.parameters()).device

    # Normalize input nodes
    steering_node_list = _normalize_nodes(steering_nodes)
    target_node_obj = _normalize_nodes([target_node])[0]
    
    # Function to extract activation value from sparse tensor
    def get_activation_value(node: Node, activations_dict: Dict[str, List[torch.Tensor]]) -> float:
        ft = node.feature_type.lower()
        layer = int(node.layer)
        pos = int(node.pos)
        feature = int(node.feature)
        
        if ft == "transcoder":
            activations = activations_dict["tc"][layer]
        elif ft == "lorsa":
            activations = activations_dict["lorsa"][layer]
        else:
            raise ValueError(f"Unknown feature_type: {ft}")
        
        # Find activation at (batch=0, pos, feature)
        target_indices = torch.tensor([0, pos, feature], device=activations.indices().device)
        matches = (activations.indices() == target_indices.unsqueeze(1)).all(dim=0)
        
        if matches.any():
            return activations.values()[matches].item()
        else:
            return 0.0
    
    # Get baseline activations
    activations_dict = {
        "tc": tc_activations,
        "lorsa": lorsa_activations
    }
    
    # Get original activation of target node
    original_activation = get_activation_value(target_node_obj, activations_dict)
    
    # Reset any existing hooks
    try:
        model.reset_hooks()
    except Exception:
        pass

    # Get original forward pass
    original_output, _ = model.run_with_cache(
        cache["embed.hook_input"],
        prepend_bos=False,
    )

    # Group steering contributions by hook name
    updates_by_hook: Dict[str, List[Tuple[int, torch.Tensor]]] = {}
    details: List[Dict[str, Any]] = []

    for node in steering_node_list:
        ft = node.feature_type.lower()
        layer = int(node.layer)
        pos = int(node.pos)
        feature = int(node.feature)

        if ft == "transcoder":
            activations = tc_activations[layer]
            if tc_WDs is None:
                raise ValueError("tc_WDs required for transcoder steering")
            WDs = tc_WDs[layer]
            hook_name = f"blocks.{layer}.hook_mlp_out"
        elif ft == "lorsa":
            activations = lorsa_activations[layer]
            if lorsa_WDs is None:
                raise ValueError("lorsa_WDs required for LORSA steering")
            WDs = lorsa_WDs[layer]
            hook_name = f"blocks.{layer}.hook_attn_out"
        else:
            raise ValueError(f"Unknown feature_type: {ft}")

        # Find activation value at specific position
        target_indices = torch.tensor([0, pos, feature], device=activations.indices().device)
        matches = (activations.indices() == target_indices.unsqueeze(1)).all(dim=0)
        
        if not matches.any():
            details.append({
                "node": repr(node),
                "found": False,
                "reason": "no activation at (batch=0, pos, feature)",
            })
            continue

        activation_value = activations.values()[matches].item()
        # Ensure WDs is not None and WDs[feature] is not None
        if WDs is None or WDs[feature] is None:
            details.append({
                "node": repr(node),
                "found": False,
                "reason": "WDs not available for this feature",
            })
            continue
        contribution_vec = activation_value * WDs[feature]  # [d_model]

        updates_by_hook.setdefault(hook_name, []).append((pos, contribution_vec))
        details.append({
            "node": repr(node),
            "found": True,
            "feature_type": ft,
            "layer": layer,
            "pos": pos,
            "feature": feature,
            "activation_value": float(activation_value),
            "hook_name": hook_name,
        })

    # Create intervention hooks
    def make_hook(updates: List[Tuple[int, torch.Tensor]]):
        def _hook(activation, hook):
            modified = activation.clone()
            for p, contrib in updates:
                modified[0, p] = modified[0, p] + (steering_scale - 1) * contrib
            return modified
        return _hook

    # Register hooks
    for hook_name, updates in updates_by_hook.items():
        model.add_hook(hook_name, make_hook(updates))

    # Run model with interventions
    modified_output, modified_cache = model.run_with_cache(
        cache["embed.hook_input"],
        prepend_bos=False,
    )

    # Clean up hooks
    try:
        model.reset_hooks()
    except Exception:
        pass

    # Recompute activations after intervention
    modified_tc_activations, modified_lorsa_activations = [], []
    
    n_layers = len(tc_activations)  # Assume same number of layers
    for layer in range(n_layers):
        # Recompute LORSA activations
        lorsa_input = modified_cache[f'blocks.{layer}.hook_attn_in']
        if lorsas is not None and layer < len(lorsas) and lorsas[layer] is not None:
            lorsa_dense_activation = lorsas[layer].encode(lorsa_input)
            lorsa_sparse_activation = lorsa_dense_activation.to_sparse_coo()
        else:
            # Create empty sparse tensor
            lorsa_sparse_activation = torch.sparse_coo_tensor(
                torch.empty(3, 0, dtype=torch.long),
                torch.empty(0),
                (1, 64, 128)
            ).to(model_device)
        modified_lorsa_activations.append(lorsa_sparse_activation)

        # Recompute transcoder activations
        tc_input = modified_cache[f'blocks.{layer}.resid_mid_after_ln']
        if transcoders is not None and layer < len(transcoders) and transcoders[layer] is not None:
            tc_dense_activation = transcoders[layer].encode(tc_input)
            tc_sparse_activation = tc_dense_activation.to_sparse_coo()
        else:
            # Create empty sparse tensor
            tc_sparse_activation = torch.sparse_coo_tensor(
                torch.empty(3, 0, dtype=torch.long),
                torch.empty(0),
                (1, 64, 128)
            ).to(model_device)
        modified_tc_activations.append(tc_sparse_activation)
    
    # Update activations dictionary
    modified_activations_dict = {
        "tc": modified_tc_activations,
        "lorsa": modified_lorsa_activations
    }
    
    # Get modified activation of target node
    modified_activation = get_activation_value(target_node_obj, modified_activations_dict)
    
    # Calculate change metrics
    if original_activation != 0:
        activation_ratio = modified_activation / original_activation
        activation_change = modified_activation - original_activation
    else:
        activation_ratio = float('inf') if modified_activation > 0 else 1.0
        activation_change = modified_activation
    
    return {
        "target_node": repr(target_node_obj),
        "original_activation": original_activation,
        "modified_activation": modified_activation,
        "activation_ratio": activation_ratio,
        "activation_change": activation_change,
        "steering_scale": steering_scale,
        "steering_nodes_count": len(steering_node_list),
        "steering_details": details,
    }


def analyze_node_interaction(
    steering_nodes: List[Union[str, dict, "Node", Tuple[int, int, int, str]]],
    target_node: Union[str, dict, "Node", Tuple[int, int, int, str]],
    steering_scale: float,
    cache,
    model,
    transcoders: Optional[Any] = None,
    lorsas: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Analyze steering impact on a single target node.

    Args:
        steering_nodes: List of nodes to steer (can be Node objects, strings, dicts, or tuples)
        target_node: Target node to measure activation change for (can be Node object, string, dict, or tuple)
        steering_scale: Steering scale factor
        cache: Model cache containing intermediate activations
        model: HookedTransformer model
        transcoders: List of transcoder models per layer
        lorsas: List of LORSA models per layer

    Returns:
        Analysis result for the target node
    """
    # Compute activations from cache
    tc_activations: List[torch.Tensor] = []
    lorsa_activations: List[torch.Tensor] = []
    model_device = next(model.parameters()).device
    for layer in range(15):
        # Get Lorsa activations
        lorsa_input = cache[f'blocks.{layer}.hook_attn_in']
        if lorsas is not None and layer < len(lorsas) and lorsas[layer] is not None:
            lorsa_dense_activation = lorsas[layer].encode(lorsa_input)
            lorsa_sparse_activation = lorsa_dense_activation.to_sparse_coo()
        else:
            # Create empty sparse tensor
            lorsa_sparse_activation = torch.sparse_coo_tensor(
                torch.empty(3, 0, dtype=torch.long),
                torch.empty(0),
                (1, 64, 128)
            ).to(model_device)
        lorsa_activations.append(lorsa_sparse_activation)

        # Get TC activations
        tc_input = cache[f'blocks.{layer}.resid_mid_after_ln']
        if transcoders is not None and layer < len(transcoders) and transcoders[layer] is not None:
            tc_dense_activation = transcoders[layer].encode(tc_input)
            tc_sparse_activation = tc_dense_activation.to_sparse_coo()
        else:
            # Create empty sparse tensor
            tc_sparse_activation = torch.sparse_coo_tensor(
                torch.empty(3, 0, dtype=torch.long),
                torch.empty(0),
                (1, 64, 128)
            ).to(model_device)
        tc_activations.append(tc_sparse_activation)

    # Extract weights from transcoders and lorsas
    tc_WDs: List[Optional[torch.Tensor]] = []
    lorsa_WDs: List[Optional[torch.Tensor]] = []

    n_layers = len(tc_activations)  # Assume same number of layers
    for layer in range(n_layers):
        # Extract transcoder weights
        if transcoders is not None and layer < len(transcoders) and transcoders[layer] is not None:
            tc_WDs.append(transcoders[layer].W_D.detach().to(model_device))
        else:
            tc_WDs.append(None)

        # Extract Lorsa weights
        if lorsas is not None and layer < len(lorsas) and lorsas[layer] is not None:
            lorsa_WDs.append(lorsas[layer].W_O.detach().to(model_device))
        else:
            lorsa_WDs.append(None)

    result = analyze_node_activation_impact(
        steering_nodes=steering_nodes,
        target_node=target_node,
        steering_scale=steering_scale,
        cache=cache,
        model=model,
        tc_activations=tc_activations,
        lorsa_activations=lorsa_activations,
        tc_WDs=tc_WDs,
        lorsa_WDs=lorsa_WDs,
        transcoders=transcoders,
        lorsas=lorsas,
    )
    return result