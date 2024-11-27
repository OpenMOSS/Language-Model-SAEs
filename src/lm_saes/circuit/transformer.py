from typing import Any

from transformer_lens import HookedTransformer

from .attributors import DirectAttributor, HierachicalAttributor
from .context import apply_sae
from .graph import Node
from ..sae import SparseAutoEncoder
from ..utils.hooks import detach_hook


def direct_attribute_transformer_with_saes(
    model: HookedTransformer,
    saes: list[SparseAutoEncoder],
    input: Any,
    target: Node,
    candidates: list[Node] | None = None,
    threshold: float = 0.1,
):
    """
    Attribute the target hook point of the model to given candidates in the model, w.r.t. the given input.
    This attribution will only consider the direct connections between the target and the candidates, but not
    indirect effects through intermediate candidates.

    Args:
        model (HookedTransformer): The model to attribute.
        saes (list[SparseAutoEncoder]): The sparse autoencoders to apply.
        input (Any): The input to the model.
        target: The target node to attribute.
        candidates: The intermediate nodes to attribute to. If None, default to all sae feature activations and all attention scores.
        threshold (float): The threshold to prune the circuit.

    Returns:
        nx.MultiDiGraph: The attributed graph, i.e. the circuit. Each node and edge should have an attribute "attribution",
            showing its "importance" w.r.t. the target.
    """

    with apply_sae(model, saes):
        with model.hooks([(f"blocks.{i}.attn.hook_attn_scores", detach_hook) for i in range(12)]):
            attributor = DirectAttributor(model)
            if candidates is None:
                candidates = [Node(f"{sae.cfg.hook_point_out}.sae.hook_feature_acts") for sae in saes] + [
                    Node(f"blocks.{i}.attn.hook_attn_scores") for i in range(12)
                ]
            return attributor.attribute(input=input, target=target, candidates=candidates, threshold=threshold)


def hierarchical_attribute_transformer_with_saes(
    model: HookedTransformer,
    saes: list[SparseAutoEncoder],
    input: Any,
    target: Node,
    candidates: list[Node] | None = None,
    threshold: float = 0.1,
):
    """
    Attribute the target hook point of the model to given candidates in the model, w.r.t. the given input.
    This attribution will consider both the direct connections between the target and the candidates, and
    indirect effects through intermediate candidates.

    Args:
        model (HookedTransformer): The model to attribute.
        saes (list[SparseAutoEncoder]): The sparse autoencoders to apply.
        input (Any): The input to the model.
        target: The target node to attribute.
        candidates: The intermediate nodes to attribute to. If None, default to all sae feature activations and all attention scores.
        threshold (float): The threshold to prune the circuit.

    Returns:
        nx.MultiDiGraph: The attributed graph, i.e. the circuit. Each node and edge should have an attribute "attribution",
            showing its "importance" w.r.t. the target.
    """

    with apply_sae(model, saes):
        with model.hooks([(f"blocks.{i}.attn.hook_attn_scores", detach_hook) for i in range(12)]):
            attributor = HierachicalAttributor(model)
            if candidates is None:
                candidates = [Node(f"{sae.cfg.hook_point_out}.sae.hook_feature_acts") for sae in saes] + [
                    Node(f"blocks.{i}.attn.hook_attn_scores") for i in range(12)
                ]
            return attributor.attribute(input=input, target=target, candidates=candidates, threshold=threshold)
