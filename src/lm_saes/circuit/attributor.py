from abc import ABC
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Tuple, Union
import networkx as nx

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint, HookedRootModule

from lm_saes.sae import SparseAutoEncoder
from lm_saes.utils.hooks import compose_hooks, detach_hook, retain_grad_hook

@dataclass
class Node:
    """
    A node in the circuit.
    """

    hook_point: str | None
    """ The hook point of the node. None means the node is the output of the model. """
    reduction: str | None
    """ The reduction function to apply to the node. """

    def reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        reductions = self.reduction.split(".") if self.reduction is not None else []
        for reduction in reductions:
            if reduction == "max":
                tensor = tensor.max()
            elif reduction == "mean":
                tensor = tensor.mean()
            elif reduction == "sum":
                tensor = tensor.sum()
            else:
                try:
                    index = int(reduction)
                    tensor = tensor[index]
                except ValueError:
                    raise ValueError(f"Unknown reduction function: {reduction} in {self.reduction}.")
        return tensor

        
    def __hash__(self):
        return hash((self.hook_point, self.reduction))
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.hook_point == other.hook_point and self.reduction == other.reduction
    
    def __str__(self) -> str:
        hook_point = self.hook_point if self.hook_point is not None else "output"
        return f"{hook_point}.{self.reduction}" if self.reduction is not None else hook_point

class Attributor(ABC):
    def __init__(
        self,
        model: HookedRootModule,
        saes: list[SparseAutoEncoder]
    ):
        self.model = model
        self.saes = saes

    def attribute(
        self,
        input: Any,
        target: Node,
        **kwargs
    ) -> nx.MultiDiGraph:
        """
        Attribute the target hook point of the model to critical points
        in the model, w.r.t. the given input.

        Args:
            input (Any): The input to the model.
            target: The target node to attribute.
            **kwargs: Additional keyword arguments.

        Returns:
            nx.MultiDiGraph: The attributed graph, i.e. the circuit. Each node and edge should have an attribute "attribution",
                showing its "importance" w.r.t. the target.
        """
        raise NotImplementedError

    @contextmanager
    def apply_saes(self):
        """
        Apply the sparse autoencoders to the model.
        """
        fwd_hooks: list[Tuple[Union[str, Callable], Callable]] = []
        def get_fwd_hooks(sae: SparseAutoEncoder) -> list[Tuple[Union[str, Callable], Callable]]:
            if sae.cfg.hook_point_in == sae.cfg.hook_point_out:
                def hook(tensor: torch.Tensor, hook: HookPoint):
                    reconstructed = sae.forward(tensor)
                    return reconstructed + (tensor - reconstructed).detach()
                return [(sae.cfg.hook_point_in, hook)]
            else:
                x = None
                def hook_in(tensor: torch.Tensor, hook: HookPoint):
                    nonlocal x
                    x = tensor
                    return tensor
                def hook_out(tensor: torch.Tensor, hook: HookPoint):
                    nonlocal x
                    assert x is not None, "hook_in must be called before hook_out."
                    reconstructed = sae.forward(x, label=tensor)
                    x = None
                    return reconstructed + (tensor - reconstructed).detach()
                return [(sae.cfg.hook_point_in, hook_in), (sae.cfg.hook_point_out, hook_out)]
        for sae in self.saes:
            hooks = get_fwd_hooks(sae)
            fwd_hooks.extend(hooks)
        with self.model.mount_hooked_modules([(sae.cfg.hook_point_out, "sae", sae) for sae in self.saes]):
            with self.model.hooks(fwd_hooks):
                yield self

    def cache_nodes(
        self,
        input: Any,
        nodes: list[Node | str],
    ):
        """
        Cache the nodes in the model forward pass.

        Args:
            input (Any): The input to the model.
            nodes (list[Node | str]): The nodes or hook points to cache.
        """
        output, cache = self.model.run_with_ref_cache(input, names_filter=[node.hook_point if isinstance(node, Node) else node for node in nodes])
        node_cache: dict[Node | str, torch.Tensor] = {node: node.reduce(cache[node.hook_point] if node.hook_point is not None else output) for node in nodes if isinstance(node, Node)}
        node_cache.update({node: cache[node] for node in nodes if isinstance(node, str)})
        return node_cache

    
class TransformerDirectAttributor(Attributor):
    def attribute(
        self, 
        input: Any,
        target: Node,
        **kwargs
    ) -> nx.MultiDiGraph:
        """
        Attribute the target hook point of the model to critical points
        in the model, w.r.t. the given input.

        Args:
            model (HookedRootModule): The model to attribute.
            saes (dict[str, SparseAutoEncoder]): The sparse autoencoders to use for attribution.
            input (Any): The input to the model.
            target_hook_point (str): The target hook point to attribute.
            target_reduction (Callable[[torch.Tensor], torch.Tensor] | None): The reduction function to apply to the target hook point.
                A reduction function should take a tensor and return a scalar (but still a torch.Tensor). None means identity function.
            **kwargs: Additional keyword arguments.

        Returns:
            nx.MultiDiGraph: The attributed graph, i.e. the circuit. Each node and edge should have an attribute "attribution",
                showing its "importance" w.r.t. the target.
        """
        
        assert isinstance(self.model, HookedTransformer), "TransformerDirectAttributor only supports attributing HookedTransformer."

        threshold: int = kwargs.get("threshold", 0.1)

        with self.apply_saes():
            fwd_hooks = ([(f"{sae.cfg.hook_point_out}.sae.hook_feature_acts", detach_hook) for sae in self.saes if f"{sae.cfg.hook_point_out}.sae.hook_feature_acts" != target.hook_point]
                        + [(f"blocks.{i}.attn.hook_attn_scores", detach_hook) for i in range(12)])
            with self.model.hooks(fwd_hooks):
                cache = self.cache_nodes(input, [f"{sae.cfg.hook_point_out}.sae.hook_feature_acts" for sae in self.saes] + [target])
                cache[target].backward()
                
                # Construct the circuit
                circuit = nx.MultiDiGraph()
                circuit.add_node(target, attribution=cache[target].item(), activation=cache[target].item())
                for sae in self.saes:
                    if f"{sae.cfg.hook_point_out}.sae.hook_feature_acts" == target.hook_point:
                        continue
                    attributions: torch.Tensor = cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts"].grad * cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts"]
                    for index in (attributions > threshold).nonzero():
                        index_str = ".".join(map(str, index.tolist()))
                        index = tuple(index)
                        circuit.add_node(Node(f"{sae.cfg.hook_point_out}.sae.hook_feature_acts", index_str), attribution=attributions[index].item(), activation=cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts"][index].item())
                        circuit.add_edge(Node(f"{sae.cfg.hook_point_out}.sae.hook_feature_acts", index_str), target, attribution=attributions[index].item(), direct_attribution=attributions[index].item())
        return circuit


class TransformerHierachicalAttributor(Attributor):
    def attribute(
        self, 
        input: Any,
        target: Node,
        **kwargs
    ) -> nx.MultiDiGraph:
        """
        Attribute the target hook point of the model to critical points
        in the model, w.r.t. the given input.

        Args:
            model (HookedRootModule): The model to attribute.
            saes (dict[str, SparseAutoEncoder]): The sparse autoencoders to use for attribution.
            input (Any): The input to the model.
            target_hook_point (str): The target hook point to attribute.
            target_reduction (Callable[[torch.Tensor], torch.Tensor] | None): The reduction function to apply to the target hook point.
                A reduction function should take a tensor and return a scalar (but still a torch.Tensor). None means identity function.
            **kwargs: Additional keyword arguments.

        Returns:
            nx.MultiDiGraph: The attributed graph, i.e. the circuit. Each node and edge should have an attribute "attribution",
                showing its "importance" w.r.t. the target.
        """
        
        assert isinstance(self.model, HookedTransformer), "TransformerHierachicalAttributor only supports attributing HookedTransformer."

        threshold: int = kwargs.get("threshold", 0.1)

        with self.apply_saes():
            def generate_attribution_score_filter_hook():
                v = None
                def fwd_hook(tensor: torch.Tensor, hook: HookPoint):
                    nonlocal v
                    v = tensor
                    return tensor
                def attribution_score_filter_hook(grad: torch.Tensor, hook: HookPoint):
                    assert v is not None, "fwd_hook must be called before attribution_score_filter_hook."
                    return (torch.where(v * grad > threshold, grad, torch.zeros_like(grad)),)
                return fwd_hook, attribution_score_filter_hook
            attribution_score_filter_hooks = {sae: generate_attribution_score_filter_hook() for sae in self.saes}
            fwd_hooks = ([(f"{sae.cfg.hook_point_out}.sae.hook_feature_acts", compose_hooks(attribution_score_filter_hooks[sae][0], retain_grad_hook)) for sae in self.saes]
                        + [(f"blocks.{i}.attn.hook_attn_scores", detach_hook) for i in range(12)])
            with self.model.hooks(
                fwd_hooks=fwd_hooks,
                bwd_hooks=[(f"{sae.cfg.hook_point_out}.sae.hook_feature_acts", attribution_score_filter_hooks[sae][1]) for sae in self.saes]
            ):
                cache = self.cache_nodes(input, [f"{sae.cfg.hook_point_out}.sae.hook_feature_acts" for sae in self.saes] + [target])
                cache[target].backward()

                # Construct the circuit
                circuit = nx.MultiDiGraph()
                circuit.add_node(target, attribution=cache[target].item(), activation=cache[target].item())
                for sae in self.saes:
                    attributions: torch.Tensor = cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts"].grad * cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts"]
                    for index in (attributions > threshold).nonzero():
                        index_str = ".".join(map(str, index.tolist()))
                        index = tuple(index)
                        circuit.add_node(Node(f"{sae.cfg.hook_point_out}.sae.hook_feature_acts", index_str), attribution=attributions[index].item(), activation=cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts"][index].item())

        return circuit

