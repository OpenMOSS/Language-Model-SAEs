from abc import ABC
from typing import Any

import networkx as nx
import torch
from transformer_lens.hook_points import HookedRootModule, HookPoint

from .graph import Node
from ..utils.hooks import compose_hooks, detach_hook, retain_grad_hook


class Cache:
    def __init__(self, output, cache: dict[str, torch.Tensor]):
        self.cache = cache
        self.output = output

    def tensor(self, node: Node) -> torch.Tensor:
        return node.reduce(self[node.hook_point])

    def grad(self, node: Node) -> torch.Tensor | None:
        grad = self[node.hook_point].grad
        return node.reduce(grad) if grad is not None else None

    def __getitem__(self, key: Node | str | None) -> torch.Tensor:
        if isinstance(key, Node):
            return self.tensor(key)
        return self.cache[key] if key is not None else self.output


class Attributor(ABC):
    def __init__(self, model: HookedRootModule):
        self.model = model

    def attribute(self, input: Any, target: Node, candidates: list[Node], **kwargs) -> nx.MultiDiGraph:
        """
        Attribute the target hook point of the model to given candidates in the model, w.r.t. the given input.

        Args:
            input (Any): The input to the model.
            target: The target node to attribute.
            candidates: The intermediate nodes to attribute to.
            **kwargs: Additional keyword arguments.

        Returns:
            nx.MultiDiGraph: The attributed graph, i.e. the circuit. Each node and edge should have an attribute "attribution",
                showing its "importance" w.r.t. the target.
        """
        raise NotImplementedError

    def cache_nodes(
        self,
        input: Any,
        nodes: list[Node],
    ):
        """
        Cache the activation of  in the model forward pass.

        Args:
            input (Any): The input to the model.
            nodes (list[Node]): The nodes to cache.
        """
        output, cache = self.model.run_with_ref_cache(input, names_filter=[node.hook_point for node in nodes])
        return Cache(output, cache)


class DirectAttributor(Attributor):
    def attribute(self, input: Any, target: Node, candidates: list[Node], **kwargs) -> nx.MultiDiGraph:
        """
        Attribute the target node of the model to given candidates in the model, w.r.t. the given input.

        Args:
            input (Any): The input to the model.
            target: The target node to attribute.
            candidates: The intermediate nodes to attribute to.
            **kwargs: Additional keyword arguments.

        Returns:
            nx.MultiDiGraph: The attributed graph, i.e. the circuit. Each node and edge should have an attribute "attribution",
                showing its "importance" w.r.t. the target.
        """

        threshold: int = kwargs.get("threshold", 0.1)

        fwd_hooks = [
            (candidate.hook_point, detach_hook) for candidate in candidates if candidate.hook_point != target.hook_point
        ]
        with self.model.hooks(fwd_hooks):
            cache = self.cache_nodes(input, candidates + [target])
            cache[target].backward()

            # Construct the circuit
            circuit = nx.MultiDiGraph()
            circuit.add_node(target, attribution=cache[target].item(), activation=cache[target].item())
            for candidate in candidates:
                if candidate.hook_point == target.hook_point:
                    continue
                grad = cache.grad(candidate)
                if grad is None:
                    continue
                attributions = grad * cache[candidate]
                if len(attributions.shape) == 0:
                    if attributions > threshold:
                        circuit.add_node(candidate, attribution=attributions.item(), activation=cache[candidate].item())
                        circuit.add_edge(
                            candidate, target, attribution=attributions.item(), direct_attribution=attributions.item()
                        )
                else:
                    for index in (attributions > threshold).nonzero():
                        index = tuple(index.tolist())
                        circuit.add_node(
                            candidate.append_reduction(*index),
                            attribution=attributions[index].item(),
                            activation=cache[candidate][index].item(),
                        )
                        circuit.add_edge(
                            candidate.append_reduction(*index),
                            target,
                            attribution=attributions[index].item(),
                            direct_attribution=attributions[index].item(),
                        )
        return circuit


class HierachicalAttributor(Attributor):
    def attribute(self, input: Any, target: Node, candidates: list[Node], **kwargs) -> nx.MultiDiGraph:
        """
        Attribute the target node of the model to given candidates in the model, w.r.t. the given input.

        Args:
            input (Any): The input to the model.
            target: The target node to attribute.
            candidates: The intermediate nodes to attribute to.
            **kwargs: Additional keyword arguments.

        Returns:
            nx.MultiDiGraph: The attributed graph, i.e. the circuit. Each node and edge should have an attribute "attribution",
                showing its "importance" w.r.t. the target.
        """

        threshold: int = kwargs.get("threshold", 0.1)

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

        attribution_score_filter_hooks = {
            candidate: generate_attribution_score_filter_hook() for candidate in candidates
        }
        fwd_hooks = [
            (candidate.hook_point, compose_hooks(attribution_score_filter_hooks[candidate][0], retain_grad_hook))
            for candidate in candidates
        ]
        with self.model.hooks(
            fwd_hooks=fwd_hooks,
            bwd_hooks=[
                (candidate.hook_point, attribution_score_filter_hooks[candidate][1]) for candidate in candidates
            ],
        ):
            cache = self.cache_nodes(input, candidates + [target])
            cache[target].backward()

            # Construct the circuit
            circuit = nx.MultiDiGraph()
            circuit.add_node(target, attribution=cache[target].item(), activation=cache[target].item())
            for candidate in candidates:
                grad = cache.grad(candidate)
                if grad is None:
                    continue
                attributions = grad * cache[candidate]
                if len(attributions.shape) == 0:
                    if attributions > threshold:
                        circuit.add_node(candidate, attribution=attributions.item(), activation=cache[candidate].item())
                else:
                    for index in (attributions > threshold).nonzero():
                        index = tuple(index.tolist())
                        circuit.add_node(
                            candidate.append_reduction(*index),
                            attribution=attributions[index].item(),
                            activation=cache[candidate][index].item(),
                        )
                        circuit.add_edge(
                            candidate.append_reduction(*index), target, attribution=attributions[index].item()
                        )

        return circuit
