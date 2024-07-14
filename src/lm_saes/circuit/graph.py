from dataclasses import dataclass
from typing import Callable

import networkx as nx
import torch


class Node:
    """
    A node in the circuit.
    """

    def __init__(self, hook_point: str | None, reduction: str | None = None):
        """
        Initialize a node.

        Args:
            hook_point (str | None): The hook point of the node. None means the node is the output of the model.
            reduction (str | None): The reduction function to apply to the node.
        """
        self.hook_point = hook_point
        self.reduction = reduction

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

    def filter(self, tensor: torch.Tensor) -> torch.Tensor:
        reductions = self.reduction.split(".") if self.reduction is not None else []
        indices: list[slice | int] = []
        for reduction in reductions:
            if reduction == "*":
                indices.append(slice(None))
            else:
                try:
                    index = int(reduction)
                    indices.append(index)
                except ValueError:
                    raise ValueError(f"Unknown reduction function: {reduction} in {self.reduction}.")
        filtered = torch.zeros_like(tensor)
        filtered[tuple(indices)] = tensor[tuple(indices)]
        return filtered
    
    def append_reduction(self, *reduction: list[str | int]) -> "Node":
        reduction: str = ".".join(map(str, reduction))
        return Node(self.hook_point, f"{self.reduction}.{reduction}" if self.reduction is not None else reduction)
    
    def __hash__(self):
        return hash((self.hook_point, self.reduction))
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.hook_point == other.hook_point and self.reduction == other.reduction
    
    def __str__(self) -> str:
        hook_point = self.hook_point if self.hook_point is not None else "output"
        return f"{hook_point}.{self.reduction}" if self.reduction is not None else hook_point
    
class CustomReductionNode(Node):
    def __init__(self, hook_point: str | None, reduction_fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__(hook_point)
        self.reduction_fn = reduction_fn

    def reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.reduction_fn(tensor)
    
    def __str__(self) -> str:
        hook_point = self.hook_point if self.hook_point is not None else "output"
        return f"{hook_point}.{self.reduction_fn.__name__}"
    
def compose_circuits(a: nx.DiGraph, b: nx.DiGraph) -> nx.DiGraph:
    """
    Compose two circuits.
    """
    c = nx.compose(a, b)
    node_data = {n: a.nodes[n]["attribution"] + b.nodes[n]["attribution"] for n in a.nodes & b.nodes}
    nx.set_node_attributes(c, node_data, "attribution")
    edge_data = {e: a.edges[e]["attribution"] + b.edges[e]["attribution"] for e in a.edges & b.edges}
    nx.set_edge_attributes(c, edge_data, "attribution")
    return c