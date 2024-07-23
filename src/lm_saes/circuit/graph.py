from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Node:
    """
    A node in the circuit.
    """

    hook_point: str | None
    """ The hook point of the node. None means the node is the output of the model. """
    reduction: str | None = None
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