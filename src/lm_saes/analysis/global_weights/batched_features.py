from dataclasses import dataclass
from jaxtyping import Float, Bool, Int
from typing import List
import torch

@dataclass
class BatchedFeatures:
    layer: Int[torch.sparse.Tensor, "batch_size"]
    index: Int[torch.sparse.Tensor, "batch_size"]
    is_lorsa: Bool[torch.sparse.Tensor, "batch_size"]
    
    def __post_init__(self):
        assert self.layer.ndim == self.index.ndim == self.is_lorsa.ndim
        assert self.layer.shape == self.index.shape == self.is_lorsa.shape
        
        if self.layer.ndim == 0:
            self.layer = self.layer.unsqueeze(0)
            self.index = self.index.unsqueeze(0)
            self.is_lorsa = self.is_lorsa.unsqueeze(0)
    
    def __len__(self):
        return self.layer.shape[0]

    def sublayer_index(self) -> Float[torch.Tensor, "batch_size"]:
        return self.layer + 0.5 * (~self.is_lorsa).int()
    
    def __getitem__(self, index):
        return BatchedFeatures(
            layer=self.layer[index],
            index=self.index[index],
            is_lorsa=self.is_lorsa[index],
        )
    
    def to_str_nodes(self):
        nodes = []
        for i in range(len(self)):
            if self.is_lorsa[i]:
                nodes.append(f"L{self.layer[i].item()}A#{self.index[i].item()}")
            else:
                nodes.append(f"L{self.layer[i].item()}M#{self.index[i].item()}")
        return nodes
    
    @classmethod
    def from_str_nodes(cls, nodes: List[str], device = 'cuda'):
        layers = []
        indices = []
        is_lorsa = []
        for node in nodes:
            if "A" in node:
                layer, index = node.split("L")[1].split("A#")[0], node.split("A#")[1]
                layers.append(int(layer))
                indices.append(int(index))
                is_lorsa.append(True)
            else:
                layer, index = node.split("L")[1].split("M#")[0], node.split("M#")[1]
                layers.append(int(layer))
                indices.append(int(index))
                is_lorsa.append(False)
        return cls(
            layer=torch.tensor(layers, device=device),
            index=torch.tensor(indices, device=device),
            is_lorsa=torch.tensor(is_lorsa, device=device)
        )
    
    @classmethod
    def empty(cls, device='cuda'):
        return cls(
            layer=torch.tensor([], device=device, dtype=torch.int32),
            index=torch.tensor([], device=device, dtype=torch.int32),
            is_lorsa=torch.tensor([], device=device, dtype=torch.bool),
        )
    
    def to(self, device: torch.device):
        return BatchedFeatures(
            layer=self.layer.to(device),
            index=self.index.to(device),
            is_lorsa=self.is_lorsa.to(device),
        )



@dataclass
class ConnectedFeatures:
    upstream_features: BatchedFeatures
    downstream_features: BatchedFeatures
    upstream_values: torch.Tensor
    downstream_values: torch.Tensor

    def sort(self):
        upstream_indices = torch.argsort(self.upstream_values, descending=True)
        downstream_indices = torch.argsort(self.downstream_values, descending=True)
        return ConnectedFeatures(
            upstream_features=self.upstream_features[upstream_indices],
            downstream_features=self.downstream_features[downstream_indices],
            upstream_values=self.upstream_values[upstream_indices],
            downstream_values=self.downstream_values[downstream_indices],
        )
