import torch
from lm_saes.analysis.global_weights.batched_features import BatchedFeatures, ConnectedFeatures
from dataclasses import dataclass
from typing import List
from lm_saes import MongoClient



@dataclass
class Link:
    source: str
    target: str
    weight: float

class Atlas:
    def __init__(self, features: BatchedFeatures):
        self.nodes = set(features.to_str_nodes())
        self.links = []

    def add_link(self, source: str, target: str, weight: float):
        self.links.append(Link(source=source, target=target, weight=weight))
    
    def update(self, feature_to_explore: BatchedFeatures, connected_features: List[ConnectedFeatures]):
        assert len(feature_to_explore) == len(connected_features)
        original_nodes = self.nodes.copy()

        for i, connected_feature in enumerate(connected_features):
            upstream_str_nodes = connected_feature.upstream_features.to_str_nodes()
            downstream_str_nodes = connected_feature.downstream_features.to_str_nodes()
            self.nodes.update(upstream_str_nodes)
            self.nodes.update(downstream_str_nodes)
            for j in range(len(connected_feature.upstream_features)):
                self.add_link(
                    connected_feature.upstream_features[j].to_str_nodes()[0],
                    feature_to_explore[i].to_str_nodes()[0],
                    connected_feature.upstream_values[j].item()
                )
            for j in range(len(connected_feature.downstream_features)):
                self.add_link(
                    feature_to_explore[i].to_str_nodes()[0],
                    connected_feature.downstream_features[j].to_str_nodes()[0],
                    connected_feature.downstream_values[j].item()
                )

        added_nodes = list(self.nodes - original_nodes)
        return BatchedFeatures.from_str_nodes(added_nodes)


    def export_to_json(self):
        return {
            "nodes": list(self.nodes),
            "links": [link.__dict__ for link in self.links]
        }
