import torch
from lm_saes.circuit.utils.batched_features import BatchedFeatures, ConnectedFeatures
from dataclasses import dataclass
from typing import List
from lm_saes import MongoClient

@dataclass
class Node:
    name: str
    influence: float
    visited: bool

class Atlas:
    def __init__(self, features: BatchedFeatures):
        assert len(features) == 1
        self.nodes = {
            features.to_str_nodes()[0]: Node(
                    name=features.to_str_nodes()[0],
                    influence=1.,
                    visited=True,
                )
            }
        self.links = dict()
        self.iteration = 0

    def add_link(self, source: str, target: str, weight: float):
        if (target, source) in self.links:
            return
        self.links[(source, target)] = weight
        target_influence = self.nodes[source].influence * weight
        if target not in self.nodes:
            self.nodes[target] = Node(
                name=target,
                influence=target_influence,
                visited=False,
            )
        else:
            self.nodes[target].influence += target_influence
    
    def update(self, feature_to_explore: BatchedFeatures, connected_features: List[ConnectedFeatures]):
        assert len(feature_to_explore) == len(connected_features)

        for i, connected_feature in enumerate(connected_features):
            for j in range(len(connected_feature.upstream_features)):
                self.add_link(
                    feature_to_explore[i].to_str_nodes()[0],
                    connected_feature.upstream_features[j].to_str_nodes()[0],
                    connected_feature.upstream_values[j].item()
                )
            for j in range(len(connected_feature.downstream_features)):
                self.add_link(
                    feature_to_explore[i].to_str_nodes()[0],
                    connected_feature.downstream_features[j].to_str_nodes()[0],
                    connected_feature.downstream_values[j].item()
                )
    
    def select_top_k_nodes_to_visit(self, k: int):
        unvisited_nodes = [node for node in self.nodes.values() if not node.visited]
        nodes_to_visit = sorted(unvisited_nodes, key=lambda x: x.influence, reverse=True)[:k]
        for node in nodes_to_visit:
            node.visited = True
        return BatchedFeatures.from_str_nodes([node.name for node in nodes_to_visit])


    def export_to_json(self):
        return {
            "nodes": [
                {node.name: node.influence}
                for node in self.nodes.values()
            ],
            "links": [
                {
                    "source": link[0],
                    "target": link[1],
                    "weight": self.links[link]
                }
                for link in self.links
            ]
        }
