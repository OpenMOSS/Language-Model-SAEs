import logging
import os
import time
from typing import List, Union
import torch
from transformers import AutoTokenizer
from pydantic import BaseModel

from ..graph import Graph, prune_graph

logger = logging.getLogger(__name__)

class Metadata(BaseModel):
    slug: str
    sae_series: str
    prompt_tokens: List[str]
    prompt: str
    
    node_threshold: float | None = None
    schema_version: int | None = 1
    lorsa_analysis_name: str | None = None
    clt_analysis_name: str | None = None

class Node(BaseModel):
    node_id: str
    feature: int
    layer: int
    ctx_idx: int
    feature_type: str
    token_prob: float = 0.0
    is_target_logit: bool = False
    run_idx: int = 0
    reverse_ctx_idx: int = 0
    jsNodeId: str
    clerp: str = ""
    influence: float | None = None
    activation: float | None = None
    lorsa_pattern: list | None = None

    def __init__(self, **data):
        if "node_id" in data and "jsNodeId" not in data:
            data["jsNodeId"] = data["node_id"]
        super().__init__(**data)

    @classmethod
    def feature_node(cls, layer, pos, feat_idx, is_lorsa, influence=None, activation=None, lorsa_pattern=None):
        """Create a feature node."""

        def cantor_pairing(x, y):
            return (x + y) * (x + y + 1) // 2 + y

        reverse_ctx_idx = 0
        layer = 2 * layer + int(not is_lorsa)
        return cls(
            node_id=f"{layer}_{feat_idx}_{pos}",
            feature=cantor_pairing(layer, feat_idx),
            layer=layer,
            ctx_idx=pos,
            feature_type="lorsa" if is_lorsa else "cross layer transcoder",
            jsNodeId=f"{layer}_{feat_idx}-{reverse_ctx_idx}",
            influence=influence,
            activation=activation,
            lorsa_pattern=lorsa_pattern.tolist(),
        )

    @classmethod
    def error_node(cls, layer, pos, is_lorsa, influence=None):
        """Create an error node."""
        reverse_ctx_idx = 0
        return cls(
            node_id=f"{layer}_error_{pos}",
            feature=-1,
            layer=layer,
            ctx_idx=pos,
            feature_type="lorsa error" if is_lorsa else "mlp reconstruction error",
            jsNodeId=f"{layer}_{pos}-{reverse_ctx_idx}",
            influence=influence,
        )

    @classmethod
    def token_node(cls, pos, vocab_idx, influence=None):
        """Create a token node."""
        return cls(
            node_id=f"E_{vocab_idx}_{pos}",
            feature=pos,
            layer=-1,
            ctx_idx=pos,
            feature_type="embedding",
            jsNodeId=f"E_{vocab_idx}-{pos}",
            influence=influence,
        )

    @classmethod
    def logit_node(
        cls,
        pos,
        vocab_idx,
        token,
        num_layers,
        target_logit=False,
        token_prob=0.0,
    ):
        """Create a logit node."""
        layer = 2 * num_layers
        return cls(
            node_id=f"{layer}_{vocab_idx}_{pos}",
            feature=vocab_idx,
            layer=layer,
            ctx_idx=pos,
            feature_type="logit",
            token_prob=token_prob,
            is_target_logit=target_logit,
            jsNodeId=f"L_{vocab_idx}-{pos}",
            clerp=f'Output "{token}" (p={token_prob:.3f})',
        )


class Link(BaseModel):
    source: str
    target: str
    weight: float


class Model(BaseModel):
    metadata: Metadata
    nodes: List[Node]
    links: List[dict]

def process_token(token: str) -> str:
    return token.replace("\n", "⏎").replace("\t", "→").replace("\r", "↵")

def load_graph_data(file_path) -> Graph:
    """Load graph data from a PyTorch file."""
    start_time = time.time()
    graph = Graph.from_pt(file_path)
    time_ms = (time.time() - start_time) * 1000
    logger.info(f"Loading graph data: {time_ms=:.2f} ms")
    return graph


def create_nodes(graph: Graph, node_mask, tokenizer, cumulative_scores, use_lorsa):
    """Create all nodes for the graph."""
    start_time = time.time()

    nodes = {}

    n_features = len(graph.selected_features)
    layers = graph.cfg.n_layers
    error_end_idx = n_features + 2 * graph.n_pos * layers if use_lorsa else n_features + graph.n_pos * layers
    token_end_idx = error_end_idx + len(graph.input_tokens)

    for node_idx in node_mask.nonzero().squeeze().tolist():
        if node_idx in range(n_features):
            if use_lorsa:
                orig_feature_idx = graph.selected_features[node_idx]
                is_lorsa = orig_feature_idx < len(graph.lorsa_active_features)
                if is_lorsa:
                    layer, pos, feat_idx = graph.lorsa_active_features[orig_feature_idx].tolist()
                    interested_activation = graph.lorsa_activation_values
                else:
                    orig_feature_idx = orig_feature_idx - len(graph.lorsa_active_features)
                    layer, pos, feat_idx = graph.clt_active_features[orig_feature_idx].tolist()
                    interested_activation = graph.clt_activation_values
            else:
                is_lorsa = False
                orig_feature_idx = graph.selected_features[node_idx]
                layer, pos, feat_idx = graph.clt_active_features[orig_feature_idx].tolist()
                interested_activation = graph.clt_activation_values
            nodes[node_idx] = Node.feature_node(
                layer,
                pos,
                feat_idx,
                is_lorsa=is_lorsa,
                influence=cumulative_scores[node_idx],
                activation=interested_activation[orig_feature_idx],
                lorsa_pattern=graph.lorsa_pattern[node_idx],
            )
            
        elif node_idx in range(n_features, error_end_idx):
            layer, pos = divmod(node_idx - n_features, graph.n_pos)
            if use_lorsa:
                is_lorsa = node_idx < n_features + graph.n_pos * layers
                if is_lorsa:
                    layer = 2 * layer
                else:
                    layer = 2 * (layer - layers) + 1
            else:
                is_lorsa = False
                layer = 2 * layer + 1
            nodes[node_idx] = Node.error_node(layer, pos, is_lorsa, influence=cumulative_scores[node_idx])
        elif node_idx in range(error_end_idx, token_end_idx):
            pos = node_idx - error_end_idx
            nodes[node_idx] = Node.token_node(
                pos, graph.input_tokens[pos], influence=cumulative_scores[node_idx]
            )
        elif node_idx in range(token_end_idx, len(cumulative_scores)):
            pos = node_idx - token_end_idx
            nodes[node_idx] = Node.logit_node(
                pos=graph.n_pos - 1,
                vocab_idx=graph.logit_tokens[pos],
                token=process_token(tokenizer.decode(graph.logit_tokens[pos])),
                target_logit=pos == 0,
                token_prob=graph.logit_probabilities[pos],
                num_layers=layers,
            )

    total_time = (time.time() - start_time) * 1000
    logger.info(f"Total node creation: {total_time=:.2f} ms")

    return nodes


def create_used_nodes_and_edges(graph: Graph, nodes, edge_mask):
    """Filter to only used nodes and create edges."""
    start_time = time.time()
    edges = edge_mask.cpu().numpy()
    dsts, srcs = edges.nonzero()
    weights = graph.adjacency_matrix.cpu().numpy()[dsts, srcs].tolist()

    used_edges = [
        {"source": nodes[src].node_id, "target": nodes[dst].node_id, "weight": weight}
        for src, dst, weight in zip(srcs, dsts, weights)
        if src in nodes and dst in nodes
    ]

    connected_ids = set()
    for edge in used_edges:
        connected_ids.add(edge["source"])
        connected_ids.add(edge["target"])

    nodes_before = len(nodes)
    used_nodes = [
        node
        for node in nodes.values()
        if node.node_id in connected_ids or node.feature_type in ["embedding", "logit"]
    ]
    nodes_after = len(used_nodes)
    logger.info(f"Filtered {nodes_before - nodes_after} nodes")

    time_ms = (time.time() - start_time) * 1000
    logger.info(f"Creating used nodes and edges: {time_ms=:.2f} ms")
    logger.info(f"Used nodes: {len(used_nodes)}, Used edges: {len(used_edges)}")

    return used_nodes, used_edges


def build_model(
    graph: Graph,
    used_nodes,
    used_edges,
    slug,
    sae_series,
    node_threshold,
    tokenizer,
    lorsa_analysis_name,
    clt_analysis_name,
):
    """Build the full model object."""
    start_time = time.time()

    meta = Metadata(
        slug=slug,
        sae_series=sae_series,
        prompt_tokens=[process_token(tokenizer.decode(t)) for t in graph.input_tokens],
        prompt=graph.input_string,
        node_threshold=node_threshold,
        lorsa_analysis_name=lorsa_analysis_name,
        clt_analysis_name=clt_analysis_name,
    )

    full_model = Model(
        metadata=meta,
        nodes=used_nodes,
        links=used_edges,
    )

    time_ms = (time.time() - start_time) * 1000
    logger.info(f"Building model: {time_ms=:.2f} ms")

    return full_model


def create_graph_files(
    graph: Graph,
    slug: str,
    output_path,
    node_threshold=0.8,
    edge_threshold=0.98,
    sae_series=None,
    lorsa_analysis_name="",
    clt_analysis_name="",
    use_lorsa:bool=True,
):
    total_start_time = time.time()

    if os.path.exists(output_path):
        assert os.path.isdir(output_path)
    else:
        os.makedirs(output_path, exist_ok=True)

    if sae_series is None:
        if graph.sae_series is None:
            raise ValueError(
                "Neither sae_series nor graph.sae_series was set. One must be set to identify "
                "which transcoders were used when creating the graph."
            )
        sae_series = graph.sae_series

    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph.to(device)
    node_mask, edge_mask, cumulative_scores = (
        el.to(device) for el in prune_graph(graph, node_threshold, edge_threshold)
    )

    print(f'{graph.cfg.tokenizer_name=}')
    tokenizer = AutoTokenizer.from_pretrained(graph.cfg.tokenizer_name)
    nodes = create_nodes(graph, node_mask, tokenizer, cumulative_scores, use_lorsa=use_lorsa)
    used_nodes, used_edges = create_used_nodes_and_edges(graph, nodes, edge_mask)
    model = build_model(
        graph,
        used_nodes,
        used_edges,
        slug,
        sae_series,
        node_threshold,
        tokenizer,
        lorsa_analysis_name,
        clt_analysis_name,
    )

    # Write the output locally
    with open(os.path.join(output_path, f"{slug}.json"), "w") as f:
        f.write(model.model_dump_json(indent=2))
    logger.info(f"Graph data written to {output_path}")

    total_time_ms = (time.time() - total_start_time) * 1000
    logger.info(f"Total execution time: {total_time_ms=:.2f} ms")