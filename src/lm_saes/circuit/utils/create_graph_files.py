import logging
import os
import time
from typing import List, Union
import torch
from transformers import AutoTokenizer
from pydantic import BaseModel

from ..graph_lc0 import Graph, prune_graph
from ..leela_board import *

from typing import Callable, Optional

logger = logging.getLogger(__name__)

class Metadata(BaseModel):
    slug: str
    sae_series: str
    prompt_tokens: List[str]
    prompt: str
    
    node_threshold: float | None = None
    schema_version: int | None = 1
    lorsa_analysis_name: str | None = None
    tc_analysis_name: str | None = None   # ← 替换 clt_analysis_name；移除 LORSA 字段
    logit_moves: List[str] = []          # ← 新增：所有 logits 的 UCI 文本
    target_move: str | None = None       # ← 可选：第一个（目标）logit 的 UCI

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

    def __init__(self, **data):
        if "node_id" in data and "jsNodeId" not in data:
            data["jsNodeId"] = data["node_id"]
        super().__init__(**data)

    @classmethod
    def feature_node(cls, layer, pos, feat_idx, is_lorsa, influence=None, 
                     activation=None):
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


class ActivationInfo(BaseModel):
    featureId: int
    type: str  # 'lorsa' or 'tc'
    layer: int
    position: int
    head_idx: int | None = None  # for lorsa features
    feature_idx: int | None = None  # for tc features
    activation_value: float
    activations: List[float]  # 64位置的激活数组
    zPatternIndices: List[List[int]] | None = None  # [[q_positions], [k_positions]]
    zPatternValues: List[float] | None = None  # z_pattern值


class Model(BaseModel):
    metadata: Metadata
    nodes: List[Node]
    links: List[dict]
    activation_info: List[ActivationInfo] | None = None  # 全局激活信息


def process_token(token: str) -> str:
    return token.replace("\n", "⏎").replace("\t", "→").replace("\r", "↵")


def process_chess_input(input_string: str) -> str:
    return input_string # TODO


def load_graph_data(file_path) -> Graph:
    """Load graph data from a PyTorch file."""
    start_time = time.time()
    graph = Graph.from_pt(file_path)
    time_ms = (time.time() - start_time) * 1000
    logger.info(f"Loading graph data: {time_ms=:.2f} ms")
    return graph


def create_nodes(
    graph: Graph,
    node_mask: torch.Tensor,
    cumulative_scores: torch.Tensor,
    to_uci: Optional[Callable[[int], str]] = None,
):
    """Create all nodes for the graph."""
    start_time = time.time()
    nodes: dict[int, Node] = {}

    n_features = len(graph.selected_features)
    print(f'{n_features = }')
    layers = graph.cfg.n_layers
    error_end_idx = n_features + 2 * graph.n_pos * layers
    token_end_idx = error_end_idx + len(graph.input_tokens)
    
    for node_idx in node_mask.nonzero().squeeze().tolist():
        if node_idx in range(n_features):
            orig_feature_idx = int(graph.selected_features[node_idx])
            is_lorsa = orig_feature_idx < len(graph.lorsa_active_features)
            if is_lorsa:
                layer, pos, feat_idx = (
                    graph.lorsa_active_features[orig_feature_idx].tolist()
                )
                # 对应的 activation 标量
                activation_value = (
                    graph.lorsa_activation_values[orig_feature_idx]
                )
            else:
                orig_feature_idx = (
                    orig_feature_idx - len(graph.lorsa_active_features)
                )
                layer, pos, feat_idx = (
                    graph.tc_active_features[orig_feature_idx].tolist()
                )
                # 对应的 activation 标量
                activation_value = (
                    graph.tc_activation_values[orig_feature_idx]
                )
            nodes[node_idx] = Node.feature_node(
                layer=layer,
                pos=pos,
                feat_idx=feat_idx,
                is_lorsa=is_lorsa,
                influence=float(cumulative_scores[node_idx]),
                activation=float(activation_value),
            )

        elif node_idx in range(n_features, error_end_idx):
            # LORSA error：按 (layer, pos) 展开
            rel = node_idx - n_features
            layer, pos = divmod(rel, graph.n_pos)
            is_lorsa = node_idx < n_features + graph.n_pos * layers
            if is_lorsa:
                layer = 2 * layer
            else:
                layer = 2 * (layer - layers) + 1
            nodes[node_idx] = Node.error_node(
                layer=int(layer),
                pos=int(pos),
                is_lorsa=is_lorsa,
                influence=float(cumulative_scores[node_idx]),
            )

        elif node_idx in range(error_end_idx, token_end_idx):
            pos = node_idx - error_end_idx

            dummy_vocab_idx = pos
            nodes[node_idx] = Node.token_node(
                pos=int(pos),
                vocab_idx=dummy_vocab_idx,
                influence=float(cumulative_scores[node_idx]),
            )

        elif node_idx in range(token_end_idx, len(cumulative_scores)):

            pos = node_idx - token_end_idx
            move_idx = int(graph.logit_tokens[pos])
            
            move_str = (
                to_uci(move_idx)
                if to_uci is not None
                else f"idx:{move_idx}"
            )
            nodes[node_idx] = Node.logit_node(
                pos=graph.n_pos - 1,
                vocab_idx=move_idx,
                token=move_str,
                target_logit=(pos == 0),
                token_prob=float(graph.logit_probabilities[pos]),
                num_layers=int(layers),
            )

    total_time = (time.time() - start_time) * 1000
    logger.info(f"Total node creation: {total_time=:.2f} ms")

    return nodes


def extract_activation_info(graph: Graph) -> List[ActivationInfo] | None:
    """从graph中提取激活信息到统一格式"""
    if graph.activation_info is None:
        return None
        
    activation_info_list = []
    
    # 优先使用k side，如果没有则使用q side
    for side in ['k', 'q']:
        if side in graph.activation_info and graph.activation_info[side] is not None:
            features = graph.activation_info[side].get('features', [])
            for feature_info in features:
                activation_info_list.append(ActivationInfo(
                    featureId=feature_info['featureId'],
                    type=feature_info['type'],
                    layer=feature_info['layer'],
                    position=feature_info['position'],
                    head_idx=feature_info.get('head_idx'),
                    feature_idx=feature_info.get('feature_idx'),
                    activation_value=feature_info['activation_value'],
                    activations=feature_info['activations'],
                    zPatternIndices=feature_info.get('zPatternIndices'),
                    zPatternValues=feature_info.get('zPatternValues'),
                ))
            break  # 找到一个side就停止
    
    return activation_info_list if activation_info_list else None


def create_used_nodes_and_edges(graph: Graph, nodes, edge_mask):
    """Filter to only used nodes and create edges."""
    start_time = time.time()
    edges = edge_mask.cpu().numpy()
    dsts, srcs = edges.nonzero()
    weights = graph.adjacency_matrix.cpu().numpy()[dsts, srcs].tolist()

    used_edges = [
        {
            "source": nodes[src].node_id,
            "target": nodes[dst].node_id,
            "weight": weight,
        }
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
        if (
            node.node_id in connected_ids
            or node.feature_type in ["embedding", "logit"]
        )
    ]
    nodes_after = len(used_nodes)
    logger.info(f"Filtered {nodes_before - nodes_after} nodes")

    time_ms = (time.time() - start_time) * 1000
    logger.info(f"Creating used nodes and edges: {time_ms=:.2f} ms")
    logger.info(
        f"Used nodes: {len(used_nodes)}, Used edges: {len(used_edges)}"
    )

    return used_nodes, used_edges


def build_model(
    graph: Graph,
    used_nodes: List[Node],
    used_edges: List[dict],
    slug: str,
    sae_series: str,
    node_threshold: float,
    lorsa_analysis_name: str,
    tc_analysis_name: str,
    logit_moves: List[str] | None = None,
    target_move: str | None = None,
):
    """Build the full model object."""
    start_time = time.time()

    meta = Metadata(
        slug=slug,
        sae_series=sae_series,
        # prompt_tokens=[process_token(tokenizer.decode(t)) for t in graph.input_tokens],
        prompt_tokens = [process_chess_input(graph.input_string)],
        prompt=graph.input_string,
        node_threshold=node_threshold,
        lorsa_analysis_name=lorsa_analysis_name,
        tc_analysis_name=tc_analysis_name,
        logit_moves=logit_moves or [],       # ← 写入
        target_move=target_move,             # ← 写入
    )

    # 提取激活信息
    activation_info = extract_activation_info(graph)
    
    full_model = Model(
        metadata=meta,
        nodes=used_nodes,
        links=used_edges,
        activation_info=activation_info,
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
    tc_analysis_name="",
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
    
    fen = graph.input_string
    lboard = None
    if fen:
        print(f'in graph input_string {fen = }')
        lboard = LeelaBoard.from_fen(fen)
    else:
        print('[Warning] fen is none')
        
    to_uci = lboard.idx2uci if lboard is not None else None 
    
    if isinstance(graph.logit_tokens, torch.Tensor):
        _logit_idxs = graph.logit_tokens.view(-1).tolist()
    else:
        _logit_idxs = list(graph.logit_tokens)
    
    
    logit_moves = [
        (to_uci(int(i)) if to_uci is not None else f"idx:{int(i)}")
        for i in _logit_idxs
    ]
    target_move = logit_moves[0] if logit_moves else None
    
    print(f'{target_move = }') 
    print(f'{graph.adjacency_matrix.shape = }')
    
    node_mask, edge_mask, cumulative_scores = (
        el.to(device) for el in prune_graph(graph, node_threshold, edge_threshold)
    )

    # tokenizer = AutoTokenizer.from_pretrained(graph.cfg.tokenizer_name)
    nodes = create_nodes(graph, node_mask, cumulative_scores, to_uci = to_uci)
    used_nodes, used_edges = create_used_nodes_and_edges(graph, nodes, edge_mask)
    model = build_model(
        graph=graph,
        used_nodes=used_nodes,
        used_edges=used_edges,
        slug=slug,
        sae_series=sae_series,
        node_threshold=node_threshold,
        # tokenizer=tokenizer,
        lorsa_analysis_name=lorsa_analysis_name,
        tc_analysis_name=tc_analysis_name,
        logit_moves = logit_moves,
        target_move = target_move,
    )

    # Write the output locally
    with open(os.path.join(output_path, f"{slug}.json"), "w") as f:
        f.write(model.model_dump_json(indent=2))
    logger.info(f"Graph data written to {output_path}")

    total_time_ms = (time.time() - total_start_time) * 1000
    logger.info(f"Total execution time: {total_time_ms=:.2f} ms")