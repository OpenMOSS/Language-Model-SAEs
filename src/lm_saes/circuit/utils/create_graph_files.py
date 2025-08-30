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
    def feature_node(cls, layer: int, pos: int, feat_idx: int,
                     influence: float | None = None, activation: float | None = None):
        """Create a TC feature node."""
        def cantor_pairing(x, y):
            return (x + y) * (x + y + 1) // 2 + y
        reverse_ctx_idx = 0
        # 采用原来“CLT 轨道”的可视化层号，保持与 logit 层 2*num_layers 的相对关系
        vis_layer = layer + 1
        return cls(
            node_id=f"{vis_layer}_{feat_idx}_{pos}",
            feature=cantor_pairing(vis_layer, feat_idx),
            layer=vis_layer,
            ctx_idx=pos,
            feature_type="transcoder",     # 原 "cross layer transcoder"
            jsNodeId=f"{vis_layer}_{feat_idx}-{reverse_ctx_idx}",
            influence=influence,
            activation=activation,
        )

    @classmethod
    def error_node(cls, layer: int, pos: int, influence: float | None = None):
        """Create a TC reconstruction error node."""
        reverse_ctx_idx = 0
        vis_layer = layer + 1
        return cls(
            node_id=f"{vis_layer}_error_{pos}",
            feature=-1,
            layer=vis_layer,
            ctx_idx=pos,
            feature_type="transcoder error",   # 统一为 TC error
            jsNodeId=f"{vis_layer}_{pos}-{reverse_ctx_idx}",
            influence=influence,
        )

    @classmethod
    def token_node(cls, pos: int, vocab_idx: int, influence: float | None = None):
        """Create a token (embedding) node."""
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
        pos: int,
        vocab_idx: int,
        token: str,
        num_layers: int,
        target_logit: bool = False,
        token_prob: float = 0.0,
    ):
        """Create a logit node."""
        # 只有 TC：feature/error 可视化层号是 1..num_layers，
        # 因此把 logit 放在最顶：num_layers + 1
        layer = num_layers + 1
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
    to_uci: Optional[Callable[[int], str]] = None,   # ← 新增
):
    """Create all nodes for the graph."""
    start_time = time.time()
    nodes: dict[int, Node] = {}

    n_features = len(graph.selected_features)              # 仅 TC 特征
    layers = graph.cfg.n_layers
    # 只有一类 error（TC error）：每层每位置一个
    error_end_idx = n_features + layers * graph.n_pos
    token_end_idx = error_end_idx + len(graph.input_tokens)
    # print(f'{n_features = }') # 38
    for node_idx in node_mask.nonzero().squeeze().tolist():
        if node_idx in range(n_features):
            # TC 特征：selected_features 映射到 tc_active_features 的索引
            orig_feature_idx = int(graph.selected_features[node_idx])
            layer, pos, feat_idx = graph.tc_active_features[orig_feature_idx].tolist()
            activation_val = graph.tc_activation_values[orig_feature_idx]
            nodes[node_idx] = Node.feature_node(
                layer=layer,
                pos=pos,
                feat_idx=feat_idx,
                influence=float(cumulative_scores[node_idx]),
                activation=float(activation_val),
            )

        elif node_idx in range(n_features, error_end_idx):
            # TC error：按 (layer, pos) 展开
            rel = node_idx - n_features
            layer, pos = divmod(rel, graph.n_pos)
            nodes[node_idx] = Node.error_node(
                layer=int(layer),
                pos=int(pos),
                influence=float(cumulative_scores[node_idx]),
            )

        elif node_idx in range(error_end_idx, token_end_idx):
            # print(f'{error_end_idx = }, {token_end_idx = }')
            # 输入 token（embedding）
            pos = node_idx - error_end_idx
            # print(f'{graph.input_tokens.shape = }, {pos = }')
            # vocab_idx = int(graph.input_tokens[pos])
            
            dummy_vocab_idx = pos
            nodes[node_idx] = Node.token_node(
                pos=int(pos),
                vocab_idx=dummy_vocab_idx,
                influence=float(cumulative_scores[node_idx]),
            )

        elif node_idx in range(token_end_idx, len(cumulative_scores)):
            # logit 节点（pos 固定为最后一个位置，vocab 来自 graph.logit_tokens）
            pos = node_idx - token_end_idx
            move_idx = int(graph.logit_tokens[pos])
            
            move_str = to_uci(move_idx) if to_uci is not None else f"idx:{move_idx}"
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


def create_used_nodes_and_edges(graph: Graph, nodes: dict[int, Node], edge_mask: torch.Tensor):
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
    used_nodes: List[Node],
    used_edges: List[dict],
    slug: str,
    sae_series: str,
    node_threshold: float,
    tc_analysis_name: str,
    logit_moves: List[str] | None = None,
    target_move: str | None = None,
):
    """Build the full model object."""
    start_time = time.time()

    print(f'in build_model:{logit_moves = }, {target_move = }')
    
    meta = Metadata(
        slug=slug,
        sae_series=sae_series,
        # prompt_tokens=[process_token(tokenizer.decode(int(t))) for t in graph.input_tokens],
        prompt_tokens = [process_chess_input(graph.input_string)],
        prompt=graph.input_string,
        node_threshold=node_threshold,
        tc_analysis_name=tc_analysis_name,
        logit_moves=logit_moves or [],       # ← 写入
        target_move=target_move,             # ← 写入
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
    output_path: str,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    sae_series: str | None = None,
    tc_analysis_name: str = "",
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
    # nodes = create_nodes(graph, node_mask, tokenizer, cumulative_scores)
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
        tc_analysis_name=tc_analysis_name,
        logit_moves = logit_moves,
        target_move = target_move,
    )

    # Write the output locally
    with open(os.path.join(output_path, f"{slug}.json"), "w") as f:
        f.write(model.model_dump_json(indent=2))
    logger.info(f"Graph data written to {output_path}")
    print(f"Graph data written to {output_path}")

    total_time_ms = (time.time() - total_start_time) * 1000
    logger.info(f"Total execution time: {total_time_ms=:.2f} ms")