import logging
import time
from typing import List

from transformers import AutoTokenizer

from lm_saes.circuit.graph import Graph, prune_graph

from .graph_file_utils import Metadata, Model, Node, process_token

logger = logging.getLogger(__name__)


def load_graph_data(file_path) -> Graph:
    """Load graph data from a PyTorch file."""
    start_time = time.time()
    graph = Graph.from_pt(file_path)
    time_ms = (time.time() - start_time) * 1000
    logger.info(f"Loading graph data: {time_ms=:.2f} ms")
    return graph


def create_nodes(graph: Graph, node_mask, tokenizer, cumulative_scores, use_lorsa, clt_names, lorsa_names):
    """Create all nodes for the graph."""
    start_time = time.time()

    nodes = {}

    n_features = len(graph.selected_features)
    layers = graph.cfg.n_layers
    error_end_idx = n_features + 2 * graph.n_pos * layers if use_lorsa else n_features + graph.n_pos * layers
    token_end_idx = error_end_idx + len(graph.input_tokens)

    for node_idx in node_mask.nonzero().squeeze().tolist():
        if node_idx in range(n_features):
            orig_feature_idx = graph.selected_features[node_idx]
            if use_lorsa:
                is_lorsa = bool(orig_feature_idx < len(graph.lorsa_active_features))
                if is_lorsa:
                    feature_tensor = graph.lorsa_active_features[orig_feature_idx]
                    layer, pos, feat_idx = feature_tensor.tolist()
                    interested_activation = graph.lorsa_activation_values
                    sae_name = lorsa_names[layer]
                else:
                    orig_feature_idx = orig_feature_idx - len(graph.lorsa_active_features)
                    feature_tensor = graph.clt_active_features[orig_feature_idx]
                    layer, pos, feat_idx = feature_tensor.tolist()
                    interested_activation = graph.clt_activation_values
                    sae_name = clt_names[layer]
            else:
                is_lorsa = False
                feature_tensor = graph.clt_active_features[orig_feature_idx]
                layer, pos, feat_idx = feature_tensor.tolist()
                interested_activation = graph.clt_activation_values
                sae_name = clt_names[layer]
            nodes[node_idx] = Node.feature_node(
                layer,
                pos,
                feat_idx,
                is_lorsa=is_lorsa,
                sae_name=sae_name,
                influence=cumulative_scores[node_idx],
                activation=interested_activation[orig_feature_idx],
                lorsa_pattern=graph.lorsa_pattern[node_idx],
                qk_tracing_results=(
                    graph.qk_tracing_results.get(orig_feature_idx.item(), None)
                    if graph.qk_tracing_results is not None
                    else None
                ),
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
                pos,
                graph.input_tokens[pos],
                token=process_token(tokenizer.decode(graph.input_tokens[pos])),
                influence=cumulative_scores[node_idx],
            )
        elif node_idx in range(token_end_idx, len(cumulative_scores)):
            pos = node_idx - token_end_idx

            # Check if this is feature tracing (logit_tokens contains feature info instead of real tokens)
            logit_token = graph.logit_tokens[pos]
            if isinstance(logit_token, (tuple, list)):
                is_feature_tracing = len(logit_token) == 4
            elif hasattr(logit_token, "shape"):
                is_feature_tracing = len(logit_token.shape) > 0 and logit_token.shape[0] == 4
            else:
                is_feature_tracing = False

            if is_feature_tracing:
                continue

            # Normal logit case - ensure vocab_idx is an integer
            vocab_idx = (
                graph.logit_tokens[pos] if isinstance(graph.logit_tokens[pos], int) else graph.logit_tokens[pos].item()
            )
            logger.info("create a logit node")
            nodes[node_idx] = Node.logit_node(
                pos=graph.n_pos - 1,
                vocab_idx=vocab_idx,
                token=process_token(tokenizer.decode(vocab_idx)),
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
        node for node in nodes.values() if node.node_id in connected_ids or node.feature_type in ["embedding", "logit"]
    ]
    nodes_after = len(used_nodes)
    logger.info(f"Filtered {nodes_before - nodes_after} nodes")

    time_ms = (time.time() - start_time) * 1000
    logger.info(f"Creating used nodes and edges: {time_ms=:.2f} ms")
    logger.info(f"Used nodes: {len(used_nodes)}, Used edges: {len(used_edges)}")

    return used_nodes, used_edges


def append_qk_tracing_results(graph: Graph, used_nodes: List[Node], clt_names, lorsa_names):
    """Append QK tracing results to the graph."""
    existing_nodes = set(used_nodes)
    from_qk_tracing_nodes = set()
    for node in used_nodes:
        if node.qk_tracing_results is not None:
            from_qk_tracing_nodes.update(node.qk_tracing_results.get_nodes())

    nodes_to_add = from_qk_tracing_nodes - existing_nodes
    for node in nodes_to_add:
        if node.feature_type == "lorsa":
            node.sae_name = lorsa_names[node.layer // 2]
        elif node.feature_type == "cross layer transcoder":
            node.sae_name = clt_names[node.layer // 2]
        used_nodes.append(node)

    for node in used_nodes:
        if node.qk_tracing_results is not None:
            node.qk_tracing_results.stringify_nodes()
    return used_nodes


def build_model(
    graph: Graph,
    used_nodes,
    used_edges,
    tokenizer,
):
    """Build the full model object."""
    start_time = time.time()

    meta = Metadata(
        prompt_tokens=[process_token(tokenizer.decode(t)) for t in graph.input_tokens],
        prompt=graph.input_string,
    )

    full_model = Model(
        metadata=meta,
        nodes=used_nodes,
        links=used_edges,
    )

    time_ms = (time.time() - start_time) * 1000
    logger.info(f"Building model: {time_ms=:.2f} ms")

    return full_model


def serialize_graph(
    graph: Graph,
    *,
    node_threshold=0.8,
    edge_threshold=0.98,
    use_lorsa: bool = True,
    clt_names: list[str],
    lorsa_names: list[str] | None = None,
):
    """Serialize the graph to a JSON object."""
    node_mask, edge_mask, cumulative_scores = prune_graph(graph, node_threshold, edge_threshold)

    tokenizer = AutoTokenizer.from_pretrained(graph.cfg.tokenizer_name)

    nodes = create_nodes(
        graph,
        node_mask,
        tokenizer,
        cumulative_scores,
        use_lorsa=use_lorsa,
        clt_names=clt_names,
        lorsa_names=lorsa_names,
    )
    used_nodes, used_edges = create_used_nodes_and_edges(graph, nodes, edge_mask)
    if use_lorsa:
        used_nodes = append_qk_tracing_results(graph, used_nodes, clt_names, lorsa_names)
    model = build_model(
        graph,
        used_nodes,
        used_edges,
        tokenizer,
    )
    return model.model_dump()
