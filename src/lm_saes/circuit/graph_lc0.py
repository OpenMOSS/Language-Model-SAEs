from typing import List, NamedTuple, Optional, Union

import torch
from transformer_lens import HookedTransformerConfig



class Graph:
    input_string: str
    input_tokens: torch.Tensor
    logit_tokens: torch.Tensor
    logit_probabilities: torch.Tensor
    lorsa_active_features: torch.Tensor
    lorsa_activation_matrix: torch.Tensor
    tc_active_features: torch.Tensor
    tc_activation_values: torch.Tensor
    adjacency_matrix: torch.Tensor
    selected_features: torch.Tensor
    cfg: HookedTransformerConfig
    sae_series: Optional[Union[str, List[str]]]
    slug: str

    def __init__(
        self,
        input_string: str,
        input_tokens: torch.Tensor,
        logit_tokens: torch.Tensor,
        logit_probabilities: torch.Tensor,
        lorsa_active_features: torch.Tensor,
        lorsa_activation_values: torch.Tensor,
        tc_active_features: torch.Tensor,
        tc_activation_values: torch.Tensor,
        selected_features: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        cfg: HookedTransformerConfig,
        slug: str = "untitled",
        sae_series: Optional[Union[str, List[str]]] = None,
    ):
        """
        A graph object containing the adjacency matrix describing the direct effect of each
        node on each other. Nodes are either non-zero transcoder features (TC), transcoder errors,
        tokens, or logits. They are stored in the order [tc_active_features[0], ..., error[layer0][position0], ...,
        tokens[0], ..., logits[top-1 logit], ...].

        Args:
            input_string (str): The input string attributed.
            input_tokens (torch.Tensor): The input tokens attributed.
            logit_tokens (torch.Tensor): The logit tokens attributed from.
            logit_probabilities (torch.Tensor): The probabilities of each logit token, given the input string.
            tc_active_features (torch.Tensor): Indices (layer, pos, feature_idx) of non-zero TC features.
            tc_activation_values (torch.Tensor): Activation values for TC features.
            selected_features (torch.Tensor): Indices of selected features (for pruning, etc).
            adjacency_matrix (torch.Tensor): The adjacency matrix.
            cfg (HookedTransformerConfig): The cfg of the model.
            sae_series (Optional[Union[str,List[str]]], optional): The identifier of the transcoders used in the graph.
        """
        self.input_string = input_string
        self.adjacency_matrix = adjacency_matrix
        self.cfg = cfg
        self.n_pos = len(input_tokens)
        self.lorsa_active_features = lorsa_active_features
        self.lorsa_activation_values = lorsa_activation_values
        self.tc_active_features = tc_active_features
        self.tc_activation_values = tc_activation_values
        self.logit_tokens = logit_tokens
        self.logit_probabilities = logit_probabilities
        self.input_tokens = input_tokens
        if sae_series is None:
            print("Graph loaded without sae_series to identify it. Uploading will not be possible.")
        self.sae_series = sae_series
        self.selected_features = selected_features
        self.slug = slug

    def to(self, device):
        """Send all relevant tensors to the device (cpu, cuda, etc.)"""
        self.adjacency_matrix = self.adjacency_matrix.to(device)
        self.lorsa_active_features = self.lorsa_active_features.to(device)
        self.lorsa_activation_values = self.lorsa_activation_values.to(device)
        self.tc_active_features = self.tc_active_features.to(device)
        self.tc_activation_values = self.tc_activation_values.to(device)
        self.logit_tokens = self.logit_tokens.to(device)
        self.logit_probabilities = self.logit_probabilities.to(device)
        self.selected_features = self.selected_features.to(device)
        self.input_tokens = self.input_tokens.to(device)

    def to_pt(self, path: str):
        """Saves the graph at the given path"""
        d = {
            "input_string": self.input_string,
            "adjacency_matrix": self.adjacency_matrix,
            "cfg": self.cfg,
            "lorsa_active_features": self.lorsa_active_features,
            "lorsa_activation_values": self.lorsa_activation_values,
            "tc_active_features": self.tc_active_features,
            "tc_activation_values": self.tc_activation_values,
            "logit_tokens": self.logit_tokens,
            "logit_probabilities": self.logit_probabilities,
            "input_tokens": self.input_tokens,
            "selected_features": self.selected_features,
            "sae_series": self.sae_series,
            "slug": self.slug,
        }
        torch.save(d, path)

    @staticmethod
    def from_pt(path: str, map_location="cpu") -> "Graph":
        """
        Load a graph from a .pt file at the given path.
        Backward-compat:
          - If 'clt_*' keys exist, they are mapped to 'tc_*'.
          - Any 'lorsa_*' keys are ignored.
          - If neither 'tc_*' nor 'clt_*' exist, empty TC tensors are created.
        """
        d = torch.load(path, weights_only=False, map_location=map_location)

        # Map CLT -> TC if present
        if "tc_active_features" not in d:
            if "clt_active_features" in d:
                d["tc_active_features"] = d.pop("clt_active_features")
            else:
                # default empty: assume indices in shape [0, 3] (layer, pos, feature)
                d["tc_active_features"] = torch.empty(0, 3, dtype=torch.long)
        if "tc_activation_values" not in d:
            if "clt_activation_values" in d:
                d["tc_activation_values"] = d.pop("clt_activation_values")
            else:
                d["tc_activation_values"] = torch.empty(0)

        # Drop LORSA if present
        # d.pop("lorsa_active_features", None)
        # d.pop("lorsa_activation_values", None)

        # Remove any lingering CLT keys
        d.pop("clt_active_features", None)
        d.pop("clt_activation_values", None)

        return Graph(**d)


def normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    normalized = matrix.abs()
    return normalized / normalized.sum(dim=1, keepdim=True).clamp(min=1e-10)


def compute_influence(A: torch.Tensor, logit_weights: torch.Tensor, max_iter: int = 1000):
    # Normally we calculate total influence B using A + A^2 + ... or (I - A)^-1 - I,
    # and do logit_weights @ B
    # But it's faster / more efficient to compute logit_weights @ A + logit_weights @ A^2
    # as follows:

    current_influence = logit_weights @ A
    influence = current_influence
    iterations = 0
    while current_influence.any():
        if iterations >= max_iter:
            raise RuntimeError(
                f"Influence computation failed to converge after {iterations} iterations"
            )
        current_influence = current_influence @ A
        influence += current_influence
        iterations += 1
    return influence


def compute_node_influence(adjacency_matrix: torch.Tensor, logit_weights: torch.Tensor):
    return compute_influence(normalize_matrix(adjacency_matrix), logit_weights)


def compute_edge_influence(pruned_matrix: torch.Tensor, logit_weights: torch.Tensor):
    normalized_pruned = normalize_matrix(pruned_matrix)
    pruned_influence = compute_influence(normalized_pruned, logit_weights)
    pruned_influence += logit_weights
    edge_scores = normalized_pruned * pruned_influence[:, None]
    return edge_scores


def find_threshold(scores: torch.Tensor, threshold: float):
    # Find score threshold that keeps the desired fraction of total influence
    sorted_scores = torch.sort(scores, descending=True).values
    cumulative_score = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)
    threshold_index = torch.searchsorted(cumulative_score, threshold)
    # make sure we don't go out of bounds (only really happens at threshold=1.0)
    threshold_index = min(threshold_index, len(cumulative_score) - 1)
    return sorted_scores[threshold_index]


class PruneResult(NamedTuple):
    node_mask: torch.Tensor  # Boolean tensor indicating which nodes to keep
    edge_mask: torch.Tensor  # Boolean tensor indicating which edges to keep
    cumulative_scores: torch.Tensor  # Tensor of cumulative influence scores for each node


def prune_graph(
    graph: Graph, node_threshold: float = 0.8, edge_threshold: float = 0.98
) -> PruneResult:
    """Prunes a graph by removing nodes and edges with low influence on the output logits.

    Args:
        graph: The graph to prune
        node_threshold: Keep nodes that contribute to this fraction of total influence
        edge_threshold: Keep edges that contribute to this fraction of total influence

    Returns:
        Tuple containing:
        - node_mask: Boolean tensor indicating which nodes to keep
        - edge_mask: Boolean tensor indicating which edges to keep
        - cumulative_scores: Tensor of cumulative influence scores for each node
    """

    if node_threshold > 1.0 or node_threshold < 0.0:
        raise ValueError("node_threshold must be between 0.0 and 1.0")
    if edge_threshold > 1.0 or edge_threshold < 0.0:
        raise ValueError("edge_threshold must be between 0.0 and 1.0")

    # Extract dimensions
    n_tokens = len(graph.input_tokens)
    n_logits = len(graph.logit_tokens)
    n_features = len(graph.selected_features)  # now refers to TC features only
    
    logit_weights = torch.zeros(
        graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device
    )
    # print(f'{logit_weights = }')
    # print(f'{n_logits = }')
    logit_weights[-n_logits:] = graph.logit_probabilities

    # Calculate node influence and apply threshold
    node_influence = compute_node_influence(graph.adjacency_matrix, logit_weights)
    node_mask = node_influence >= find_threshold(node_influence, node_threshold)
    # Always keep tokens and logits
    node_mask[-n_logits - n_tokens:] = True

    # Create pruned matrix with selected nodes
    pruned_matrix = graph.adjacency_matrix.clone()
    pruned_matrix[~node_mask] = 0
    pruned_matrix[:, ~node_mask] = 0

    # Calculate edge influence and apply threshold
    edge_scores = compute_edge_influence(pruned_matrix, logit_weights)
    edge_mask = edge_scores >= find_threshold(edge_scores.flatten(), edge_threshold)
    old_node_mask = node_mask.clone()
    # Ensure feature and error nodes have outgoing edges
    node_mask[: -n_logits - n_tokens] &= edge_mask[:, : -n_logits - n_tokens].any(0)
    # Ensure **feature** nodes (TC) have incoming edges
    node_mask[:n_features] &= edge_mask[:n_features].any(1)

    # Iteratively prune until stable
    while not torch.all(node_mask == old_node_mask):
        old_node_mask[:] = node_mask
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False

        node_mask[: -n_logits - n_tokens] &= edge_mask[:, : -n_logits - n_tokens].any(0)
        node_mask[:n_features] &= edge_mask[:n_features].any(1)

    # Cumulative influence curve
    sorted_scores, sorted_indices = torch.sort(node_influence, descending=True)
    cumulative_scores = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)
    final_scores = torch.zeros_like(node_influence)
    final_scores[sorted_indices] = cumulative_scores

    return PruneResult(node_mask, edge_mask, final_scores)
