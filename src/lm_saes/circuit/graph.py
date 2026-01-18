from typing import Any, List, NamedTuple, Optional, Union

import numpy as np
import torch
from transformer_lens import HookedTransformerConfig

from .utils.attn_scores_attribution import QKTracingResults


class Graph:
    input_string: str
    input_tokens: torch.Tensor
    logit_tokens: torch.Tensor
    lorsa_active_features: torch.Tensor
    lorsa_activation_values: torch.Tensor
    clt_active_features: torch.Tensor
    clt_activation_values: torch.Tensor
    adjacency_matrix: torch.Tensor
    qk_tracing_results: QKTracingResults
    selected_features: torch.Tensor
    logit_probabilities: torch.Tensor
    cfg: HookedTransformerConfig
    sae_series: Optional[Union[str, List[str]]]
    slug: str
    use_lorsa: bool
    lorsa_pattern: torch.Tensor
    z_pattern: torch.Tensor

    def __init__(
        self,
        input_string: str,
        input_tokens: torch.Tensor,
        logit_tokens: torch.Tensor,
        logit_probabilities: torch.Tensor,
        lorsa_active_features: torch.Tensor,
        lorsa_activation_values: torch.Tensor,
        clt_active_features: torch.Tensor,
        clt_activation_values: torch.Tensor,
        selected_features: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        qk_tracing_results: QKTracingResults,
        cfg: HookedTransformerConfig,
        slug: str = "untitled",
        sae_series: Optional[Union[str, List[str]]] = None,
        use_lorsa: bool = True,
        lorsa_pattern: torch.Tensor = None,
        z_pattern: torch.Tensor = None,
    ):
        """
        A graph object containing the adjacency matrix describing the direct effect of each
        node on each other. Nodes are either non-zero transcoder features (LORSA or CLT), transcoder errors,
        tokens, or logits. They are stored in the order [lorsa_active_features[0], ..., clt_active_features[-1],
        error[layer0][position0], ..., tokens[0], ..., logits[top-1 logit], ...].

        Args:
            input_string (str): The input string attributed.
            input_tokens (torch.Tensor): The input tokens attributed.
            logit_tokens (torch.Tensor): The logit tokens attributed from.
            logit_probabilities (torch.Tensor): The probabilities of each logit token, given the input string.
            lorsa_active_features (torch.Tensor): Indices (layer, pos, feature_idx) of non-zero LORSA features.
            lorsa_activation_values (torch.Tensor): Activation values for LORSA features.
            clt_active_features (torch.Tensor): Indices (layer, pos, feature_idx) of non-zero CLT features.
            clt_activation_values (torch.Tensor): Activation values for CLT features.
            selected_features (torch.Tensor): Indices of selected features (for pruning, etc).
            adjacency_matrix (torch.Tensor): The adjacency matrix.
            qk_tracing_results (QKTracingResults): The QK tracing results.
            cfg (HookedTransformerConfig): The cfg of the model.
            sae_series (Optional[Union[str,List[str]]], optional): The identifier of the transcoders used in the graph.
        """
        self.input_string = input_string
        self.adjacency_matrix = adjacency_matrix
        self.cfg = cfg
        self.n_pos = len(input_tokens)
        self.lorsa_active_features = lorsa_active_features
        self.lorsa_activation_values = lorsa_activation_values
        self.clt_active_features = clt_active_features
        self.clt_activation_values = clt_activation_values
        self.logit_tokens = logit_tokens
        self.logit_probabilities = logit_probabilities
        self.input_tokens = input_tokens
        self.use_lorsa = use_lorsa
        self.lorsa_pattern = lorsa_pattern
        self.z_pattern = z_pattern
        self.qk_tracing_results = qk_tracing_results
        if sae_series is None:
            print("Graph loaded without sae_series to identify it. Uploading will not be possible.")
        self.sae_series = sae_series
        self.selected_features = selected_features
        self.slug = slug

    def __repr__(self) -> str:
        """Return a string representation of the graph with basic information."""
        n_tokens = len(self.input_tokens)
        n_logits = len(self.logit_tokens)
        n_selected_features = len(self.selected_features)

        # Count LORSA features if used
        n_lorsa_features = (
            len(self.lorsa_active_features) if self.use_lorsa and self.lorsa_active_features is not None else 0
        )

        # Count CLT features
        n_clt_features = len(self.clt_active_features) if self.clt_active_features is not None else 0

        # Get model info
        model_name = getattr(self.cfg, "model_name", "Unknown")
        n_layers = getattr(self.cfg, "n_layers", "Unknown")

        # Truncate input string if too long
        input_preview = self.input_string

        # Format SAE series info
        sae_info = str(self.sae_series) if self.sae_series is not None else "None"

        # Get adjacency matrix shape
        adj_shape = tuple(self.adjacency_matrix.shape)

        lines = [
            f"Graph(slug='{self.slug}')",
            f"  Input: '{input_preview}'",
            f"  Model: {model_name} ({n_layers} layers)",
            f"  Tokens: {n_tokens}",
            f"  Logits: {n_logits}",
            f"  Selected features: {n_selected_features}",
            f"  Lorsa features: {n_lorsa_features}" + (" (enabled)" if self.use_lorsa else " (disabled)"),
            f"  CLT features: {n_clt_features}",
            f"  Adjacency matrix: {adj_shape}",
            f"  SAE series: {sae_info}",
        ]
        return "\n".join(lines)

    def to(self, device):
        """Send all relevant tensors to the device (cpu, cuda, etc.)"""
        self.adjacency_matrix = self.adjacency_matrix.to(device)
        self.lorsa_active_features = self.lorsa_active_features.to(device) if self.use_lorsa else None
        self.lorsa_activation_values = self.lorsa_activation_values.to(device) if self.use_lorsa else None
        self.clt_active_features = self.clt_active_features.to(device)
        self.clt_activation_values = self.clt_activation_values.to(device)
        self.logit_tokens = self.logit_tokens.to(device)
        self.logit_probabilities = self.logit_probabilities.to(device)
        self.selected_features = self.selected_features.to(device)
        self.input_tokens = self.input_tokens.to(device)
        self.lorsa_pattern = self.lorsa_pattern.to(device)
        self.z_pattern = self.z_pattern.to(device)

    def to_pt(self, path: str):
        """Saves the graph at the given path"""
        d = {
            "input_string": self.input_string,
            "adjacency_matrix": self.adjacency_matrix,
            "cfg": self.cfg,
            "lorsa_active_features": self.lorsa_active_features,
            "lorsa_activation_values": self.lorsa_activation_values,
            "clt_active_features": self.clt_active_features,
            "clt_activation_values": self.clt_activation_values,
            "logit_tokens": self.logit_tokens,
            "logit_probabilities": self.logit_probabilities,
            "input_tokens": self.input_tokens,
            "selected_features": self.selected_features,
            "sae_series": self.sae_series,
            "slug": self.slug,
            "lorsa_pattern": self.lorsa_pattern,
            "z_pattern": self.z_pattern,
        }
        torch.save(d, path)

    @staticmethod
    def from_pt(path: str, map_location="cpu") -> "Graph":
        """Load a graph (saved using graph.to_pt) from a .pt file at the given path."""
        d = torch.load(path, weights_only=False, map_location=map_location)
        return Graph(**d)

    def to_dict(self) -> dict[str, Any]:
        """Convert Graph to a dictionary with numpy arrays for GridFS storage.

        This method converts all tensors to numpy arrays and serializes complex
        objects like cfg and qk_tracing_results using pickle.

        Returns:
            Dictionary representation of the Graph suitable for MongoDB/GridFS storage.
        """
        import pickle

        def tensor_to_numpy(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
            if t is None:
                return None
            return t.cpu().numpy()

        return {
            "input_string": self.input_string,
            "adjacency_matrix": tensor_to_numpy(self.adjacency_matrix),
            "cfg": pickle.dumps(self.cfg),  # Serialize HookedTransformerConfig
            "lorsa_active_features": tensor_to_numpy(self.lorsa_active_features),
            "lorsa_activation_values": tensor_to_numpy(self.lorsa_activation_values),
            "clt_active_features": tensor_to_numpy(self.clt_active_features),
            "clt_activation_values": tensor_to_numpy(self.clt_activation_values),
            "logit_tokens": tensor_to_numpy(self.logit_tokens),
            "logit_probabilities": tensor_to_numpy(self.logit_probabilities),
            "input_tokens": tensor_to_numpy(self.input_tokens),
            "selected_features": tensor_to_numpy(self.selected_features),
            "sae_series": self.sae_series,
            "slug": self.slug,
            "use_lorsa": self.use_lorsa,
            "lorsa_pattern": tensor_to_numpy(self.lorsa_pattern),
            "z_pattern": tensor_to_numpy(self.z_pattern),
            "qk_tracing_results": pickle.dumps(self.qk_tracing_results),  # Serialize QKTracingResults
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Graph":
        """Reconstruct a Graph from a dictionary (created by to_dict).

        Args:
            d: Dictionary representation from to_dict().

        Returns:
            Reconstructed Graph object.
        """
        import pickle

        def numpy_to_tensor(arr: Optional[np.ndarray]) -> Optional[torch.Tensor]:
            if arr is None:
                return None
            return torch.from_numpy(arr)

        cfg = pickle.loads(d["cfg"])
        qk_tracing_results = pickle.loads(d["qk_tracing_results"])

        return Graph(
            input_string=d["input_string"],
            input_tokens=numpy_to_tensor(d["input_tokens"]),
            logit_tokens=numpy_to_tensor(d["logit_tokens"]),
            logit_probabilities=numpy_to_tensor(d["logit_probabilities"]),
            lorsa_active_features=numpy_to_tensor(d["lorsa_active_features"]),
            lorsa_activation_values=numpy_to_tensor(d["lorsa_activation_values"]),
            clt_active_features=numpy_to_tensor(d["clt_active_features"]),
            clt_activation_values=numpy_to_tensor(d["clt_activation_values"]),
            selected_features=numpy_to_tensor(d["selected_features"]),
            adjacency_matrix=numpy_to_tensor(d["adjacency_matrix"]),
            qk_tracing_results=qk_tracing_results,
            cfg=cfg,
            slug=d.get("slug", "untitled"),
            sae_series=d.get("sae_series"),
            use_lorsa=d.get("use_lorsa", True),
            lorsa_pattern=numpy_to_tensor(d.get("lorsa_pattern")),
            z_pattern=numpy_to_tensor(d.get("z_pattern")),
        )


def normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    normalized = matrix.abs()
    return normalized / normalized.sum(dim=1, keepdim=True).clamp(min=1e-10)


def compute_influence(A: torch.Tensor, logit_weights: torch.Tensor, max_iter: int = 10000):
    # Normally we calculate total influence B using A + A^2 + ... or (I - A)^-1 - I,
    # and do logit_weights @ B
    # But it's faster / more efficient to compute logit_weights @ A + logit_weights @ A^2
    # as follows:

    current_influence = logit_weights @ A
    influence = current_influence
    iterations = 0
    while current_influence.any():
        if iterations >= max_iter:
            raise RuntimeError(f"Influence computation failed to converge after {iterations} iterations")
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


def prune_graph(graph: Graph, node_threshold: float = 0.8, edge_threshold: float = 0.98) -> PruneResult:
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
    n_features = len(graph.selected_features)

    logit_weights = torch.zeros(graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device)
    logit_weights[-n_logits:] = graph.logit_probabilities

    # Calculate node influence and apply threshold
    node_influence = compute_node_influence(graph.adjacency_matrix, logit_weights)
    node_mask = node_influence >= find_threshold(node_influence, node_threshold)
    # Always keep tokens and logits
    node_mask[-n_logits - n_tokens :] = True

    # Create pruned matrix with selected nodes
    pruned_matrix = graph.adjacency_matrix.clone()
    pruned_matrix[~node_mask] = 0
    pruned_matrix[:, ~node_mask] = 0
    # we could also do iterative pruning here (see below)

    # Calculate edge influence and apply threshold
    edge_scores = compute_edge_influence(pruned_matrix, logit_weights)
    edge_mask = edge_scores >= find_threshold(edge_scores.flatten(), edge_threshold)
    old_node_mask = node_mask.clone()
    # Ensure feature and error nodes have outgoing edges
    node_mask[: -n_logits - n_tokens] &= edge_mask[:, : -n_logits - n_tokens].any(0)
    # Ensure feature nodes have incoming edges
    node_mask[:n_features] &= edge_mask[:n_features].any(1)

    # iteratively prune until all nodes missing incoming / outgoing edges are gone
    # (each pruning iteration potentially opens up new candidates for pruning)
    # this should not take more than n_layers + 1 iterations
    while not torch.all(node_mask == old_node_mask):
        old_node_mask[:] = node_mask
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False

        # Ensure feature and error nodes have outgoing edges
        node_mask[: -n_logits - n_tokens] &= edge_mask[:, : -n_logits - n_tokens].any(0)
        # Ensure feature nodes have incoming edges
        node_mask[:n_features] &= edge_mask[:n_features].any(1)

    # Calculate cumulative influence scores
    sorted_scores, sorted_indices = torch.sort(node_influence, descending=True)
    cumulative_scores = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)
    final_scores = torch.zeros_like(node_influence)
    final_scores[sorted_indices] = cumulative_scores

    return PruneResult(node_mask, edge_mask, final_scores)
