from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel


class Metadata(BaseModel):
    prompt_tokens: List[str]
    prompt: str

    schema_version: int | None = 1


class Node(BaseModel):
    node_id: str
    feature: int | None = None
    layer: int
    ctx_idx: int
    feature_type: str
    token: str | None = None
    token_prob: float | None = None
    sae_name: str | None = None
    is_target_logit: bool = False
    influence: float | None = None
    activation: float | None = None
    lorsa_pattern: list | None = None
    qk_tracing_results: Optional["QKTracingResults"] = None
    is_from_qk_tracing: bool = False

    @classmethod
    def feature_node(
        cls,
        layer,
        pos,
        feat_idx,
        is_lorsa,
        influence=None,
        activation=None,
        lorsa_pattern=None,
        qk_tracing_results=None,
        sae_name=None,
        is_from_qk_tracing=False,
    ):
        """Create a feature node."""

        # Ensure all parameters are scalars
        layer = int(layer) if not isinstance(layer, int) else layer
        feat_idx = int(feat_idx) if not isinstance(feat_idx, int) else feat_idx
        pos = int(pos) if not isinstance(pos, int) else pos
        is_lorsa = bool(is_lorsa)
        layer = 2 * layer + int(not is_lorsa)
        return cls(
            node_id=f"{layer}_{feat_idx}_{pos}",
            feature=feat_idx,
            layer=layer,
            ctx_idx=pos,
            feature_type="lorsa" if is_lorsa else "cross layer transcoder",
            influence=influence,
            activation=activation,
            lorsa_pattern=lorsa_pattern.tolist() if lorsa_pattern is not None else None,
            qk_tracing_results=qk_tracing_results,
            sae_name=sae_name,
            is_from_qk_tracing=is_from_qk_tracing,
        )

    @classmethod
    def token_node(cls, pos, vocab_idx, token, influence=None):
        """Create a token node."""
        return cls(
            node_id=f"E_{vocab_idx}_{pos}",
            layer=-1,
            ctx_idx=pos,
            feature_type="embedding",
            influence=influence,
            token=token,
        )

    @classmethod
    def error_node(cls, layer, pos, is_lorsa, influence=None, is_from_qk_tracing=False):
        """Create an error node."""
        return cls(
            node_id=f"{layer}_error_{pos}",
            layer=layer,
            ctx_idx=pos,
            feature_type="lorsa error" if is_lorsa else "mlp reconstruction error",
            influence=influence,
            is_from_qk_tracing=is_from_qk_tracing,
        )

    @classmethod
    def bias_node(cls, layer, pos, bias_name, influence=None, is_from_qk_tracing=False):
        """Create a bias node."""
        return cls(
            node_id=f"{layer}_{bias_name}_{pos}",
            layer=layer,
            ctx_idx=pos,
            feature_type="bias",
            influence=influence,
            is_from_qk_tracing=is_from_qk_tracing,
        )

    @classmethod
    def logit_node(
        cls,
        pos,
        vocab_idx,
        token,
        num_layers,
        target_logit,
        token_prob,
    ):
        """Create a logit node."""
        layer = 2 * num_layers
        return cls(
            node_id=f"{layer}_{vocab_idx}_{pos}",
            feature=vocab_idx,
            layer=layer,
            ctx_idx=pos,
            feature_type="logit",
            token=token,
            token_prob=token_prob,
            is_target_logit=target_logit,
        )

    def __eq__(self, other):
        return self.node_id == other.node_id

    def __hash__(self):
        return hash(self.node_id)


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


@dataclass
class QKTracingResults:
    """Results from QK tracing analysis.

    pair_wise_contributors: List of tuples containing (q_node, k_node, score) for top pairwise contributors.
    top_q_marginal_contributors: List of tuples containing (q_node, score) for top Q marginal contributors.
    top_k_marginal_contributors: List of tuples containing (k_node, score) for top K marginal contributors.
    """

    NodeType = Node | str

    pair_wise_contributors: list[tuple[NodeType, NodeType, float]]
    top_q_marginal_contributors: list[tuple[NodeType, float]]
    top_k_marginal_contributors: list[tuple[NodeType, float]]

    def get_nodes(self) -> set[Node]:
        all_relevant_nodes = set()
        for q_node, k_node, _ in self.pair_wise_contributors:
            all_relevant_nodes.add(q_node)
            all_relevant_nodes.add(k_node)
        for q_node, _ in self.top_q_marginal_contributors:
            all_relevant_nodes.add(q_node)
        for k_node, _ in self.top_k_marginal_contributors:
            all_relevant_nodes.add(k_node)
        return all_relevant_nodes

    def stringify_nodes(self):
        assert all(
            isinstance(q_node, Node) and isinstance(k_node, Node) for q_node, k_node, _ in self.pair_wise_contributors
        ) or all(
            isinstance(q_node, str) and isinstance(k_node, str) for q_node, k_node, _ in self.pair_wise_contributors
        )
        assert all(isinstance(q_node, Node) for q_node, _ in self.top_q_marginal_contributors) or all(
            isinstance(q_node, str) for q_node, _ in self.top_q_marginal_contributors
        )
        assert all(isinstance(k_node, Node) for k_node, _ in self.top_k_marginal_contributors) or all(
            isinstance(k_node, str) for k_node, _ in self.top_k_marginal_contributors
        )

        if len(self.pair_wise_contributors) > 0 and isinstance(self.pair_wise_contributors[0][0], str):
            return self

        self.pair_wise_contributors = [
            (q_node.node_id, k_node.node_id, score) for q_node, k_node, score in self.pair_wise_contributors
        ]
        self.top_q_marginal_contributors = [
            (q_node.node_id, score) for q_node, score in self.top_q_marginal_contributors
        ]
        self.top_k_marginal_contributors = [
            (k_node.node_id, score) for k_node, score in self.top_k_marginal_contributors
        ]
        return self
