"""Forward-only global-weight and inhibitory scans for the local model API."""

import re
from collections.abc import Iterable

import torch

from llamascopium.backend.language_model import TransformerLensLanguageModel
from llamascopium.circuits.global_weights.inhibitory import (
    Feature,
    InhibitoryConnection,
    InhibitoryWeights,
    compute_inhibitory_global_weights,
)
from llamascopium.circuits.global_weights.weights import GlobalConnection, GlobalWeights, compute_global_weights
from llamascopium.circuits.utils.batched_features import BatchedFeatures, ConnectedFeatures
from llamascopium.models.lorsa import LowRankSparseAttention
from llamascopium.models.sae import SparseAutoEncoder


def _feature_batch(features: list[Feature], device: torch.device) -> BatchedFeatures:
    layers = []
    is_lorsa = []
    for name, _ in features:
        match = re.fullmatch(r"blocks\.(\d+)\.hook_(attn|mlp)_out", name)
        if match is None:
            raise ValueError(f"Unsupported feature hook: {name}")
        layers.append(int(match[1]))
        is_lorsa.append(match[2] == "attn")
    return BatchedFeatures(
        layer=torch.tensor(layers, device=device, dtype=torch.long),
        index=torch.tensor([index for _, index in features], device=device, dtype=torch.long),
        is_lorsa=torch.tensor(is_lorsa, device=device, dtype=torch.bool),
    )


def _connected_features(
    upstream: list[GlobalConnection], downstream: list[GlobalConnection], device: torch.device
) -> ConnectedFeatures:
    return ConnectedFeatures(
        upstream_features=_feature_batch([(edge.source, edge.source_feature) for edge in upstream], device),
        downstream_features=_feature_batch([(edge.target, edge.target_feature) for edge in downstream], device),
        upstream_values=torch.tensor([edge.weight for edge in upstream], device=device),
        downstream_values=torch.tensor([edge.weight for edge in downstream], device=device),
    ).sort()


class ForwardOnlyReplacementModel:
    """Use the existing SAE hooks and forward-only weight accumulators."""

    def __init__(
        self,
        model: TransformerLensLanguageModel,
        dictionaries: list[SparseAutoEncoder | LowRankSparseAttention],
    ) -> None:
        self.model = model
        self.dictionaries = dictionaries
        self.interested_features: list[Feature] = []
        self.global_weights: GlobalWeights | None = None
        self.inhibitory_weights: InhibitoryWeights | None = None

    def prepare_virtual_weights(self, interested_features: list[Feature] | BatchedFeatures) -> None:
        if not interested_features:
            raise ValueError("At least one feature is required")
        if isinstance(interested_features, BatchedFeatures):
            interested_features = [
                (f"blocks.{layer}.hook_{'attn' if is_lorsa else 'mlp'}_out", index)
                for layer, index, is_lorsa in zip(
                    interested_features.layer.tolist(),
                    interested_features.index.tolist(),
                    interested_features.is_lorsa.tolist(),
                )
            ]
        self.interested_features = list(dict.fromkeys(interested_features))
        self.global_weights = None
        self.inhibitory_weights = None

    def record_activation_matrix(
        self, inputs: Iterable[str | torch.Tensor], *, mode: str = "global"
    ) -> GlobalWeights | InhibitoryWeights:
        if not self.interested_features:
            raise ValueError("Call prepare_virtual_weights before scanning")
        if mode == "global":
            self.global_weights = compute_global_weights(
                self.model, self.dictionaries, inputs, self.interested_features
            )
            return self.global_weights
        if mode == "inhibitory":
            self.inhibitory_weights = compute_inhibitory_global_weights(
                self.model, self.dictionaries, inputs, self.interested_features
            )
            return self.inhibitory_weights
        raise ValueError(f"Unknown global-weight mode: {mode}")

    def parse_global_weight_results(self, top_k: int = 10) -> list[ConnectedFeatures]:
        if self.global_weights is None:
            raise ValueError("Run a global scan before parsing results")
        return [
            _connected_features(
                self.global_weights.topk(feature, top_k),
                self.global_weights.downstream(feature, top_k),
                torch.device("cpu"),
            )
            for feature in self.interested_features
        ]

    def parse_inhibitory_results(self, top_k: int = 10) -> list[list[InhibitoryConnection]]:
        if self.inhibitory_weights is None:
            raise ValueError("Run an inhibitory scan before parsing results")
        return [self.inhibitory_weights.topk(feature, top_k) for feature in self.interested_features]
