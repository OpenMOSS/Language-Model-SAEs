from dataclasses import dataclass
from typing import Literal

import torch

FeatureType = Literal["transcoder", "lorsa"]


@dataclass
class FeatureSpecParams:
    """Feature specification parameters."""
    feature_type: FeatureType
    layer: int
    feature_id: int


class FeatureSpec:
    def __init__(self, params: FeatureSpecParams):
        self.feature_type = params.feature_type
        self.layer = params.layer
        self.feature_id = params.feature_id

def get_feature_vector(
    lorsas: list,
    transcoders: dict[int, torch.nn.Module],
    feature_type: FeatureType,
    layer: int,
    feature_id: int,
) -> torch.Tensor:
    if feature_type == "transcoder":
        sae = transcoders[layer]
        return sae.W_D[feature_id]
    if feature_type == "lorsa":
        lorsa = lorsas[layer]
        return lorsa.W_O[feature_id]
    raise ValueError(f"Invalid feature type: {feature_type}")


def get_feature_encoder_vector(
    lorsas: list,
    transcoders: dict[int, torch.nn.Module],
    feature_type: FeatureType,
    layer: int,
    feature_id: int,
) -> torch.Tensor:
    if feature_type == "transcoder":
        sae = transcoders[layer]
        # W_E: [d_model, d_sae] -> W_E.T: [d_sae, d_model]
        return sae.W_E.T[feature_id]
    if feature_type == "lorsa":
        lorsa = lorsas[layer]
        return lorsa.W_V[feature_id]
    raise ValueError(f"Invalid feature type: {feature_type}")