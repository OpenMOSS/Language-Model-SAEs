import torch
import torch.nn as nn

from typing import Dict, Literal, Union, overload, List
from jaxtyping import Float

import math

from lm_saes.sae import SparseAutoEncoder
from lm_saes.config import SAEConfig, LanguageModelSAETrainingConfig
from lm_saes.activation.activation_store import ActivationStore
from lm_saes.utils.huggingface import parse_pretrained_name_or_path

class PartialEncoder():
    """Partial Encoder of Sparse AutoEncoder model.
    
    Compress the input activation tensor into a part of a high-dimensional but sparse feature activation tensor.

    """
    
    def __init__(self, sae: SparseAutoEncoder, cfg:SAEConfig, start_index, end_index):
        """Initialize the Partial Encoder

        Args:
            cfg (SAEConfig): The configuration of the model.
            start_index: the start index of the feature activation tensor.
            end_index: the end index of the feature activation tensor.
        """
        
        self.cfg = cfg
        self.start_index = start_index
        self.end_index = end_index
        
        n_features = end_index - start_index
        self.n_features = n_features
        
        self.encoder = torch.nn.Linear(
            cfg.d_model, n_features, bias=True, device=sae.encoder.weight.device, dtype=cfg.dtype
        )
        
        self.encoder.weight.data.copy_(sae.encoder.weight.data[start_index:end_index, :])
        self.encoder.bias.data.copy_(sae.encoder.bias.data[start_index:end_index])
        
        self.encoder = self.encoder.to(cfg.device)
        
        self.decoder_bias = sae.decoder.bias.data.clone()
        
        self.decoder_bias = self.decoder_bias.to(cfg.device)
        
        if cfg.use_glu_encoder:

            self.encoder_glu = torch.nn.Linear(
                cfg.d_model, n_features, bias=True, device=sae.encoder_glu.weight.device, dtype=cfg.dtype
            )
            self.encoder_glu.weight.data.copy_(sae.encoder_glu.weight.data[start_index:end_index, :])
            self.encoder_glu.bias.data.copy_(sae.encoder_glu.bias.data[start_index:end_index])
            self.encoder_glu.to(cfg.device)

        self.feature_act_mask = torch.nn.Parameter(
            torch.ones((n_features,), dtype=cfg.dtype, device=sae.feature_act_mask.device)
        )
        self.feature_act_scale = torch.nn.Parameter(
            torch.ones((n_features,), dtype=cfg.dtype, device=sae.feature_act_scale.device)
        )
        self.feature_act_mask.data.copy_(sae.feature_act_mask.data[start_index:end_index])
        self.feature_act_scale.data.copy_(sae.feature_act_scale.data[start_index:end_index])
        self.feature_act_mask = self.feature_act_mask.to(cfg.device)
        self.feature_act_scale = self.feature_act_scale.to(cfg.device)
        
    
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        label: (
            Union[
                Float[torch.Tensor, "batch d_model"],
                Float[torch.Tensor, "batch seq_len d_model"],
            ]
            | None
        ) = None,
        return_hidden_pre: bool = False,
    ) -> Union[
        Float[torch.Tensor, "batch d_sae"],
        Float[torch.Tensor, "batch seq_len d_sae"],
        tuple[
            Union[
                Float[torch.Tensor, "batch d_sae"],
                Float[torch.Tensor, "batch seq_len d_sae"],
            ],
            Union[
                Float[torch.Tensor, "batch d_sae"],
                Float[torch.Tensor, "batch seq_len d_sae"],
            ],
        ],
    ]:
        """Encode the model activation x into partial feature activations.

        Args:
            x (torch.Tensor): The input activation tensor.
            label (torch.Tensor, optional): The label activation tensor in transcoder training. Used for normalizing the feature activations. Defaults to None, which means using x as the label.
            return_hidden_pre (bool, optional): Whether to return the hidden pre-activation. Defaults to False.

        Returns:
            torch.Tensor: a part of the feature activations.

        """

        if label is None:
            label = x

        if self.cfg.use_decoder_bias and self.cfg.apply_decoder_bias_to_pre_encoder:
            x = x - self.decoder_bias # type: ignore

        x = x * self.compute_norm_factor(x, hook_point="in")

        hidden_pre = self.encoder(x)

        if self.cfg.use_glu_encoder:
            hidden_pre_glu = torch.sigmoid(self.encoder_glu(x))

            hidden_pre = hidden_pre * hidden_pre_glu

        hidden_pre = hidden_pre / self.compute_norm_factor(label, hook_point="in")

        feature_acts = (
            self.feature_act_mask
            * self.feature_act_scale
            * torch.clamp(hidden_pre, min=0.0)
        )

        if return_hidden_pre:
            return feature_acts, hidden_pre
        return feature_acts
     

    def compute_norm_factor(
        self, x: torch.Tensor, hook_point: str
    ) -> float | torch.Tensor:
        """Compute the normalization factor for the activation vectors."""

        # Normalize the activation vectors to have L2 norm equal to sqrt(d_model)
        if self.cfg.norm_activation == "token-wise":
            return math.sqrt(self.cfg.d_model) / torch.norm(x, 2, dim=-1, keepdim=True)
        elif self.cfg.norm_activation == "batch-wise":
            return math.sqrt(self.cfg.d_model) / torch.norm(
                x, 2, dim=-1, keepdim=True
            ).mean(dim=-2, keepdim=True)
        elif self.cfg.norm_activation == "dataset-wise":
            assert (
                self.cfg.dataset_average_activation_norm is not None
            ), "dataset_average_activation_norm must be provided for dataset-wise normalization"
            return (
                math.sqrt(self.cfg.d_model)
                / self.cfg.dataset_average_activation_norm[hook_point]
            )
        else:
            return torch.tensor(1.0, dtype=self.cfg.dtype, device=self.cfg.device)
    