import math
from typing import Any, Literal, Union, overload, override

import einops
import torch
import torch.nn as nn
from jaxtyping import Float

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.config import CrossCoderConfig


class CrossCoder(AbstractSparseAutoEncoder):
    """Sparse AutoEncoder model.

    An autoencoder model that learns to compress the input activation tensor into a high-dimensional but sparse feature activation tensor.

    Can also act as a transcoder model, which learns to compress the input activation tensor into a feature activation tensor, and then reconstruct a label activation tensor from the feature activation tensor.
    """

    def __init__(self, cfg: CrossCoderConfig):
        super(CrossCoder, self).__init__(cfg)
        self.cfg = cfg

        # Assertions
        assert cfg.sparsity_include_decoder_norm, "Sparsity should include decoder norm in CrossCoder"
        assert (
            not cfg.apply_decoder_bias_to_pre_encoder
        ), "Decoder bias should not be applied to pre-encoder in CrossCoder"
        assert cfg.use_decoder_bias, "Decoder bias should be used in CrossCoder"
        assert not cfg.use_triton_kernel, "Triton kernel is not supported in CrossCoder"

        # Initialize weights and biases
        self.W_E = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_sae, device=cfg.device, dtype=cfg.dtype))
        self.b_E = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_sae, device=cfg.device, dtype=cfg.dtype))
        self.W_D = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_sae, cfg.d_model, device=cfg.device, dtype=cfg.dtype))
        self.b_D = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, device=cfg.device, dtype=cfg.dtype))

    def init_parameters(self, **kwargs) -> None:
        """Initialize the weights of the model."""
        # Initialize a single head's weights
        W_E_per_head = torch.empty(
            self.cfg.d_model, self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype
        ).uniform_(-kwargs["encoder_uniform_bound"], kwargs["encoder_uniform_bound"])
        W_D_per_head = torch.empty(
            self.cfg.d_sae, self.cfg.d_model, device=self.cfg.device, dtype=self.cfg.dtype
        ).uniform_(-kwargs["decoder_uniform_bound"], kwargs["decoder_uniform_bound"])

        # Repeat for all heads
        W_E = einops.repeat(W_E_per_head, "d_model d_sae -> n_heads d_model d_sae", n_heads=self.cfg.n_heads)
        W_D = einops.repeat(W_D_per_head, "d_sae d_model -> n_heads d_sae d_model", n_heads=self.cfg.n_heads)

        # Assign to parameters
        self.W_E.data.copy_(W_E)
        self.W_D.data.copy_(W_D)
        self.b_E.data.copy_(torch.zeros(self.cfg.n_heads, self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype))
        self.b_D.data.copy_(
            torch.zeros(self.cfg.n_heads, self.cfg.d_model, device=self.cfg.device, dtype=self.cfg.dtype)
        )

    @overload
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        return_hidden_pre: Literal[False] = False,
        **kwargs,
    ) -> Union[Float[torch.Tensor, "batch d_sae"], Float[torch.Tensor, "batch seq_len d_sae"]]: ...

    @overload
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch n_heads d_model"],
            Float[torch.Tensor, "batch seq_len n_heads d_model"],
        ],
        return_hidden_pre: Literal[True],
        **kwargs,
    ) -> tuple[
        Union[
            Float[torch.Tensor, "batch n_heads d_sae"],
            Float[torch.Tensor, "batch seq_len n_heads d_sae"],
        ],
        Union[
            Float[torch.Tensor, "batch n_heads d_sae"],
            Float[torch.Tensor, "batch seq_len n_heads d_sae"],
        ],
    ]: ...

    @override
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch n_heads d_model"],
            Float[torch.Tensor, "batch seq_len n_heads d_model"],
        ],
        return_hidden_pre: bool = False,
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, "batch n_heads d_sae"],
        Float[torch.Tensor, "batch seq_len n_heads d_sae"],
        tuple[
            Union[
                Float[torch.Tensor, "batch n_heads d_sae"],
                Float[torch.Tensor, "batch seq_len n_heads d_sae"],
            ],
            Union[
                Float[torch.Tensor, "batch n_heads d_sae"],
                Float[torch.Tensor, "batch seq_len n_heads d_sae"],
            ],
        ],
    ]:
        """Encode the input tensor.

        Args:
            x: Input tensor of shape (n_heads, d_model).

        Returns:
            Encoded tensor of shape (n_heads, d_sae).
        """
        # Apply encoding per head
        hidden_pre = (
            einops.einsum(x, self.W_E, "... n_heads d_model, n_heads d_model d_sae -> ... n_heads d_sae") + self.b_E
        )

        # Sum across heads and add bias
        accumulated_hidden_pre = einops.einsum(hidden_pre, "n_heads d_sae -> d_sae")

        # Apply activation function
        feature_acts = accumulated_hidden_pre * self.activation_function(accumulated_hidden_pre * self.decoder_norm())

        return feature_acts

    @override
    def decode(
        self,
        feature_acts: Union[
            Float[torch.Tensor, "batch d_sae"],
            Float[torch.Tensor, "batch seq_len d_sae"],
        ],
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, "batch d_model"],
        Float[torch.Tensor, "batch seq_len d_model"],
    ]:  # may be overridden by subclasses
        """Decode the encoded tensor.

        Args:
            x: Encoded tensor of shape (n_heads, d_sae).

        Returns:
            Decoded tensor of shape (n_heads, d_model).
        """
        return (
            einops.einsum(feature_acts, self.W_D, "... d_sae, n_heads d_sae d_model -> ... n_heads d_model") + self.b_D
        )

    @override
    def decoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Calculate the norm of the decoder weights.

        Returns:
            Norm of decoder weights of shape (n_heads, d_sae).
        """
        return torch.norm(self.W_D, dim=-1, keepdim=keepdim)

    @override
    def encoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Calculate the norm of the encoder weights.

        Returns:
            Norm of encoder weights of shape (n_heads, d_sae).
        """
        return torch.norm(self.W_E, dim=-2, keepdim=keepdim)

    @override
    def decoder_bias_norm(self) -> torch.Tensor:
        return torch.norm(self.b_D, dim=-1, keepdim=True)

    @override
    def set_decoder_to_fixed_norm(self, value: float, force_exact: bool):
        if force_exact:
            self.W_D.data *= value / self.decoder_norm(keepdim=True)
            self.b_D.data *= value / self.decoder_bias_norm()
        else:
            self.W_D.data *= value / torch.clamp(self.decoder_norm(keepdim=True), min=value)
            self.b_D.data *= value / torch.clamp(self.decoder_bias_norm(), min=value)

    @override
    def set_encoder_to_fixed_norm(self, value: float):
        self.W_E.data *= value / self.encoder_norm(keepdim=True)

    @override
    def standardize_parameters_of_dataset_norm(self, dataset_average_activation_norm: dict[str, float] | None):
        """
        Standardize the parameters of the model to account for dataset_norm during inference.
        """
        assert self.cfg.norm_activation == "dataset-wise"
        assert self.dataset_average_activation_norm is not None or dataset_average_activation_norm is not None
        if dataset_average_activation_norm is not None:
            self.set_dataset_average_activation_norm(dataset_average_activation_norm)
        assert self.dataset_average_activation_norm is not None
        norm_factors = torch.tensor(
            [
                math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[hook_point]
                for hook_point in self.cfg.hook_points
            ],
            dtype=self.cfg.dtype,
            device=self.cfg.device,
        )
        self.b_E.data = self.b_E.data / norm_factors.view(self.cfg.n_heads, 1)
        self.b_D.data = self.b_D.data / norm_factors.view(self.cfg.n_heads, 1)
        self.cfg.norm_activation = "inference"

    @override
    def init_encoder_with_decoder_transpose(self, factor: float = 1.0):
        self.W_E.data = (
            einops.rearrange(self.W_D.data, "n_heads d_sae d_model -> n_heads d_model d_sae").clone().contiguous()
            * factor
        )

    @override
    def prepare_input(self, batch: dict[str, torch.Tensor], **kwargs) -> tuple[torch.Tensor, dict[str, Any]]:
        return torch.stack([batch[hook_point] for hook_point in self.cfg.hook_points], dim=-2), {}

    @override
    def prepare_label(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        return self.prepare_input(batch)[0]
