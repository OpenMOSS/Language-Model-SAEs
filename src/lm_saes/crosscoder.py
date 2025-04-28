import math
from typing import Any, Literal, Union, cast, overload, override

import einops
import torch
import torch.nn as nn
from jaxtyping import Float
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.config import CrossCoderConfig
from lm_saes.utils.distributed import distribute_tensor_on_dim, placements_from_dim_map


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

    @torch.no_grad()
    def init_parameters(self, **kwargs) -> None:
        """Initialize the weights of the model."""
        super().init_parameters(**kwargs)
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
        self.W_E.copy_(W_E)
        self.W_D.copy_(W_D)
        self.b_E.copy_(torch.zeros(self.cfg.n_heads, self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype))
        self.b_D.copy_(torch.zeros(self.cfg.n_heads, self.cfg.d_model, device=self.cfg.device, dtype=self.cfg.dtype))

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
        if self.device_mesh is not None and not isinstance(x, DTensor):
            x = distribute_tensor_on_dim(x, self.device_mesh, {})

        # Apply encoding per head
        hidden_pre = (
            einops.einsum(x, self.W_E, "... n_heads d_model, n_heads d_model d_sae -> ... n_heads d_sae") + self.b_E
        )
        # Sum across heads and add bias
        accumulated_hidden_pre = einops.einsum(hidden_pre, "... n_heads d_sae -> ... d_sae")
        accumulated_hidden_pre = einops.repeat(
            accumulated_hidden_pre, "... d_sae -> ... n_heads d_sae", n_heads=self.cfg.n_heads
        )

        # Apply activation function
        feature_acts = accumulated_hidden_pre * self.activation_function(accumulated_hidden_pre * self.decoder_norm())

        if return_hidden_pre:
            return feature_acts, accumulated_hidden_pre
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

        reconstructed = (
            einops.einsum(feature_acts, self.W_D, "... n_heads d_sae, n_heads d_sae d_model -> ... n_heads d_model")
            + self.b_D
        )
        return reconstructed

    @override
    def decoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Calculate the norm of the decoder weights.

        Returns:
            Norm of decoder weights of shape (n_heads, d_sae).
        """
        if not isinstance(self.W_D, DTensor):
            return torch.norm(self.W_D, dim=-1, keepdim=keepdim)
        else:
            assert self.device_mesh is not None
            return DTensor.from_local(
                torch.norm(self.W_D.to_local(), dim=-1, keepdim=keepdim),
                device_mesh=self.device_mesh,
                placements=placements_from_dim_map({"head": 0, "model": 1}, self.device_mesh),
            )

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
    @torch.no_grad()
    def set_decoder_to_fixed_norm(self, value: float, force_exact: bool):
        if force_exact:
            self.W_D.mul_(value / self.decoder_norm(keepdim=True))
        else:
            self.W_D.mul_(value / torch.clamp(self.decoder_norm(keepdim=True), min=value))

    @override
    @torch.no_grad()
    def set_encoder_to_fixed_norm(self, value: float):
        self.W_E.mul_(value / self.encoder_norm(keepdim=True))

    @override
    @torch.no_grad()
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
        self.b_E.div_(norm_factors.view(self.cfg.n_heads, 1))
        self.b_D.div_(norm_factors.view(self.cfg.n_heads, 1))
        self.cfg.norm_activation = "inference"

    @override
    @torch.no_grad()
    def init_encoder_with_decoder_transpose(self, factor: float = 1.0):
        transposed_decoder = (
            einops.rearrange(self.W_D, "n_heads d_sae d_model -> n_heads d_model d_sae").clone().contiguous() * factor
        )
        self.W_E.copy_(transposed_decoder)

    @override
    def prepare_input(self, batch: dict[str, torch.Tensor], **kwargs) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.device_mesh is None or "head" not in cast(tuple[str, ...], self.device_mesh.mesh_dim_names):
            return torch.stack([batch[hook_point] for hook_point in self.cfg.hook_points], dim=-2), {}
        else:
            local_activations = batch[self.cfg.hook_points[self.device_mesh.get_local_rank("head")]]
            local_activations = einops.rearrange(local_activations, "... d_model -> ... 1 d_model")
            return DTensor.from_local(
                local_activations,
                device_mesh=self.device_mesh,
                placements=placements_from_dim_map({"head": -2}, self.device_mesh),
            ), {}

    @override
    def prepare_label(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        return self.prepare_input(batch)[0]

    @override
    @torch.no_grad()
    def transform_to_unit_decoder_norm(self):
        raise NotImplementedError("Transform to unit decoder norm is not supported for CrossCoder")

    @override
    def tensor_parallel(self, device_mesh: DeviceMesh):
        super().tensor_parallel(device_mesh)
        W_E = distribute_tensor_on_dim(self.W_E, device_mesh, {"head": 0, "model": 2})
        self.register_parameter("W_E", nn.Parameter(W_E))
        W_D = distribute_tensor_on_dim(self.W_D, device_mesh, {"head": 0, "model": 1})
        self.register_parameter("W_D", nn.Parameter(W_D))
        b_E = distribute_tensor_on_dim(self.b_E, device_mesh, {"head": 0, "model": 1})
        self.register_parameter("b_E", nn.Parameter(b_E))
        b_D = distribute_tensor_on_dim(self.b_D, device_mesh, {"head": 0})
        self.register_parameter("b_D", nn.Parameter(b_D))
