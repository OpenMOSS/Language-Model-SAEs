import math
from typing import Any, Literal, Optional, Union, cast, overload

import einops
import torch
import torch.distributed.tensor
import torch.nn as nn
from jaxtyping import Float
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Shard
from torch.distributed.tensor.experimental import local_map
from typing_extensions import override

from lm_saes.abstract_sae import (
    AbstractSparseAutoEncoder,
    BaseSAEConfig,
    register_sae_config,
    register_sae_model,
)
from lm_saes.utils.distributed import DimMap
from lm_saes.utils.distributed.ops import full_tensor, item
from lm_saes.utils.distributed.utils import replace_placements
from lm_saes.utils.misc import get_slice_length
from lm_saes.utils.tensor_specs import TensorSpecs
from lm_saes.utils.timer import timer


class CrossCoderSpecs(TensorSpecs):
    """Tensor specs for CrossCoder with n_heads dimension."""

    @staticmethod
    def feature_acts(tensor: torch.Tensor) -> tuple[str, ...]:
        if tensor.ndim == 3:
            return ("batch", "heads", "sae")
        elif tensor.ndim == 4:
            return ("batch", "context", "heads", "sae")
        else:
            raise ValueError(f"Cannot infer tensor specs for tensor with {tensor.ndim} dimensions.")

    @staticmethod
    def reconstructed(tensor: torch.Tensor) -> tuple[str, ...]:
        if tensor.ndim == 3:
            return ("batch", "heads", "model")
        elif tensor.ndim == 4:
            return ("batch", "context", "heads", "model")
        else:
            raise ValueError(f"Cannot infer tensor specs for tensor with {tensor.ndim} dimensions.")

    @staticmethod
    def label(tensor: torch.Tensor) -> tuple[str, ...]:
        return CrossCoderSpecs.reconstructed(tensor)


@register_sae_config("crosscoder")
class CrossCoderConfig(BaseSAEConfig):
    sae_type: str = "crosscoder"
    hook_points: list[str]

    @property
    def associated_hook_points(self) -> list[str]:
        return self.hook_points

    @property
    def n_heads(self) -> int:
        return len(self.hook_points)


@register_sae_model("crosscoder")
class CrossCoder(AbstractSparseAutoEncoder):
    """Sparse AutoEncoder model.

    An autoencoder model that learns to compress the input activation tensor into a high-dimensional but sparse feature activation tensor.

    Can also act as a transcoder model, which learns to compress the input activation tensor into a feature activation tensor, and then reconstruct a label activation tensor from the feature activation tensor.
    """

    specs: type[TensorSpecs] = CrossCoderSpecs
    """Tensor specs for CrossCoder with n_heads dimension."""

    def __init__(self, cfg: CrossCoderConfig, device_mesh: Optional[DeviceMesh] = None):
        super(CrossCoder, self).__init__(cfg, device_mesh)
        self.cfg = cfg

        # Assertions
        assert cfg.sparsity_include_decoder_norm, "Sparsity should include decoder norm in CrossCoder"
        assert cfg.use_decoder_bias, "Decoder bias should be used in CrossCoder"
        assert not cfg.use_triton_kernel, "Triton kernel is not supported in CrossCoder"

        # Initialize weights and biases
        if device_mesh is None:
            self.W_E = nn.Parameter(
                torch.empty(cfg.n_heads, cfg.d_model, cfg.d_sae, device=cfg.device, dtype=cfg.dtype)
            )
            self.b_E = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_sae, device=cfg.device, dtype=cfg.dtype))
            self.W_D = nn.Parameter(
                torch.empty(cfg.n_heads, cfg.d_sae, cfg.d_model, device=cfg.device, dtype=cfg.dtype)
            )
            self.b_D = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, device=cfg.device, dtype=cfg.dtype))
        else:
            self.W_E = nn.Parameter(
                torch.distributed.tensor.empty(
                    cfg.n_heads,
                    cfg.d_model,
                    cfg.d_sae,
                    dtype=cfg.dtype,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["W_E"].placements(device_mesh),
                )
            )
            self.b_E = nn.Parameter(
                torch.distributed.tensor.empty(
                    cfg.n_heads,
                    cfg.d_sae,
                    dtype=cfg.dtype,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["b_E"].placements(device_mesh),
                )
            )
            self.W_D = nn.Parameter(
                torch.distributed.tensor.empty(
                    cfg.n_heads,
                    cfg.d_sae,
                    cfg.d_model,
                    dtype=cfg.dtype,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["W_D"].placements(device_mesh),
                )
            )
            self.b_D = nn.Parameter(
                torch.distributed.tensor.empty(
                    cfg.n_heads,
                    cfg.d_model,
                    dtype=cfg.dtype,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["b_D"].placements(device_mesh),
                )
            )

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
        if self.device_mesh is None:
            W_E = einops.repeat(W_E_per_head, "d_model d_sae -> n_heads d_model d_sae", n_heads=self.cfg.n_heads)
            W_D = einops.repeat(W_D_per_head, "d_sae d_model -> n_heads d_sae d_model", n_heads=self.cfg.n_heads)
            b_E = torch.zeros(self.cfg.n_heads, self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype)
            b_D = torch.zeros(self.cfg.n_heads, self.cfg.d_model, device=self.cfg.device, dtype=self.cfg.dtype)
        else:
            with timer.time("init_parameters_distributed"):
                W_E_slices = self.dim_maps()["W_E"].local_slices(
                    (self.cfg.n_heads, self.cfg.d_model, self.cfg.d_sae), self.device_mesh
                )
                W_D_slices = self.dim_maps()["W_D"].local_slices(
                    (self.cfg.n_heads, self.cfg.d_sae, self.cfg.d_model), self.device_mesh
                )
                W_E_head_repeats = get_slice_length(W_E_slices[0], self.cfg.n_heads)
                W_D_head_repeats = get_slice_length(W_D_slices[0], self.cfg.n_heads)
                W_E_local = einops.repeat(
                    W_E_per_head[*W_E_slices[1:]], "d_model d_sae -> n_heads d_model d_sae", n_heads=W_E_head_repeats
                )
                W_D_local = einops.repeat(
                    W_D_per_head[*W_D_slices[1:]], "d_sae d_model -> n_heads d_sae d_model", n_heads=W_D_head_repeats
                )
                W_E = DTensor.from_local(
                    W_E_local, self.device_mesh, self.dim_maps()["W_E"].placements(self.device_mesh)
                )
                W_D = DTensor.from_local(
                    W_D_local, self.device_mesh, self.dim_maps()["W_D"].placements(self.device_mesh)
                )
                b_E = torch.distributed.tensor.zeros(
                    self.cfg.n_heads,
                    self.cfg.d_sae,
                    device_mesh=self.device_mesh,
                    placements=self.dim_maps()["b_E"].placements(self.device_mesh),
                    dtype=self.cfg.dtype,
                )
                b_D = torch.distributed.tensor.zeros(
                    self.cfg.n_heads,
                    self.cfg.d_model,
                    device_mesh=self.device_mesh,
                    placements=self.dim_maps()["b_D"].placements(self.device_mesh),
                    dtype=self.cfg.dtype,
                )

        # Assign to parameters
        self.W_E.copy_(W_E)
        self.W_D.copy_(W_D)
        self.b_E.copy_(b_E)
        self.b_D.copy_(b_D)

    @timer.time("encode_apply_encoding")
    def _apply_encoding(
        self,
        x: Union[
            Float[torch.Tensor, "batch n_heads d_model"],
            Float[torch.Tensor, "batch seq_len n_heads d_model"],
        ],
        *,
        no_einsum: bool = True,
    ) -> Union[
        Float[torch.Tensor, "batch n_heads d_sae"],
        Float[torch.Tensor, "batch seq_len n_heads d_sae"],
    ]:
        """Apply encoding transformation to input tensor.

        Args:
            x: Input tensor of shape (..., n_heads, d_model).

        Returns:
            Encoded tensor of shape (..., n_heads, d_sae).
        """
        if no_einsum:
            # TODO: Test consistency of this implementation
            def _apply_encoding_local_no_einsum(x: torch.Tensor, W_E: torch.Tensor, b_E: torch.Tensor) -> torch.Tensor:
                return torch.vmap(torch.matmul, in_dims=(-2, 0), out_dims=-2)(x, W_E) + b_E

            if self.device_mesh is not None:
                out_placements = DimMap({"data": 0, "head": -2, "model": -1}).placements(self.device_mesh)

                def _apply_encoding_no_einsum(x: torch.Tensor, W_E: torch.Tensor, b_E: torch.Tensor) -> torch.Tensor:
                    assert isinstance(x, DTensor) and isinstance(W_E, DTensor) and isinstance(b_E, DTensor)
                    return DTensor.from_local(
                        _apply_encoding_local_no_einsum(
                            x.to_local(),
                            W_E.to_local(
                                grad_placements=replace_placements(
                                    W_E.placements, W_E.device_mesh, "data", Partial("sum")
                                )
                            ),
                            b_E.to_local(
                                grad_placements=replace_placements(
                                    b_E.placements, b_E.device_mesh, "data", Partial("sum")
                                )
                            ),
                        ),
                        device_mesh=self.device_mesh,
                        placements=out_placements,
                    )
            else:
                _apply_encoding_no_einsum = _apply_encoding_local_no_einsum
            return _apply_encoding_no_einsum(x, self.W_E, self.b_E)
        else:
            return (
                einops.einsum(x, self.W_E, "... n_heads d_model, n_heads d_model d_sae -> ... n_heads d_sae") + self.b_E
            )

    @overload
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        return_hidden_pre: Literal[False] = False,
        *,
        no_einsum: bool = True,
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
        *,
        no_einsum: bool = True,
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
    @timer.time("encode")
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch n_heads d_model"],
            Float[torch.Tensor, "batch seq_len n_heads d_model"],
        ],
        return_hidden_pre: bool = False,
        *,
        no_einsum: bool = True,
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
            x: Input tensor of shape (..., n_heads, d_model).

        Returns:
            Encoded tensor of shape (..., n_heads, d_sae).
        """
        # Apply encoding per head
        hidden_pre = self._apply_encoding(x, no_einsum=no_einsum)

        # Sum across heads and add bias
        if not isinstance(hidden_pre, DTensor):
            accumulated_hidden_pre = torch.sum(hidden_pre, dim=-2)  # "... n_heads d_sae -> ... d_sae"
        else:
            accumulated_hidden_pre = cast(
                DTensor,
                cast(
                    DTensor,
                    local_map(
                        lambda x: torch.sum(x, dim=-2, keepdim=True),
                        list(hidden_pre.placements),
                    )(hidden_pre),
                ).sum(dim=-2),
            )  # "... n_heads d_sae -> ... d_sae"

            with timer.time("encode_redistribute_tensor_pre_repeat"):
                accumulated_hidden_pre = DimMap({"data": 0, "model": -1}).redistribute(accumulated_hidden_pre)

        accumulated_hidden_pre = einops.repeat(
            accumulated_hidden_pre, "... d_sae -> ... n_heads d_sae", n_heads=self.cfg.n_heads
        )

        with timer.time("encode_redistribute_tensor_post_repeat"):
            if isinstance(accumulated_hidden_pre, DTensor):
                accumulated_hidden_pre = DimMap({"data": 0, "head": -2, "model": -1}).redistribute(
                    accumulated_hidden_pre
                )

        # Apply activation function
        feature_acts = self.activation_function(accumulated_hidden_pre * self.decoder_norm()) / self.decoder_norm()

        if return_hidden_pre:
            return feature_acts, accumulated_hidden_pre
        return feature_acts

    def _apply_decoding(
        self,
        feature_acts: Union[
            Float[torch.Tensor, "batch n_heads d_sae"],
            Float[torch.Tensor, "batch seq_len n_heads d_sae"],
        ],
        *,
        no_einsum: bool = True,
    ) -> Union[
        Float[torch.Tensor, "batch n_heads d_model"],
        Float[torch.Tensor, "batch seq_len n_heads d_model"],
    ]:
        """Apply decoding transformation to feature activations.

        Args:
            feature_acts: Feature activations tensor of shape (..., n_heads, d_sae).

        Returns:
            Decoded tensor of shape (..., n_heads, d_model).
        """
        if no_einsum:

            def _apply_decoding_local_no_einsum(
                feature_acts: torch.Tensor, W_D: torch.Tensor, b_D: torch.Tensor
            ) -> torch.Tensor:
                return torch.vmap(torch.matmul, in_dims=(-2, 0), out_dims=-2)(feature_acts, W_D) + b_D

            if self.device_mesh is not None:
                out_placements = DimMap({"data": 0, "head": -2}).placements(self.device_mesh)

                def _apply_decoding_no_einsum(
                    feature_acts: torch.Tensor, W_D: torch.Tensor, b_D: torch.Tensor
                ) -> torch.Tensor:
                    assert isinstance(feature_acts, DTensor) and isinstance(W_D, DTensor) and isinstance(b_D, DTensor)
                    # feature_acts = DimMap({"head": -2, "model": -1}).redistribute(feature_acts)
                    return DTensor.from_local(
                        _apply_decoding_local_no_einsum(
                            feature_acts.to_local(),
                            W_D.to_local(
                                grad_placements=replace_placements(
                                    W_D.placements, W_D.device_mesh, "data", Partial("sum")
                                )
                            ),
                            b_D.to_local(
                                grad_placements=replace_placements(
                                    b_D.placements, b_D.device_mesh, "data", Partial("sum")
                                )
                            ),
                        ),
                        device_mesh=self.device_mesh,
                        placements=out_placements,
                    )
            else:
                _apply_decoding_no_einsum = _apply_decoding_local_no_einsum
            return _apply_decoding_no_einsum(feature_acts, self.W_D, self.b_D)

        else:
            return (
                einops.einsum(feature_acts, self.W_D, "... n_heads d_sae, n_heads d_sae d_model -> ... n_heads d_model")
                + self.b_D
            )

    @override
    @timer.time("decode")
    def decode(
        self,
        feature_acts: Union[
            Float[torch.Tensor, "batch d_sae"],
            Float[torch.Tensor, "batch seq_len d_sae"],
        ],
        *,
        no_einsum: bool = True,
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
        return self._apply_decoding(feature_acts, no_einsum=no_einsum)

    @override
    def decoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Calculate the norm of the decoder weights.

        Returns:
            Norm of decoder weights of shape (n_heads, d_sae).
        """
        with timer.time("decoder_norm_computation"):
            if not isinstance(self.W_D, DTensor):
                return torch.norm(self.W_D, dim=-1, keepdim=keepdim)
            else:
                assert self.device_mesh is not None
                return DTensor.from_local(
                    torch.norm(self.W_D.to_local(), dim=-1, keepdim=keepdim),
                    device_mesh=self.device_mesh,
                    placements=DimMap({"head": 0, "model": 1}).placements(self.device_mesh),
                )

    @override
    @timer.time("encoder_norm")
    def encoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Calculate the norm of the encoder weights.

        Returns:
            Norm of encoder weights of shape (n_heads, d_sae).
        """
        return torch.norm(self.W_E, dim=-2, keepdim=keepdim)

    @override
    @timer.time("decoder_bias_norm")
    def decoder_bias_norm(self) -> torch.Tensor:
        return torch.norm(self.b_D, dim=-1, keepdim=True)

    @override
    @timer.time("set_decoder_to_fixed_norm")
    @torch.no_grad()
    def set_decoder_to_fixed_norm(self, value: float, force_exact: bool):
        if force_exact:
            self.W_D.mul_(value / self.decoder_norm(keepdim=True))
        else:
            self.W_D.mul_(value / torch.clamp(self.decoder_norm(keepdim=True), min=value))

    @override
    @timer.time("set_encoder_to_fixed_norm")
    @torch.no_grad()
    def set_encoder_to_fixed_norm(self, value: float):
        self.W_E.mul_(value / self.encoder_norm(keepdim=True))

    @override
    @timer.time("standardize_parameters_of_dataset_norm")
    @torch.no_grad()
    def standardize_parameters_of_dataset_norm(self):
        """
        Standardize the parameters of the model to account for dataset_norm during inference.
        """
        assert self.cfg.norm_activation == "dataset-wise"
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
    @timer.time("init_encoder_with_decoder_transpose")
    @torch.no_grad()
    def init_encoder_with_decoder_transpose(self, factor: float = 1.0):
        transposed_decoder = (
            einops.rearrange(self.W_D, "n_heads d_sae d_model -> n_heads d_model d_sae").clone().contiguous() * factor
        )
        self.W_E.copy_(transposed_decoder)

    @override
    @timer.time("prepare_input")
    def prepare_input(
        self, batch: dict[str, torch.Tensor], **kwargs
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        def pad_to_d_model(x: torch.Tensor) -> torch.Tensor:
            # TODO: Support padding for distributed setting
            if x.shape[-1] > self.cfg.d_model:
                raise ValueError(f"Input tensor has {x.shape[-1]} dimensions, but expected {self.cfg.d_model}.")
            elif x.shape[-1] < self.cfg.d_model:
                assert not isinstance(x, DTensor), "Padding for DTensor is not supported"
                zero_padding = torch.zeros(
                    *x.shape[:-1], self.cfg.d_model - x.shape[-1], device=x.device, dtype=x.dtype
                )
                return torch.cat([x, zero_padding], dim=-1)
            else:
                return x

        # The following code is to stack the activations per head to (batch, ..., n_heads, d_model)
        if self.device_mesh is None or "head" not in cast(tuple[str, ...], self.device_mesh.mesh_dim_names):
            encoder_kwargs = {}
            decoder_kwargs = {}
            return (
                torch.stack([pad_to_d_model(batch[hook_point]) for hook_point in self.cfg.hook_points], dim=-2),
                encoder_kwargs,
                decoder_kwargs,
            )
        else:
            # The following code stacks the activations in a distributed setting. It's a bit complicated so I'll try to explain it in detail.

            # The input batch is not a dict of common DTensor-ed activations. It can be first sharded across the heads, which are keys of the dict. The values (DTensor activations) have a sub-mesh of the device mesh with the head dimension removed.

            # Each head-parallelized process only has a subset of the hook points. Retrieve the local hook points.
            # the `DimMap.local_slices` returns the range of indices of the local hook points in the global hook points list.
            local_hook_points = self.cfg.hook_points[
                DimMap({"head": 0}).local_slices((len(self.cfg.hook_points),), self.device_mesh)[0]
            ]

            # Stack the activations per head to (batch, ..., n_heads_per_process, d_model)
            if any(not isinstance(batch[hook_point], DTensor) for hook_point in local_hook_points):
                per_process_activations = torch.stack(
                    [pad_to_d_model(batch[hook_point]) for hook_point in local_hook_points], dim=-2
                )
                output_dim_map = DimMap({"head": -2})
            else:
                batch: dict[str, DTensor] = cast(dict[str, DTensor], batch)

                # We need to do some ugly local mapping and check since the `torch.stack` over DTensor results in buggy output placements.
                first_hook_point_activations = batch[local_hook_points[0]]
                assert all(
                    ((placement.dim + first_hook_point_activations.ndim) % first_hook_point_activations.ndim)
                    <= first_hook_point_activations.ndim - 2
                    for placement in first_hook_point_activations.placements
                    if isinstance(placement, Shard)
                )  # Check the input tensor is not sharded over the last dimension

                per_process_activations = torch.stack(
                    [pad_to_d_model(batch[hook_point].to_local()) for hook_point in local_hook_points],
                    dim=-2,
                )

                assert "head" not in cast(tuple[str, ...], first_hook_point_activations.device_mesh.mesh_dim_names), (
                    "Head dimension should not be in the device mesh of per head activations, as it should be added in the cross-head concatenation"
                )

                # Build the output dimension map by adding the head dimension.
                # This dimension map should work no matter whether the activations are data parallelized or not.
                output_dim_map = DimMap({"head": -2}) | DimMap.from_placements(
                    first_hook_point_activations.placements, first_hook_point_activations.device_mesh
                )

            encoder_kwargs = {}
            decoder_kwargs = {}
            return (
                DTensor.from_local(
                    per_process_activations,
                    device_mesh=self.device_mesh,
                    placements=output_dim_map.placements(self.device_mesh),
                ),
                encoder_kwargs,
                decoder_kwargs,
            )

    @override
    @timer.time("prepare_label")
    def prepare_label(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        return self.prepare_input(batch)[0]

    @override
    @torch.no_grad()
    def compute_training_metrics(
        self,
        *,
        feature_acts: torch.Tensor,
        l_rec: torch.Tensor,
        l0: torch.Tensor,
        explained_variance: torch.Tensor,
        **kwargs,
    ) -> dict[str, float]:
        """Compute per-head training metrics for CrossCoder."""
        assert explained_variance.ndim == 1 and len(explained_variance) == len(self.cfg.hook_points)
        feature_act_spec = self.specs.feature_acts(feature_acts)
        l0_spec = tuple(spec for spec in feature_act_spec if spec != "sae")
        l_rec_spec = tuple(
            spec for spec in feature_act_spec if spec != "model" and spec != "batch" and spec != "context"
        )
        metrics = {}
        for i, k in enumerate(self.cfg.hook_points):
            metrics.update(
                {
                    f"crosscoder_metrics/{k}/explained_variance": item(explained_variance[i].mean()),
                    f"crosscoder_metrics/{k}/l0": item(l0.select(l0_spec.index("heads"), i).mean()),
                    f"crosscoder_metrics/{k}/l_rec": item(l_rec.select(l_rec_spec.index("heads"), i).mean()),
                }
            )
        indices = feature_acts.amax(dim=1).nonzero(as_tuple=True)
        activated_feature_acts = feature_acts.permute(0, 2, 1)[indices].permute(1, 0)
        activated_decoder_norms = full_tensor(self.decoder_norm())[:, indices[1]]
        mean_decoder_norm_non_activated_in_activated = item(activated_decoder_norms[activated_feature_acts == 0].mean())
        mean_decoder_norm_activated_in_activated = item(activated_decoder_norms[activated_feature_acts != 0].mean())
        metrics.update(
            {
                "crosscoder_metrics/mean_decoder_norm_non_activated_in_activated": mean_decoder_norm_non_activated_in_activated,
                "crosscoder_metrics/mean_decoder_norm_activated_in_activated": mean_decoder_norm_activated_in_activated,
            }
        )
        return metrics

    @override
    @torch.no_grad()
    def transform_to_unit_decoder_norm(self):
        raise NotImplementedError("Transform to unit decoder norm is not supported for CrossCoder")

    def dim_maps(self) -> dict[str, DimMap]:
        """Return a dictionary mapping parameter names to dimension maps.

        Returns:
            A dictionary mapping parameter names to DimMap objects.
        """
        parent_maps = super().dim_maps()
        crosscoder_maps = {
            "W_E": DimMap({"head": 0, "model": 2}),
            "W_D": DimMap({"head": 0, "model": 1}),
            "b_E": DimMap({"head": 0, "model": 1}),
            "b_D": DimMap({"head": 0}),
        }
        return parent_maps | crosscoder_maps

    @torch.no_grad()
    @timer.time("decoder_inner_product_matrices")
    def decoder_inner_product_matrices(self) -> Float[torch.Tensor, "d_sae n_head n_head"]:
        inner_product_matrices = einops.einsum(self.W_D, self.W_D, "i d_sae d_model, j d_sae d_model -> d_sae i j")
        inner_product_matrices = einops.einsum(self.W_D, self.W_D, "i d_sae d_model, j d_sae d_model -> d_sae i j")
        return inner_product_matrices

    @torch.no_grad()
    @timer.time("decoder_similarity_matrices")
    def decoder_similarity_matrices(self) -> Float[torch.Tensor, "d_sae n_head n_head"]:
        inner_product_matrices = self.decoder_inner_product_matrices()
        decoder_norms = self.decoder_norm()
        decoder_norm_products = einops.einsum(decoder_norms, decoder_norms, "i d_sae, j d_sae -> d_sae i j")
        return inner_product_matrices / decoder_norm_products
