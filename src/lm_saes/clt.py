"""Cross Layer Transcoder (CLT) implementation.

Based on the methodology described in "Circuit Tracing: Revealing Computational Graphs in Language Models"
from Anthropic (https://transformer-circuits.pub/2025/attribution-graphs/methods.html).

A CLT consists of L encoders and L(L+1)/2 decoders where each encoder at layer L
reads from the residual stream at that layer and can decode to layers L through L-1.
This enables linear attribution between features across layers.
"""

import math
from typing import Any, Literal, Optional, Union, overload

import einops
import torch
import torch.distributed.tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from typing_extensions import override
from pydantic import Field

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.config import BaseSAEConfig, CLTConfig
from lm_saes.utils.distributed import DimMap


def _upper_triangular_index(layer_from: int, layer_to: int, n_layers: int) -> int:
    """Convert (layer_from, layer_to) to index in upper triangular storage.

    We store decoders for (layer_from=i, layer_to=j) where i <= j.
    The mapping is: flatten the upper triangular matrix by rows.

    Mapping:
    (0, 0) -> 0
    (0, 1) -> 1
    (1, 1) -> 2
    (0, 2) -> 3
    (1, 2) -> 4
    (2, 2) -> 5

    Args:
        layer_from: Source layer (0 to n_layers-1)
        layer_to: Target layer (layer_from to n_layers-1)
        n_layers: Total number of layers

    Returns:
        Index in the flattened upper triangular storage
    """
    assert 0 <= layer_from < n_layers
    assert layer_from <= layer_to < n_layers

    # Number of elements in rows 0 to layer_from-1
    elements_before_row = layer_from * n_layers - (layer_from * (layer_from - 1)) // 2
    # Position within row layer_from
    pos_in_row = layer_to - layer_from

    return int(elements_before_row + pos_in_row)  # Ensure integer index


def _total_upper_triangular_elements(n_layers: int) -> int:
    """Calculate total number of elements in upper triangular matrix."""
    return (n_layers * (n_layers + 1)) // 2


class CrossLayerTranscoder(AbstractSparseAutoEncoder):
    """Cross Layer Transcoder (CLT) implementation.

    A CLT has L encoders (one per layer) and L(L+1)/2 decoders arranged in an upper
    triangular pattern. Each encoder at layer L reads from the residual stream at that
    layer, and features can decode to layers L through L-1.

    We store all parameters in the same object and shard
    them across GPUs for efficient distributed training.
    """

    def __init__(self, cfg: CLTConfig, device_mesh: Optional[DeviceMesh] = None):
        """Initialize the Cross Layer Transcoder.
        
        Args:
            cfg: Configuration for the CLT.
            device_mesh: Device mesh for distributed training.
        """
        super().__init__(cfg, device_mesh)
        self.cfg = cfg

        # CLT requires specific configuration settings
        assert cfg.sparsity_include_decoder_norm, "CLT requires sparsity_include_decoder_norm=True"
        assert cfg.use_decoder_bias, "CLT requires use_decoder_bias=True"
        assert not cfg.apply_decoder_bias_to_pre_encoder, "CLT requires apply_decoder_bias_to_pre_encoder=False"

        # Calculate storage dimensions
        self.n_decoder_matrices = _total_upper_triangular_elements(cfg.n_layers)

        # Initialize weights and biases for cross-layer architecture
        if device_mesh is None:
            # L encoders: one for each layer
            self.W_E = nn.Parameter(
                torch.empty(cfg.n_layers, cfg.d_model, cfg.d_sae, device=cfg.device, dtype=cfg.dtype)
            )
            self.b_E = nn.Parameter(torch.empty(cfg.n_layers, cfg.d_sae, device=cfg.device, dtype=cfg.dtype))

            # L(L+1)/2 decoders: upper triangular pattern stored efficiently
            self.W_D = nn.Parameter(
                torch.empty(self.n_decoder_matrices, cfg.d_sae, cfg.d_model, device=cfg.device, dtype=cfg.dtype)
            )
            self.b_D = nn.Parameter(torch.empty(self.n_decoder_matrices, cfg.d_model, device=cfg.device, dtype=cfg.dtype))
        else:
            # Distributed initialization - shard along feature dimension
            self.W_E = nn.Parameter(
                torch.distributed.tensor.empty(
                    cfg.n_layers,
                    cfg.d_model,
                    cfg.d_sae,
                    dtype=cfg.dtype,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["W_E"].placements(device_mesh),
                )  # shard along d_model
            )
            self.b_E = nn.Parameter(
                torch.distributed.tensor.empty(
                    cfg.n_layers,
                    cfg.d_sae,
                    dtype=cfg.dtype,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["b_E"].placements(device_mesh),
                )  # shard along d_sae
            )
            self.W_D = nn.Parameter(
                torch.distributed.tensor.empty(
                    self.n_decoder_matrices,
                    cfg.d_sae,
                    cfg.d_model,
                    dtype=cfg.dtype,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["W_D"].placements(device_mesh),
                )  # shard along d_sae
            )
            self.b_D = nn.Parameter(
                torch.distributed.tensor.empty(
                    self.n_decoder_matrices,
                    cfg.d_model,
                    dtype=cfg.dtype,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["b_D"].placements(device_mesh),
                )  # shard along d_model
            )

    def get_decoder_weights(self, layer_from: int, layer_to: int) -> torch.Tensor:
        """Get decoder weights for layer_from -> layer_to.

        Args:
            layer_from: Source layer (0 to n_layers-1)
            layer_to: Target layer (layer_from to n_layers-1)

        Returns:
            Decoder weights for the specified layer pair
        """
        idx = _upper_triangular_index(layer_from, layer_to, self.cfg.n_layers)
        return self.W_D[idx]

    def get_decoder_bias(self, layer_from: int, layer_to: int) -> torch.Tensor:
        """Get decoder bias for layer_from -> layer_to.

        Args:
            layer_from: Source layer (0 to n_layers-1)
            layer_to: Target layer (layer_from to n_layers-1)

        Returns:
            Decoder bias for the specified layer pair
        """
        idx = _upper_triangular_index(layer_from, layer_to, self.cfg.n_layers)
        return self.b_D[idx]

    @overload
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch n_layers d_model"],
            Float[torch.Tensor, "batch seq_len n_layers d_model"],
        ],
        return_hidden_pre: Literal[False] = False,
        **kwargs,
    ) -> Union[Float[torch.Tensor, "batch n_layers d_sae"], Float[torch.Tensor, "batch seq_len n_layers d_sae"]]: ...

    @overload
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch n_layers d_model"],
            Float[torch.Tensor, "batch seq_len n_layers d_model"],
        ],
        return_hidden_pre: Literal[True],
        **kwargs,
    ) -> tuple[
        Union[
            Float[torch.Tensor, "batch n_layers d_sae"],
            Float[torch.Tensor, "batch seq_len n_layers d_sae"],
        ],
        Union[
            Float[torch.Tensor, "batch n_layers d_sae"],
            Float[torch.Tensor, "batch seq_len n_layers d_sae"],
        ],
    ]: ...

    @override
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch n_layers d_model"],
            Float[torch.Tensor, "batch seq_len n_layers d_model"],
        ],
        return_hidden_pre: bool = False,
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, "batch n_layers d_sae"],
        Float[torch.Tensor, "batch seq_len n_layers d_sae"],
        tuple[
            Union[
                Float[torch.Tensor, "batch n_layers d_sae"],
                Float[torch.Tensor, "batch seq_len n_layers d_sae"],
            ],
            Union[
                Float[torch.Tensor, "batch n_layers d_sae"],
                Float[torch.Tensor, "batch seq_len n_layers d_sae"],
            ],
        ],
    ]:
        """Encode input activations to CLT features using L encoders.

        Args:
            x: Input activations from all layers (..., n_layers, d_model)
            return_hidden_pre: Whether to return pre-activation values

        Returns:
            Feature activations for all layers (..., n_layers, d_sae)
        """
        # Apply each encoder to its corresponding layer: x[..., layer, :] @ W_E[layer] + b_E[layer]
        hidden_pre = torch.einsum("...ld,lds->...ls", x, self.W_E) + self.b_E

        # Apply activation function (ReLU, TopK, etc.)
        activation_mask = self.activation_function(hidden_pre)
        feature_acts = hidden_pre * activation_mask

        if return_hidden_pre:
            return feature_acts, hidden_pre
        return feature_acts

    @override
    def decode(
        self,
        feature_acts: Union[
            Float[torch.Tensor, "batch n_layers d_sae"],
            Float[torch.Tensor, "batch seq_len n_layers d_sae"],
        ],
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, "batch n_layers d_model"],
        Float[torch.Tensor, "batch seq_len n_layers d_model"],
    ]:
        """Decode CLT features to output activations using the upper triangular pattern.

        The output at layer L is the sum of contributions from all layers 0 through L:
        y_L = Σ_{i=0}^{L} W_D[i→L] @ feature_acts[..., i, :] + b_D[L]

        Args:
            feature_acts: CLT feature activations (..., n_layers, d_sae)

        Returns:
            Reconstructed activations for all layers (..., n_layers, d_model)
        """
        batch_shape = feature_acts.shape[:-2]
        output_shape = batch_shape + (self.cfg.n_layers, self.cfg.d_model)

        # Initialize output tensor
        if isinstance(feature_acts, DTensor):
            assert self.device_mesh is not None
            reconstructed = torch.distributed.tensor.zeros(
                output_shape,
                dtype=feature_acts.dtype,
                device_mesh=self.device_mesh,
                placements=[torch.distributed.tensor.Replicate()],
            )
        else:
            reconstructed = torch.zeros(output_shape, device=feature_acts.device, dtype=feature_acts.dtype)

        # For each output layer L
        for layer_to in range(self.cfg.n_layers):
            # Sum contributions from all layers 0 through L
            for layer_from in range(layer_to + 1):
                decoder_weights = self.get_decoder_weights(layer_from, layer_to)
                decoder_bias = self.get_decoder_bias(layer_from, layer_to)
                # Add contribution: feature_acts[..., layer_from, :] @ decoder_weights
                contribution = torch.einsum("...s,sd->...d", feature_acts[..., layer_from, :], decoder_weights)
                # Add bias to contribution before accumulating
                contribution = contribution + decoder_bias
                reconstructed[..., layer_to, :] += contribution
                print(contribution)

        return reconstructed

    @override
    def decoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Compute the effective norm of decoder weights for each feature."""

        if not isinstance(self.W_D, DTensor):
            return torch.norm(self.W_D, p=2, dim=-1, keepdim=keepdim).mean(dim=-1, keepdim=keepdim)
        else:
            assert self.device_mesh is not None
            decoder_norm = torch.norm(self.W_D.to_local(), p=2, dim=-1, keepdim=keepdim).mean(dim=-1, keepdim=keepdim)
            decoder_norm = DTensor.from_local(
                decoder_norm,
                device_mesh=self.device_mesh,
                placements=self.dim_maps()["W_D"].placements(self.device_mesh)[1:],  # Skip decoder matrix dimension
            )
            return decoder_norm.redistribute(placements=[torch.distributed.tensor.Replicate()], async_op=True).to_local()

    @override
    def encoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Compute the norm of encoder weights averaged across layers."""
        if not isinstance(self.W_E, DTensor):
            # W_E shape: (n_layers, d_model, d_sae)
            # Compute norm along d_model dimension (dim=-2), then average across layers (dim=0)
            return torch.norm(self.W_E, p=2, dim=-2, keepdim=keepdim).mean(dim=-1, keepdim=keepdim)
        else:
            assert self.device_mesh is not None
            encoder_norm = torch.norm(self.W_E.to_local(), p=2, dim=-2, keepdim=keepdim).mean(dim=-1, keepdim=keepdim)
            encoder_norm = DTensor.from_local(
                encoder_norm,
                device_mesh=self.device_mesh,
                placements=self.dim_maps()["W_E"].placements(self.device_mesh)[2:],  # Skip layer and model dimensions
            )
            return encoder_norm.redistribute(
                placements=[torch.distributed.tensor.Replicate()], async_op=True
            ).to_local()

    @override
    def decoder_bias_norm(self) -> torch.Tensor:
        """Compute the norm of decoder bias averaged across all decoders."""
        if not isinstance(self.b_D, DTensor):
            # b_D shape: (n_decoder_matrices, d_model)
            # Compute norm along d_model dimension (dim=-1), then average across decoders (dim=0)
            return torch.norm(self.b_D, p=2, dim=-1, keepdim=False)
        else:
            assert self.device_mesh is not None
            bias_norm = torch.norm(self.b_D.to_local(), p=2, dim=-1)
            bias_norm = DTensor.from_local(
                bias_norm, device_mesh=self.device_mesh, placements=[torch.distributed.tensor.Replicate()]
            )
            return bias_norm.to_local()

    @override
    def set_decoder_to_fixed_norm(self, value: float, force_exact: bool):
        raise NotImplementedError("set_decoder_to_fixed_norm does not make sense for CLT")

    @override
    @torch.no_grad()
    def set_encoder_to_fixed_norm(self, value: float):
        """Set encoder weights to fixed norm."""
        encoder_norm = self.encoder_norm(keepdim=True)  # Shape: (1, d_sae)
        # W_E shape: (n_layers, d_model, d_sae)
        # encoder_norm shape: (1, d_sae) -> need to broadcast to (1, 1, d_sae)
        self.W_E.data *= value / encoder_norm.unsqueeze(0)

    @override
    @torch.no_grad()
    def standardize_parameters_of_dataset_norm(self, dataset_average_activation_norm: "dict[str, float] | None"):
        """Standardize parameters for dataset-wise normalization during inference."""
        assert self.cfg.norm_activation == "dataset-wise"
        assert self.dataset_average_activation_norm is not None or dataset_average_activation_norm is not None

        if dataset_average_activation_norm is not None:
            self.set_dataset_average_activation_norm(dataset_average_activation_norm)
        assert self.dataset_average_activation_norm is not None

        # For CLT, we need to handle multiple input and output layers
        for layer_idx in range(self.cfg.n_layers):
            hook_point_in = self.cfg.hook_points_in[layer_idx]
            hook_point_out = self.cfg.hook_points_out[layer_idx]

            # Input normalization factor for this layer (from hook_points_in)
            input_norm_factor: float = (
                math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[hook_point_in]
            )

            # Output normalization factor for this layer (from hook_points_out)
            output_norm_factor: float = (
                math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[hook_point_out]
            )

            # Adjust encoder bias for this layer
            self.b_E.data[layer_idx] = self.b_E.data[layer_idx] / input_norm_factor

            # Adjust decoder weights and biases for decoders writing to this layer
            for layer_from in range(layer_idx + 1):
                decoder_idx = _upper_triangular_index(layer_from, layer_idx, self.cfg.n_layers)
                self.W_D.data[decoder_idx] = self.W_D.data[decoder_idx] * input_norm_factor / output_norm_factor
                # Adjust decoder bias for this specific decoder
                self.b_D.data[decoder_idx] = self.b_D.data[decoder_idx] / output_norm_factor

        self.cfg.norm_activation = "inference"

    @override
    @torch.no_grad()
    def init_encoder_with_decoder_transpose(self, factor: float = 1.0):
        raise NotImplementedError("init_encoder_with_decoder_transpose does not make sense for CLT")

    @override
    def prepare_input(self, batch: "dict[str, torch.Tensor]", **kwargs) -> "tuple[torch.Tensor, dict[str, Any]]":
        """Prepare input tensor from batch by stacking all layer activations from hook_points_in."""
        x_layers = []
        for hook_point in self.cfg.hook_points_in:
            if hook_point not in batch:
                raise ValueError(f"Missing hook point {hook_point} in batch")
            x_layers.append(batch[hook_point])
        x = torch.stack(x_layers, dim=-2)  # (..., n_layers, d_model)
        return x, {}

    @override
    def prepare_label(self, batch: "dict[str, torch.Tensor]", **kwargs) -> torch.Tensor:
        """Prepare label tensor from batch using hook_points_out."""
        x_layers = []
        for hook_point in self.cfg.hook_points_out:
            if hook_point not in batch:
                raise ValueError(f"Missing hook point {hook_point} in batch")
            x_layers.append(batch[hook_point])
        labels = torch.stack(x_layers, dim=-2)  # (..., n_layers, d_model)
        return labels

    @override
    @torch.no_grad()
    def transform_to_unit_decoder_norm(self):
        raise NotImplementedError("transform_to_unit_decoder_norm does not make sense for CLT")

    def tensor_parallel(self, device_mesh: DeviceMesh):
        """Distribute the parameters across multiple devices."""
        self.device_mesh = device_mesh

        # Distribute all parameters
        self.W_E = nn.Parameter(self.dim_maps()["W_E"].distribute(self.W_E, device_mesh))
        self.b_E = nn.Parameter(self.dim_maps()["b_E"].distribute(self.b_E, device_mesh))
        self.W_D = nn.Parameter(self.dim_maps()["W_D"].distribute(self.W_D, device_mesh))
        self.b_D = nn.Parameter(self.dim_maps()["b_D"].distribute(self.b_D, device_mesh))

    def dim_maps(self) -> "dict[str, DimMap]":
        """Return dimension maps for distributed training along feature dimension."""
        base_maps = super().dim_maps()

        clt_maps = {
            "W_E": DimMap({"model": 2}),  # Shard along d_sae dimension
            "b_E": DimMap({"model": 1}),  # Shard along d_sae dimension
            "W_D": DimMap({"model": 1}),  # Shard along d_sae dimension
            "b_D": DimMap({}),  # Replicate decoder biases
        }

        return base_maps | clt_maps

    @override
    def load_distributed_state_dict(
        self, state_dict: "dict[str, torch.Tensor]", device_mesh: DeviceMesh, prefix: str = ""
    ) -> None:
        """Load distributed state dict."""
        super().load_distributed_state_dict(state_dict, device_mesh, prefix)
        self.device_mesh = device_mesh

        # Load all parameters
        for param_name in ["W_E", "b_E", "W_D", "b_D"]:
            self.register_parameter(
                param_name,
                nn.Parameter(state_dict[f"{prefix}{param_name}"].to(getattr(self, param_name).dtype)),
            )

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, strict_loading: bool = True, **kwargs):
        """Load a pretrained CLT model."""
        cfg = BaseSAEConfig.from_pretrained(pretrained_name_or_path, strict_loading=strict_loading, **kwargs)
        model = cls.from_config(cfg)
        return model

    def cross_layer_attribution(
        self,
        x: Union[
            Float[torch.Tensor, "batch n_layers d_model"],
            Float[torch.Tensor, "batch seq_len n_layers d_model"],
        ],
    ) -> dict[str, Any]:
        """Compute cross-layer attribution for input activations.

        This method demonstrates how CLT enables attribution across layers
        by showing how features from each input layer contribute to each output layer.

        Args:
            x: Input activations from all layers (..., n_layers, d_model)

        Returns:
            Dictionary containing attribution information for each layer
        """
        feature_acts = self.encode(x)
        layer_outputs = self.decode(feature_acts)

        attribution_data: dict[str, Any] = {
            "feature_activations": feature_acts,
            "layer_outputs": layer_outputs,
        }

        # Add detailed per-layer attributions showing the cross-layer structure
        for layer_to in range(self.cfg.n_layers):
            layer_data = {}
            for layer_from in range(layer_to + 1):
                # Get contribution from layer_from to layer_to
                decoder_weights = self.get_decoder_weights(layer_from, layer_to)
                contribution = torch.einsum("...s,sd->...d", feature_acts[..., layer_from, :], decoder_weights)
                layer_data[f"from_layer_{layer_from}"] = contribution

            attribution_data[f"contributions_to_layer_{layer_to}"] = layer_data

        return attribution_data
