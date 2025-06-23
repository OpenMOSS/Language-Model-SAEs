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
        assert not cfg.sparsity_include_decoder_norm, "CLT requires sparsity_include_decoder_norm=False"
        assert cfg.use_decoder_bias, "CLT requires use_decoder_bias=True"
        assert not cfg.apply_decoder_bias_to_pre_encoder, "CLT requires apply_decoder_bias_to_pre_encoder=False"

        # Initialize weights and biases for cross-layer architecture
        if device_mesh is None:
            # L encoders: one for each layer
            self.W_E = nn.Parameter(
                torch.empty(cfg.n_layers, cfg.d_model, cfg.d_sae, device=cfg.device, dtype=cfg.dtype)
            )
            self.b_E = nn.Parameter(torch.empty(cfg.n_layers, cfg.d_sae, device=cfg.device, dtype=cfg.dtype))

            # L decoder groups: W_D[i] contains decoders from layers 0..i to layer i
            self.W_D = nn.ParameterList([
                nn.Parameter(data=torch.empty(i + 1, cfg.d_sae, cfg.d_model, device=cfg.device, dtype=cfg.dtype))
                for i in range(cfg.n_layers)
            ])
            
            # L decoder biases: one bias per target layer
            self.b_D = nn.ParameterList([
                nn.Parameter(torch.empty(cfg.d_model, device=cfg.device, dtype=cfg.dtype))
                for _ in range(cfg.n_layers)
            ])
            if cfg.act_fn == "jumprelu":
                self.log_jump_relu_threshold = nn.Parameter(torch.tensor(cfg.jump_relu_threshold, device=cfg.device, dtype=cfg.dtype))
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
            
            # L decoder groups: W_D[i] contains decoders from layers 0..i to layer i
            self.W_D = nn.ParameterList([
                nn.Parameter(torch.distributed.tensor.empty(
                    i + 1,
                    cfg.d_sae,
                    cfg.d_model,
                    dtype=cfg.dtype,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["W_D"].placements(device_mesh),
                ))  # shard along d_sae
                for i in range(cfg.n_layers)
            ])
            
            self.b_D = nn.ParameterList([
                nn.Parameter(
                    torch.distributed.tensor.empty(
                        cfg.d_model,
                        dtype=cfg.dtype,
                        device_mesh=device_mesh,
                        placements=self.dim_maps()["b_D"].placements(device_mesh),
                    )
                )
                for _ in range(cfg.n_layers)
            ])

    @override
    @torch.no_grad()
    def init_parameters(self, **kwargs):
        """Initialize parameters.
        
        Encoders: uniformly initialized in range (-1/sqrt(d_sae), 1/sqrt(d_sae))
        Decoders at layer L: uniformly initialized in range (-1/sqrt(L*d_model), 1/sqrt(L*d_model))
        """
        super().init_parameters(**kwargs)
        
        # Initialize encoder weights and biases
        encoder_bound = 1.0 / math.sqrt(self.cfg.d_sae)
        
        if self.device_mesh is None:
            # Non-distributed initialization
            
            # Initialize encoder weights: (n_layers, d_model, d_sae)
            W_E = torch.empty(
                self.cfg.n_layers, self.cfg.d_model, self.cfg.d_sae, 
                device=self.cfg.device, dtype=self.cfg.dtype
            ).uniform_(-encoder_bound, encoder_bound)
            
            # Initialize encoder biases: (n_layers, d_sae) - set to zero
            nn.init.zeros_(self.b_E)
            
            # Initialize decoder weights
            W_D_initialized = []
            for layer_to in range(self.cfg.n_layers):
                # Initialize decoder weights for layer layer_to
                # W_D[layer_to] has shape (layer_to+1, d_sae, d_model)
                # Scale by 1/sqrt(L*d_model) where L is the number of contributing layers
                scale = 1.0 / math.sqrt((layer_to + 1) * self.cfg.d_model)
                W_D_layer = torch.empty(layer_to + 1, self.cfg.d_sae, self.cfg.d_model, device=self.cfg.device, dtype=self.cfg.dtype)
                nn.init.uniform_(W_D_layer, -scale, scale)
                W_D_initialized.append(W_D_layer)
            
            # Initialize decoder biases
            for layer_to in range(self.cfg.n_layers):
                # Initialize decoder bias for layer layer_to to zero
                nn.init.zeros_(self.b_D[layer_to])
            
        else:
            # Distributed initialization
            # Initialize encoder weights
            W_E_local = torch.empty(
                self.cfg.n_layers, self.cfg.d_model, self.cfg.d_sae,
                device=self.cfg.device, dtype=self.cfg.dtype
            ).uniform_(-encoder_bound, encoder_bound)
            W_E = self.dim_maps()["W_E"].distribute(W_E_local, self.device_mesh)
            
            # Initialize encoder biases
            nn.init.zeros_(self.b_E)
            
            # Initialize decoder weights for each layer
            W_D_initialized = []
            for layer_to in range(self.cfg.n_layers):
                decoder_bound = 1.0 / math.sqrt((layer_to + 1) * self.cfg.d_model)
                W_D_layer_local = torch.empty(
                    layer_to + 1, self.cfg.d_sae, self.cfg.d_model,
                    device=self.cfg.device, dtype=self.cfg.dtype
                ).uniform_(-decoder_bound, decoder_bound)
                W_D_layer = self.dim_maps()["W_D"].distribute(W_D_layer_local, self.device_mesh)
                W_D_initialized.append(W_D_layer)
            
            # Initialize decoder biases
            for layer_to in range(self.cfg.n_layers):
                # Initialize decoder bias for layer layer_to to zero
                nn.init.zeros_(self.b_D[layer_to])
        
        # Copy initialized values to parameters
        self.W_E.copy_(W_E)
        
        for layer_to, W_D_layer in enumerate(W_D_initialized):
            self.W_D[layer_to].copy_(W_D_layer)
        
        # Initialize jump ReLU threshold if using jump ReLU activation
        if self.cfg.act_fn == "jumprelu" and hasattr(self, 'log_jump_relu_threshold'):
            if kwargs.get("init_log_jumprelu_threshold_value") is not None:
                self.log_jump_relu_threshold.data.fill_(kwargs["init_log_jumprelu_threshold_value"])

    def get_decoder_weights(self, layer_to: int) -> torch.Tensor:
        """Get decoder weights for all layers from 0..layer_to to layer_to.

        Args:
            layer_to: Target layer (0 to n_layers-1)

        Returns:
            Decoder weights for all source layers to the specified target layer
        """
        return self.W_D[layer_to]

    def get_decoder_bias(self, layer_to: int) -> torch.Tensor:
        """Get decoder bias for target layer layer_to.
        
        Args:
            layer_to: Target layer index (0 to n_layers-1)
            
        Returns:
            Decoder bias tensor of shape (d_model,)
        """
        return self.b_D[layer_to]

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
        reconstructed = []
    
        # For each output layer L
        for layer_to in range(self.cfg.n_layers):
            decoder_weights = self.get_decoder_weights(layer_to)  #(layer_to+1, d_sae, d_model)
            decoder_bias = self.get_decoder_bias(layer_to)  #(d_model,)
            
            if self.device_mesh is not None:
                assert isinstance(feature_acts, DTensor)
                feature_acts_per_layer = DTensor.from_local(feature_acts.to_local()[..., :layer_to + 1, :], device_mesh=self.device_mesh, placements=DimMap({"model": 2}).placements(self.device_mesh))
            else:
                feature_acts_per_layer = feature_acts[..., :layer_to + 1, :]
            
            print("Feature acts per layer", feature_acts_per_layer.shape, feature_acts_per_layer.placements)
            print("Decoder weights", decoder_weights.shape, decoder_weights.placements)

            # Compute weighted sum of features from layers 0 to layer_to
            contribution = torch.einsum("...ls,lsd->...d", feature_acts_per_layer, decoder_weights)
            # Add bias contribution (single bias vector for this target layer)
            if isinstance(decoder_bias, DTensor):
                reconstructed.append((contribution + decoder_bias).full_tensor())  # pyright: ignore
            else:
                reconstructed.append(contribution + decoder_bias)
        
        reconstructed = torch.stack(reconstructed, dim=-2)

        return reconstructed

    @override 
    def decoder_norm(self, keepdim: bool = False) -> Float[torch.Tensor, "n_decoder_matrices"]:
        """Compute the effective norm of decoder weights for each feature."""
        # Collect norms from all decoder groups
        all_norms = []
        for layer_to in range(self.cfg.n_layers):
            decoder_weights = self.W_D[layer_to]  # Shape: (layer_to+1, d_sae, d_model)
            if not isinstance(decoder_weights, DTensor):
                norms = torch.norm(decoder_weights, p=2, dim=-1, keepdim=keepdim).mean(dim=-1, keepdim=keepdim)
                all_norms.append(norms)
            else: 
                assert self.device_mesh is not None
                norms = torch.norm(decoder_weights.to_local(), p=2, dim=-1, keepdim=keepdim).mean(dim=-1, keepdim=keepdim)
                norms = DTensor.from_local(
                    norms,
                    device_mesh=self.device_mesh,
                    placements=self.dim_maps()["W_D"].placements(self.device_mesh)[1:],  # Skip first dimension
                )
                all_norms.append(norms.redistribute(placements=[torch.distributed.tensor.Replicate()], async_op=True).to_local())
        
        # Average across all decoders
        all_norms_tensor = torch.cat(all_norms, dim=0)  # Concatenate along decoder dimension
        return all_norms_tensor

    @override
    def encoder_norm(self, keepdim: bool = False) -> Float[torch.Tensor, "n_layers"]:
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
    def decoder_bias_norm(self) -> Float[torch.Tensor, "n_layers"]:
        """Compute the norm of decoder bias for each target layer."""
        bias_norms = []
        for layer_to in range(self.cfg.n_layers):
            bias = self.b_D[layer_to]  # Shape: (d_model,)
            if not isinstance(bias, DTensor):
                norm = torch.norm(bias, p=2, dim=-1, keepdim=False)
                bias_norms.append(norm)
            else:
                assert self.device_mesh is not None
                norm = torch.norm(bias.to_local(), p=2, dim=-1, keepdim=False)
                norm = DTensor.from_local(
                    norm, device_mesh=self.device_mesh, placements=[torch.distributed.tensor.Replicate()]
                )
                bias_norms.append(norm.to_local())
        
        return torch.stack(bias_norms, dim=0)  # Shape: (n_layers,)

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

            # Adjust decoder weights for decoders writing to this layer
            # W_D[layer_idx] contains decoders from layers 0..layer_idx to layer_idx
            self.W_D[layer_idx].data = self.W_D[layer_idx].data * input_norm_factor / output_norm_factor
            
            # Adjust decoder bias for this specific decoder
            self.b_D[layer_idx].data = self.b_D[layer_idx].data / output_norm_factor

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
        
        if isinstance(self.W_E, DTensor) and not isinstance(x, DTensor):
            assert self.device_mesh is not None
            x = DTensor.from_local(
                x, device_mesh=self.device_mesh, placements=[torch.distributed.tensor.Replicate()]
            )
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

        # Load encoder parameters
        for param_name in ["W_E", "b_E"]:
            self.register_parameter(
                param_name,
                nn.Parameter(state_dict[f"{prefix}{param_name}"].to(getattr(self, param_name).dtype)),
            )
        
        # Load W_D ModuleList parameters
        for layer_to in range(self.cfg.n_layers):
            param_name = f"W_D.{layer_to}"
            self.W_D[layer_to] = nn.Parameter(
                state_dict[f"{prefix}{param_name}"].to(self.W_D[layer_to].dtype)
            )

        # Load b_D parameters
        for layer_to in range(self.cfg.n_layers):
            param_name = f"b_D.{layer_to}"
            self.b_D[layer_to] = nn.Parameter(
                state_dict[f"{prefix}{param_name}"].to(self.b_D[layer_to].dtype)
            )

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, strict_loading: bool = True, **kwargs):
        """Load a pretrained CLT model."""
        cfg = BaseSAEConfig.from_pretrained(pretrained_name_or_path, strict_loading=strict_loading, **kwargs)
        model = cls.from_config(cfg)
        return model
