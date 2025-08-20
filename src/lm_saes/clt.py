"""Cross Layer Transcoder (CLT) implementation.

Based on the methodology described in "Circuit Tracing: Revealing Computational Graphs in Language Models"
from Anthropic (https://transformer-circuits.pub/2025/attribution-graphs/methods.html).

A CLT consists of L encoders and L(L+1)/2 decoders where each encoder at layer L
reads from the residual stream at that layer and can decode to layers L through L-1.
This enables linear attribution between features across layers.
"""

import math
from torch._tensor import Tensor
from typing import Any, Callable, Literal, Optional, Union, overload, List

import einops
import torch
import torch.distributed as dist
import torch.distributed.tensor
import torch.nn as nn
from jaxtyping import Float
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from typing_extensions import override

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.activation_functions import JumpReLU
from lm_saes.config import CLTConfig
from lm_saes.utils.distributed import DimMap
from lm_saes.utils.logging import get_distributed_logger
from lm_saes.utils.timer import timer
from lm_saes.kernels.kernels import TritonDecoderAutogradTopK
from lm_saes.kernels.kernels import get_sparse_representation

logger = get_distributed_logger("clt")

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
        # assert not cfg.sparsity_include_decoder_norm, "CLT requires sparsity_include_decoder_norm=False"
        # assert cfg.use_decoder_bias, "CLT requires use_decoder_bias=True"

        # Initialize weights and biases for cross-layer architecture
        if device_mesh is None:
            # L encoders: one for each layer
            self.W_E = nn.Parameter(
                torch.empty(cfg.n_layers, cfg.d_model, cfg.d_sae, device=cfg.device, dtype=cfg.dtype)
            )
            self.b_E = nn.Parameter(torch.empty(cfg.n_layers, cfg.d_sae, device=cfg.device, dtype=cfg.dtype))

            # L decoder groups: W_D[i] contains decoders from layers 0..i to layer i
            self.W_D = nn.ParameterList(
                [
                    nn.Parameter(data=torch.empty(i + 1, cfg.d_sae, cfg.d_model, device=cfg.device, dtype=cfg.dtype))
                    for i in range(cfg.n_layers)
                ]
            )

            # L decoder biases: one bias per target layer
            self.b_D = nn.ParameterList(
                [
                    nn.Parameter(torch.empty(cfg.d_model, device=cfg.device, dtype=cfg.dtype))
                    for _ in range(cfg.n_layers)
                ]
            )
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
                )  # shard along d_sae
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
            self.W_D = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.distributed.tensor.empty(
                            i + 1,
                            cfg.d_sae,
                            cfg.d_model,
                            dtype=cfg.dtype,
                            device_mesh=device_mesh,
                            placements=self.dim_maps()["W_D"].placements(device_mesh),
                        )
                    )  # shard along d_sae
                    for i in range(cfg.n_layers)
                ]
            )

            self.b_D = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.distributed.tensor.empty(
                            cfg.d_model,
                            dtype=cfg.dtype,
                            device_mesh=device_mesh,
                            placements=self.dim_maps()["b_D"].placements(device_mesh),
                        )
                    )
                    for _ in range(cfg.n_layers)
                ]
            )


    def activation_function_factory(
        self, device_mesh: DeviceMesh | None = None
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        assert self.cfg.act_fn.lower() in [
            "relu",
            "topk",
            "jumprelu",
            "batchtopk",
            "batchlayertopk",
        ], f"Not implemented activation function {self.cfg.act_fn}"
        if self.cfg.act_fn.lower() == "relu":
            return lambda x: x.gt(0).to(x.dtype)
        elif self.cfg.act_fn.lower() == "jumprelu":
            return JumpReLU(
                self.cfg.jumprelu_threshold_window,
                shape=(self.cfg.n_layers, self.cfg.d_sae,),
                dims_to_keep_in_bwd=(-2, -1),
                device=self.cfg.device,
                dtype=self.cfg.dtype if self.cfg.promote_act_fn_dtype is None else self.cfg.promote_act_fn_dtype,
                device_mesh=device_mesh,
            )

        elif self.cfg.act_fn.lower() == "topk":
            if device_mesh is not None:
                from lm_saes.utils.distributed import distributed_topk
                
                def topk_activation(x: Union[
                        Float[torch.Tensor, "batch n_layer d_sae"],
                        Float[torch.Tensor, "batch seq_len n_layer d_sae"],
                    ],
                ):
                    return distributed_topk(
                        x,
                        k=self.current_k,
                        device_mesh=device_mesh,
                        dim=-1,
                        mesh_dim_name="model",
                    )
            else:
                def topk_activation(
                    x: Union[
                        Float[torch.Tensor, "batch n_layer d_sae"],
                        Float[torch.Tensor, "batch seq_len n_layer d_sae"],
                    ],
                ):
                    from lm_saes.utils.math import topk
                    return topk(
                        x,
                        k=self.current_k,
                        dim=-1,
                    )
            
            return topk_activation

        elif self.cfg.act_fn.lower() == "batchtopk":
            if device_mesh is not None:
                from lm_saes.utils.distributed import distributed_topk
                
                def batch_topk(x: Union[
                        Float[torch.Tensor, "batch n_layer d_sae"],
                        Float[torch.Tensor, "batch seq_len n_layer d_sae"],
                    ],
                ):
                    x = x * x.gt(0).to(x.dtype)
                    return distributed_topk(
                        x,
                        k=self.current_k,
                        device_mesh=device_mesh,
                        dim=(-2, -1),
                        mesh_dim_name="model",
                    )
            else:
                # single-GPU batchtopk
                from lm_saes.utils.math import topk
                def batch_topk(x: Union[
                        Float[torch.Tensor, "batch n_layer d_sae"],
                        Float[torch.Tensor, "batch seq_len n_layer d_sae"],
                    ],
                ):
                    x = x * x.gt(0).to(x.dtype)
                    return topk(
                        x,
                        k=self.current_k,
                        dim=(-2, -1),
                    )
            
            return batch_topk
            
        elif self.cfg.act_fn.lower() == "batchlayertopk":
            if device_mesh is not None:
                from lm_saes.utils.distributed import distributed_topk
                
                def batch_layer_topk(x: Union[
                        Float[torch.Tensor, "batch n_layer d_sae"],
                        Float[torch.Tensor, "batch seq_len n_layer d_sae"],
                    ],
                ):
                    x = x * x.gt(0).to(x.dtype)
                    if x.ndim == 4:
                        x = x.flatten(end_dim=1)
                    result = distributed_topk(
                        x,
                        k=self.current_k * x.size(0),
                        device_mesh=device_mesh,
                        dim=(-3, -2, -1),
                        mesh_dim_name="model",
                    )
                    if x.ndim == 4:
                        result = result.unflatten(dim=0, sizes=(x.size(0), x.size(1)))
                    return result
            else:
                # single-GPU batchtopk
                from lm_saes.utils.math import topk
                def batch_layer_topk(x: Union[
                        Float[torch.Tensor, "batch n_layer d_sae"],
                        Float[torch.Tensor, "batch seq_len n_layer d_sae"],
                    ],
                ):
                    x = x * x.gt(0).to(x.dtype)
                    if x.ndim == 4:
                        x = x.flatten(end_dim=1)
                    result = topk(
                        x,
                        k=self.current_k * x.size(0),
                        dim=(-3, -2, -1),
                    )
                    if x.ndim == 4:
                        result = result.unflatten(dim=0, sizes=(x.size(0), x.size(1)))
                    return result

            return batch_layer_topk  # type: ignore

        raise ValueError(f"Not implemented activation function {self.cfg.act_fn}")
        
    @override
    @torch.no_grad()
    def init_parameters(self, **kwargs):
        """Initialize parameters.

        Encoders: uniformly initialized in range (-1/sqrt(d_sae), 1/sqrt(d_sae))
        Decoders at layer L: uniformly initialized in range (-1/sqrt(L*d_model), 1/sqrt(L*d_model))
        """
        super().init_parameters(**kwargs)  # jump ReLU threshold is initialized in super()

        # Initialize encoder weights and biases
        encoder_bound = 1.0 / math.sqrt(self.cfg.d_sae)

        if self.device_mesh is None:
            # Non-distributed initialization

            # Initialize encoder weights: (n_layers, d_model, d_sae)
            W_E = torch.empty(
                self.cfg.n_layers, self.cfg.d_model, self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype
            ).uniform_(-encoder_bound, encoder_bound)

            # Initialize encoder biases: (n_layers, d_sae) - set to zero
            nn.init.zeros_(self.b_E)

            # Initialize decoder weights
            W_D_initialized = []
            scale = 1.0 / math.sqrt(self.cfg.n_layers * self.cfg.d_model)
            for layer_to in range(self.cfg.n_layers):
                # Initialize decoder weights for layer layer_to
                # W_D[layer_to] has shape (layer_to+1, d_sae, d_model)
                # Scale by 1/sqrt(L*d_model) where L is the number of contributing layers
                
                W_D_layer = torch.empty(
                    layer_to + 1, self.cfg.d_sae, self.cfg.d_model, device=self.cfg.device, dtype=self.cfg.dtype
                )
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
                self.cfg.n_layers, self.cfg.d_model, self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype
            ).uniform_(-encoder_bound, encoder_bound)
            W_E = self.dim_maps()["W_E"].distribute(W_E_local, self.device_mesh)

            # Initialize encoder biases
            nn.init.zeros_(self.b_E)

            # Initialize decoder weights for each layer
            W_D_initialized = []
            for layer_to in range(self.cfg.n_layers):
                decoder_bound = 1.0 / math.sqrt(self.cfg.n_layers * self.cfg.d_model)
                W_D_layer_local = torch.empty(
                    layer_to + 1, self.cfg.d_sae, self.cfg.d_model, device=self.cfg.device, dtype=self.cfg.dtype
                ).uniform_(-decoder_bound, decoder_bound)
                W_D_layer = self.dim_maps()["W_D"].distribute(tensor=W_D_layer_local, device_mesh=self.device_mesh)
                W_D_initialized.append(W_D_layer)

            # Initialize decoder biases
            for layer_to in range(self.cfg.n_layers):
                # Initialize decoder bias for layer layer_to to zero
                nn.init.zeros_(self.b_D[layer_to])

        # Copy initialized values to parameters
        self.W_E.copy_(W_E)

        for layer_to, W_D_layer in enumerate(W_D_initialized):
            self.W_D[layer_to].copy_(W_D_layer)

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
        with timer.time("encoder_matmul"):
            hidden_pre = torch.einsum("...ld,lds->...ls", x, self.W_E) + self.b_E

        if self.cfg.sparsity_include_decoder_norm:
            decoder_norms = self.decoder_norm_per_feature()
            hidden_pre = hidden_pre * decoder_norms
        
        # Apply activation function (ReLU, TopK, etc.)
        with timer.time("activation_function"):
            feature_acts = self.activation_function(hidden_pre)
        
        if self.cfg.sparsity_include_decoder_norm:
            feature_acts = feature_acts / decoder_norms

        if return_hidden_pre:
            return feature_acts, hidden_pre
        return feature_acts

    def encode_single_layer(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        layer: int,
        return_hidden_pre: bool = False,
        **kwargs,
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
        """Encode input activations to CLT features using L encoders.

        Args:
            x: Input activations from a given layer (..., d_model)
            layer: The layer to encode
            return_hidden_pre: Whether to return pre-activation values

        Returns:
            Feature activations for the given layer (..., d_sae)
        """
        # Apply each encoder to its corresponding layer: x[..., layer, :] @ W_E[layer] + b_E[layer]
        hidden_pre = torch.einsum("...d,ds->...s", x, self.W_E[layer]) + self.b_E[layer]

        if self.cfg.sparsity_include_decoder_norm:
            hidden_pre = hidden_pre * self.decoder_norm_per_feature(layer)

        # Apply activation function (ReLU, TopK, etc.)
        if self.cfg.act_fn.lower() == "jumprelu":
            assert isinstance(self.activation_function, JumpReLU)
            jumprelu_threshold = self.activation_function.get_jumprelu_threshold()
            feature_acts = hidden_pre * hidden_pre.gt(jumprelu_threshold[layer])
        else:
            feature_acts = self.activation_function(hidden_pre)

        if return_hidden_pre:
            return feature_acts, hidden_pre
        return feature_acts

    @override
    def decode(
        self,
        feature_acts: Union[
            Float[torch.Tensor, "batch n_layers d_sae"],
            Float[torch.Tensor, "batch seq_len n_layers d_sae"],
            List[Float[torch.sparse.Tensor, "seq_len d_sae"]],
        ],
        batch_first: bool = False,
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, "n_layers batch d_model"],
        Float[torch.Tensor, "n_layers batch seq_len d_model"],
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
        batch_size = feature_acts.size(0)
        feature_acts = [
            fa.to_sparse_csr() for fa in feature_acts.to_local().permute(1, 0, 2)
        ]
        # For each output layer L
        for layer_to in range(self.cfg.n_layers):
            decoder_bias = self.get_decoder_bias(layer_to)  # (d_model,)

            # we only compute W_D @ feature_acts here, without b_D
            if isinstance(feature_acts, list) and isinstance(feature_acts[0], torch.Tensor) and feature_acts[0].layout == torch.sparse_coo:
                contribution = self._decode_single_output_layer_coo(feature_acts, layer_to)
            elif self.cfg.decode_with_csr:
                contribution = self._decode_single_output_layer_csr(feature_acts, layer_to, batch_size=batch_size)
            else:
                contribution = self._decode_single_output_layer_dense(feature_acts, layer_to)

            # Add bias contribution (single bias vector for this target layer)
            contribution = contribution + decoder_bias
            if isinstance(contribution, DTensor):
                contribution = contribution.full_tensor()
            reconstructed.append(contribution)

        return torch.stack(reconstructed, dim=1 if batch_first else 0)
    
    def _decode_single_output_layer_dense(self, feature_acts: Union[
        Float[torch.Tensor, "batch n_layers d_sae"],
        Float[torch.Tensor, "batch seq_len n_layers d_sae"],
    ], layer_to: int) -> Union[
        Float[torch.Tensor, "batch d_model"],
        Float[torch.Tensor, "batch seq_len d_model"],
    ]:
        """Decode features for a single output layer using upper triangular pattern."""
        decoder_weights = self.get_decoder_weights(layer_to)  # (layer_to+1, d_sae, d_model)
        feature_acts_per_layer = feature_acts[..., : layer_to + 1, :]
        contribution = feature_acts_per_layer.permute(1, 0, 2) @ decoder_weights
        return contribution.sum(0)
    
    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def _decode_single_output_layer_csr(self, feature_acts: Union[
        Float[torch.Tensor, "batch n_layers d_sae"],
        Float[torch.Tensor, "batch seq_len n_layers d_sae"],
    ], layer_to: int, batch_size: int) -> Union[
        Float[torch.Tensor, "batch d_model"],
        Float[torch.Tensor, "batch seq_len d_model"],
    ]:
        """Decode features for a single output layer using upper triangular pattern."""
        decoder_weights = self.get_decoder_weights(layer_to)  # (layer_to+1, d_sae, d_model)
        if self.device_mesh is not None:
            decoder_weights = decoder_weights.to_local()

        contribution = torch.zeros(
            batch_size,
            decoder_weights.size(-1),
            device=self.cfg.device,
            dtype=self.cfg.dtype,
        )
        
        for i in range(layer_to + 1):
            contribution = contribution + torch.sparse.mm(
                feature_acts[i],
                decoder_weights[i],
            )  # type: ignore

        if self.device_mesh is not None:
            contribution = DTensor.from_local(
                contribution.unsqueeze(1),
                device_mesh=self.device_mesh,
                placements=self.dim_maps()["W_D"].placements(self.device_mesh)
            )
            contribution = contribution.sum(dim=1)

        return contribution
    
    @torch.no_grad()
    def _decode_single_output_layer_coo(
        self,
        feature_acts: List[Float[torch.sparse.Tensor, "seq_len d_sae"]],
        layer_to: int,
    ) -> Union[
        Float[torch.Tensor, "batch d_model"],
        Float[torch.Tensor, "batch seq_len d_model"],
    ]:
        assert isinstance(feature_acts, list)
        for fa in feature_acts:
            assert fa.is_sparse
            assert fa.ndim == 2

        decoder_weights = self.get_decoder_weights(layer_to)  # (layer_to+1, d_sae, d_model)

        total_contribution = torch.zeros(
            *feature_acts[0].shape[:-1],
            decoder_weights.shape[-1],
            device=feature_acts[0].device,
            dtype=feature_acts[0].dtype
        )  # shape (seq_len, d_model)

        # each feature_acts[i] has indices: (2, sum of K over seq_len)
        # each feature_acts[i] has values: (sum of K over seq_len)

        for layer_from in range(layer_to + 1):
            sparse_tensor = feature_acts[layer_from]
            seq_indices = sparse_tensor.indices()[0]  # Sequence position indices
            feature_indices = sparse_tensor.indices()[1]  # Feature indices
            values = sparse_tensor.values()  # Active values
            
            # Get decoder weights for active features
            active_decoder_vecs = decoder_weights[layer_from][feature_indices]  # (K, d_model)
            
            # Compute contribution for each active feature: values * decoder_weights
            scaled_decoder_vecs = active_decoder_vecs * values.unsqueeze(-1)  # (K, d_model)
            
            # Accumulate contributions to the appropriate sequence positions
            total_contribution.index_add_(0, seq_indices, scaled_decoder_vecs)

        return total_contribution
    
    @override
    def decoder_norm(self, keepdim: bool = False):
        """Compute the effective norm of decoder weights for each feature."""
        # Collect norms from all decoder groups
        return torch.ones(self.cfg.n_decoders, device=self.cfg.device, dtype=self.cfg.dtype)

    @override
    def encoder_norm(self, keepdim: bool = False):
        """Compute the norm of encoder weights averaged across layers."""
        return torch.ones(self.cfg.n_layers, device=self.cfg.device, dtype=self.cfg.dtype)

    @override
    def decoder_bias_norm(self):
        """Compute the norm of decoder bias for each target layer."""
        return torch.ones(self.cfg.n_layers, device=self.cfg.device, dtype=self.cfg.dtype)

    @override
    def set_decoder_to_fixed_norm(self, value: float, force_exact: bool):
        raise NotImplementedError("set_decoder_to_fixed_norm does not make sense for CLT")

    @override
    @torch.no_grad()
    def set_encoder_to_fixed_norm(self, value: float):
        """Set encoder weights to fixed norm."""
        raise NotImplementedError("set_encoder_to_fixed_norm does not make sense for CLT")

    @torch.no_grad()
    def decoder_norm_per_feature(
        self,
        layer: int | None = None,
    ) -> Union[Float[torch.Tensor, "n_layers d_sae"], DTensor]:
        """
        Compute the norm of decoder weights for each feature.
        If layer is not None, only compute the norm for the decoder weights from layer to subsequent layers.
        """

        if self.device_mesh is None:
            decoder_norms = torch.zeros(
                self.cfg.n_layers, self.cfg.d_sae,
                dtype=self.cfg.dtype, 
                device=self.cfg.device, 
            )
        else:
            decoder_norms = torch.distributed.tensor.zeros(
                self.cfg.n_layers, self.cfg.d_sae,
                dtype=self.cfg.dtype, 
                device_mesh=self.device_mesh, 
                placements=self.dim_maps()["decoder_norms"].placements(self.device_mesh)
            )
        if layer is not None:
            for layer_to, decoder_weights in enumerate(self.W_D[layer:]):
                layer_to += layer
                decoder_norms[layer_to] = decoder_weights[layer].pow(2).sum(dim=-1).sqrt()
        else:
            for layer_to, decoder_weights in enumerate(self.W_D):
                decoder_norms[:layer_to + 1] = decoder_norms[:layer_to + 1] + decoder_weights.pow(2).sum(dim=-1)
            decoder_norms = decoder_norms.sqrt()
        return decoder_norms

    def decoder_norm_per_decoder(self) -> Union[Float[torch.Tensor, "n_decoders"], DTensor]:  # noqa: F821
        """Compute the L2 norm of decoder weights for each decoder (layer_from -> layer_to).
        Returns:
            norms: torch.Tensor or DTensor of shape (n_decoders,), where n_decoders = n_layers * (n_layers + 1) // 2
        """
        n_decoders: int = self.cfg.n_layers * (self.cfg.n_layers + 1) // 2
        if self.device_mesh is None:
            decoder_norms = torch.zeros(
                n_decoders, self.cfg.d_sae,
                dtype=self.cfg.dtype,
                device=self.cfg.device,
            )
        else:
            decoder_norms = torch.distributed.tensor.zeros(
                n_decoders, self.cfg.d_sae,
                dtype=self.cfg.dtype,
                device_mesh=self.device_mesh,
                placements=self.dim_maps()["decoder_norms"].placements(self.device_mesh)
            )
        idx = 0
        for layer_to, decoder_weights in enumerate(self.W_D):
            for layer_from in range(layer_to + 1):
                decoder_norms[idx] = decoder_weights[layer_from].pow(2).sum(dim=-1)
                idx += 1
        decoder_norms = decoder_norms.sqrt().mean(dim=-1)
        return decoder_norms

    @override
    @torch.no_grad()
    def standardize_parameters_of_dataset_norm(self):
        """Standardize parameters for dataset-wise normalization during inference."""
        assert self.cfg.norm_activation == "dataset-wise"
        assert self.dataset_average_activation_norm is not None

        def input_norm_factor(layer: int) -> float:
            return math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_points_in[layer]]
        
        def output_norm_factor(layer: int) -> float:
            return math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_points_out[layer]]

        # For CLT, we need to handle multiple input and output layers
        for layer_from in range(self.cfg.n_layers):
            # Adjust encoder bias for this layer
            self.b_E.data[layer_from].div_(input_norm_factor(layer_from))

            if self.cfg.act_fn.lower() == "jumprelu":
                assert isinstance(self.activation_function, JumpReLU)
                threshold = self.activation_function.log_jumprelu_threshold.data[layer_from].exp()
                threshold = threshold / input_norm_factor(layer_from)
                self.activation_function.log_jumprelu_threshold.data[layer_from] = torch.log(threshold)

        for layer_to in range(self.cfg.n_layers):
            self.b_D[layer_to].data.div_(output_norm_factor(layer_to))
            for layer_from in range(layer_to + 1):
                self.W_D[layer_to].data[layer_from].mul_(input_norm_factor(layer_from) / output_norm_factor(layer_to))

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
            x = DTensor.from_local(x, device_mesh=self.device_mesh, placements=[torch.distributed.tensor.Replicate()])
        return x, {}
    
    def prepare_input_single_layer(self, batch: "dict[str, torch.Tensor]", layer: int, **kwargs) -> "tuple[torch.Tensor, dict[str, Any]]":
        """Prepare input tensor from batch by stacking all layer activations from hook_points_in."""
        hook_point_in = self.cfg.hook_points_in[layer]
        if hook_point_in not in batch:
            raise ValueError(f"Missing hook point {hook_point_in} in batch")
        x = batch[hook_point_in]
        if isinstance(self.W_E, DTensor) and not isinstance(x, DTensor):
            assert self.device_mesh is not None
            x = DTensor.from_local(x, device_mesh=self.device_mesh, placements=[torch.distributed.tensor.Replicate()])
        return x, {}

    @override
    def prepare_label(self, batch: "dict[str, torch.Tensor]", **kwargs) -> torch.Tensor:
        """Prepare label tensor from batch using hook_points_out."""
        x_layers = []
        for hook_point in self.cfg.hook_points_out:
            if hook_point not in batch:
                raise ValueError(f"Missing hook point {hook_point} in batch")
            x_layers.append(batch[hook_point])
        labels = torch.stack(x_layers, dim=0)  # (n_layers, ..., d_model)
        return labels
    

    @overload
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        use_batch_norm_mse: bool = False,
        sparsity_loss_type: Literal["power", "tanh", "tanh-quad", None] = None,
        tanh_stretch_coefficient: float = 4.0,
        p: int = 1,
        l1_coefficient: float = 1.0,
        return_aux_data: Literal[True] = True,
        **kwargs,
    ) -> tuple[
        Float[torch.Tensor, " batch"],
        tuple[dict[str, Optional[torch.Tensor]], dict[str, torch.Tensor]],
    ]: ...

    @overload
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        use_batch_norm_mse: bool = False,
        sparsity_loss_type: Literal["power", "tanh", "tanh-quad", None] = None,
        tanh_stretch_coefficient: float = 4.0,
        p: int = 1,
        l1_coefficient: float = 1.0,
        return_aux_data: Literal[False],
        **kwargs,
    ) -> Float[torch.Tensor, " batch"]: ...

    @timer.time("compute_loss")
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        label: (
            Optional[Union[
                Float[torch.Tensor, "batch d_model"],
                Float[torch.Tensor, "batch seq_len d_model"],
            ]]
        ) = None,
        *,
        use_batch_norm_mse: bool = False,
        sparsity_loss_type: Literal["power", "tanh", "tanh-quad", None] = None,
        tanh_stretch_coefficient: float = 4.0,
        frequency_scale: float = 0.01,
        p: int = 1,
        l1_coefficient: float = 1.0,
        return_aux_data: bool = True,
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, " batch"],
        tuple[
            Float[torch.Tensor, " batch"],
            tuple[dict[str, Optional[torch.Tensor]], dict[str, torch.Tensor]],
        ],
    ]:
        """Compute the loss for the autoencoder.
        Ensure that the input activations are normalized by calling `normalize_activations` before calling this method.
        """
        x, encoder_kwargs = self.prepare_input(batch)
        label = self.prepare_label(batch, **kwargs)

        with timer.time("encode"):
            feature_acts = self.encode(x, **encoder_kwargs)

        with timer.time("decode"):
            reconstructed = self.decode(feature_acts, **kwargs)

        with timer.time("loss_calculation"):
            l_rec = (reconstructed - label).pow(2)
            if use_batch_norm_mse:
                l_rec = (
                    l_rec
                    / (label - label.mean(dim=1, keepdim=True)).pow_(2).sum(dim=-1, keepdim=True).clamp(min=1e-8).sqrt()
                )
            l_rec = l_rec.sum(dim=-1)
            if isinstance(l_rec, DTensor):
                l_rec: Tensor = l_rec.full_tensor()
            loss_dict: dict[str, Optional[torch.Tensor]] = {
                "l_rec": l_rec,
            }
            loss = l_rec.mean()

            if sparsity_loss_type is not None:
                decoder_norm: Union[Float[torch.Tensor, "n_layers d_sae"], DTensor] = self.decoder_norm_per_feature()
                with timer.time("sparsity_loss_calculation"):
                    if sparsity_loss_type == "power":
                        l_s = torch.norm(feature_acts * decoder_norm, p=p, dim=-1)
                    elif sparsity_loss_type == "tanh":
                        l_s = torch.tanh(tanh_stretch_coefficient * feature_acts * decoder_norm).sum(dim=-1)
                    elif sparsity_loss_type == "tanh-quad":
                        approx_frequency = einops.reduce(
                            torch.tanh(tanh_stretch_coefficient * feature_acts * decoder_norm),
                            "... d_sae -> d_sae",
                            "mean",
                        )
                        l_s = (approx_frequency * (1 + approx_frequency / frequency_scale)).sum(dim=-1)
                    else:
                        raise ValueError(f"sparsity_loss_type f{sparsity_loss_type} not supported.")
                    if isinstance(l_s, DTensor):
                        l_s = l_s.full_tensor()
                    l_s = l1_coefficient * l_s
                    # WARNING: Some DTensor bugs make if l1_coefficient * l_s goes before full_tensor, the l1_coefficient value will be internally cached. Furthermore, it will cause the backward pass to fail with redistribution error. See https://github.com/pytorch/pytorch/issues/153603 and https://github.com/pytorch/pytorch/issues/153615 .
                    loss_dict["l_s"] = l_s
                    loss = loss + l_s.mean()
            else:
                loss_dict["l_s"] = None

        if return_aux_data:
            aux_data = {
                "feature_acts": feature_acts,
                "reconstructed": reconstructed,
            }
            return loss, (loss_dict, aux_data)
        return loss

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
            "decoder_norms": DimMap({"model": 1}),  # Shard along d_sae dimension
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
            self.W_D[layer_to] = nn.Parameter(state_dict[f"{prefix}{param_name}"].to(self.W_D[layer_to].dtype))

        # Load b_D parameters
        for layer_to in range(self.cfg.n_layers):
            param_name = f"b_D.{layer_to}"
            self.b_D[layer_to] = nn.Parameter(state_dict[f"{prefix}{param_name}"].to(self.b_D[layer_to].dtype))
    

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str,
        strict_loading: bool = True,
        fold_activation_scale: bool = True,
        **kwargs,
    ):
        cfg = CLTConfig.from_pretrained(
            pretrained_name_or_path,
            strict_loading=strict_loading,
            **kwargs,
        )
        model = cls.from_config(
            cfg,
            fold_activation_scale=fold_activation_scale,
        )
        return model
