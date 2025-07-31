import math
from typing import Any, Literal, Union, overload

import einops
import torch
import torch.nn as nn
from jaxtyping import Float
from torch.distributed.device_mesh import DeviceMesh
from typing_extensions import override

from .abstract_sae import AbstractSparseAutoEncoder
from .config import MOLTConfig


class MixtureOfLinearTransform(AbstractSparseAutoEncoder):
    """Mixture of Linear Transforms (MOLT) model.
    
    MOLT uses d_sae linear transforms, each with its own rank for UtVt decomposition.
    The rank of each transform is determined by the rank_distribution configuration.
    """

    def __init__(self, cfg: MOLTConfig, device_mesh: DeviceMesh | None = None) -> None:
        super().__init__(cfg, device_mesh=device_mesh)
        self.cfg = cfg

        # Generate rank assignment for each linear transform
        self.rank_assignments = self._generate_rank_assignments()

        # Encoder parameters (standard SAE encoder)
        if device_mesh is None:
            self.W_E = nn.Parameter(torch.empty(cfg.d_model, cfg.d_sae, device=cfg.device, dtype=cfg.dtype))
            self.b_E = nn.Parameter(torch.empty(cfg.d_sae, device=cfg.device, dtype=cfg.dtype))
            
            # Decoder parameters: d_sae linear transforms, each with UtVt decomposition
            # Group by rank for efficient parameter storage
            self.U_matrices = nn.ParameterDict()
            self.V_matrices = nn.ParameterDict()
            
            for rank in cfg.available_ranks:
                count = sum(1 for r in self.rank_assignments if r == rank)
                if count > 0:
                    self.U_matrices[str(rank)] = nn.Parameter(
                        torch.empty(count, cfg.d_model, rank, device=cfg.device, dtype=cfg.dtype)
                    )
                    self.V_matrices[str(rank)] = nn.Parameter(
                        torch.empty(count, rank, cfg.d_model, device=cfg.device, dtype=cfg.dtype)
                    )
            
            if cfg.use_decoder_bias:
                self.b_D = nn.Parameter(torch.empty(cfg.d_model, device=cfg.device, dtype=cfg.dtype))
        else:
            # TODO: Implement distributed version if needed
            raise NotImplementedError("Distributed MOLT is not yet implemented")

    def _generate_rank_assignments(self) -> list[int]:
        """Generate rank assignment for each of the d_sae linear transforms."""
        assignments = []
        
        # Assign ranks according to proportions
        for rank, proportion in sorted(self.cfg.rank_distribution.items()):
            count = int(self.cfg.d_sae * proportion)
            assignments.extend([rank] * count)
        
        # Handle any remaining transforms due to rounding
        while len(assignments) < self.cfg.d_sae:
            # Assign remaining to the most common rank
            most_common_rank = max(self.cfg.rank_distribution.keys(), 
                                 key=lambda k: self.cfg.rank_distribution[k])
            assignments.append(most_common_rank)
        
        # Truncate if we have too many (shouldn't happen with proper proportions)
        assignments = assignments[:self.cfg.d_sae]
        
        return assignments

    @override
    def encoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Compute the norm of the encoder weight."""
        return torch.norm(self.W_E, p=2, dim=0, keepdim=keepdim).to(self.cfg.device)

    @override
    def decoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Compute the Frobenius norm of each linear transform's UtVt decomposition."""
        # Pre-compute norms for all rank groups and concatenate
        norm_list = []
        
        for rank in self.cfg.available_ranks:
            rank_str = str(rank)
            if rank_str in self.U_matrices:
                U = self.U_matrices[rank_str]  # (count, d_model, rank)
                V = self.V_matrices[rank_str]  # (count, rank, d_model)
                
                # Compute ||U_i @ V_i||_F for each transform (mathematically correct)
                UV = torch.bmm(U, V)  # (count, d_model, d_model)
                UV_norms = torch.norm(UV.view(UV.shape[0], -1), p='fro', dim=1)  # (count,)
                norm_list.append(UV_norms)
        
        if not norm_list:
            norms = torch.zeros(self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype)
        else:
            # Concatenate all norms in correct order
            norms = torch.cat(norm_list, dim=0)  # (d_sae,)
        
        if keepdim:
            return norms.unsqueeze(-1)
        else:
            return norms

    @override
    def decoder_bias_norm(self) -> torch.Tensor:
        if not self.cfg.use_decoder_bias:
            raise ValueError("Decoder bias is not used")
        return torch.norm(self.b_D, p=2, dim=0, keepdim=True).to(self.cfg.device)

    @override
    def set_decoder_to_fixed_norm(self, value: float, force_exact: bool) -> None:
        # Scale all U and V matrices proportionally
        for rank_str in self.U_matrices.keys():
            U = self.U_matrices[rank_str]  # (count, d_model, rank)
            V = self.V_matrices[rank_str]  # (count, rank, d_model)
            
            # Compute current norms for each transform: ||U_i @ V_i||_F
            UV = torch.bmm(U, V)  # (count, d_model, d_model)
            current_norms = torch.norm(UV.view(UV.shape[0], -1), p='fro', dim=1)  # (count,)
            
            # Compute scale factors (split equally between U and V)
            scale_factors = (value / current_norms) ** 0.5  # Split between U and V
            if not force_exact:
                scale_factors = torch.where(current_norms >= value, scale_factors, 1.0)
            
            # Apply scaling
            U.data.mul_(scale_factors.view(-1, 1, 1))
            V.data.mul_(scale_factors.view(-1, 1, 1))

    @torch.no_grad()
    def set_encoder_to_fixed_norm(self, value: float) -> None:
        self.W_E.mul_(value / self.encoder_norm(keepdim=True))

    @override
    def transform_to_unit_decoder_norm(self) -> None:
        # Set each transform to unit norm
        for rank_str in self.U_matrices.keys():
            U = self.U_matrices[rank_str]  # (count, d_model, rank)
            V = self.V_matrices[rank_str]  # (count, rank, d_model)
            
            # Compute current norms for each transform: ||U_i @ V_i||_F
            UV = torch.bmm(U, V)  # (count, d_model, d_model)
            current_norms = torch.norm(UV.view(UV.shape[0], -1), p='fro', dim=1)  # (count,)
            
            # Scale to unit norm (split equally between U and V)
            scale_factors = (1.0 / current_norms) ** 0.5
            U.data.mul_(scale_factors.view(-1, 1, 1))
            V.data.mul_(scale_factors.view(-1, 1, 1))

    @override
    def standardize_parameters_of_dataset_norm(self, dataset_average_activation_norm: dict[str, float] | None) -> None:
        # Similar to SAE standardization
        assert self.cfg.norm_activation == "dataset-wise"
        assert self.dataset_average_activation_norm is not None or dataset_average_activation_norm is not None
        if dataset_average_activation_norm is not None:
            self.set_dataset_average_activation_norm(dataset_average_activation_norm)
        assert self.dataset_average_activation_norm is not None
        
        input_norm_factor: float = (
            math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_point_in]
        )
        output_norm_factor: float = (
            math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_point_out]
        )
        
        self.b_E.div_(input_norm_factor)
        if self.cfg.use_decoder_bias:
            self.b_D.div_(output_norm_factor)
        
        # Scale decoder matrices
        scale_factor = input_norm_factor / output_norm_factor
        for rank_str in self.U_matrices.keys():
            self.U_matrices[rank_str].data.mul_(scale_factor ** 0.5)
            self.V_matrices[rank_str].data.mul_(scale_factor ** 0.5)
        
        self.cfg.norm_activation = "inference"

    @override
    def init_encoder_with_decoder_transpose(self, factor: float = 1.0):
        raise NotImplementedError("init_encoder_with_decoder_transpose does not make sense for MOLT")

    @override
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
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
        # Standard encoder: ϕ(et ⋅ x − bt)
        hidden_pre = x @ self.W_E + self.b_E
        
        # Apply activation function with decoder norm for sparsity
        if self.cfg.sparsity_include_decoder_norm:
            sparsity_scores = hidden_pre * self.decoder_norm()
        else:
            sparsity_scores = hidden_pre

        activation_mask = self.activation_function(sparsity_scores)
        feature_acts = hidden_pre * activation_mask

        if return_hidden_pre:
            return feature_acts, hidden_pre
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
    ]:
        # MOLT decode: get original input from kwargs
        if "original_x" not in kwargs:
            raise ValueError("MOLT decode requires 'original_x' in kwargs")
        
        x = kwargs["original_x"]
        
        # Pre-compute all U @ V matrices for better efficiency
        UV_list = []
        count_list = []
        
        for rank in self.cfg.available_ranks:
            rank_str = str(rank)
            if rank_str in self.U_matrices:
                U = self.U_matrices[rank_str]  # (count, d_model, rank)
                V = self.V_matrices[rank_str]  # (count, rank, d_model)
                
                # Compute U_i @ V_i for each transform using batch matrix multiplication
                UV = torch.bmm(U, V)  # (count, d_model, d_model)
                UV_list.append(UV)
                count_list.append(UV.shape[0])
        
        if not UV_list:
            return torch.zeros_like(x)
        
        # Concatenate all transforms for batch processing
        UV_all = torch.cat(UV_list, dim=0)  # (d_sae, d_model, d_model)
        
        # Apply all transforms to input x in one operation using einops
        # x: (..., d_model), UV_all: (d_sae, d_model, d_model)
        # Result: (..., d_sae, d_model)
        transformed_x_all = einops.einsum(
            x, UV_all, 
            '... d_model, d_sae d_model d_model -> ... d_sae d_model'
        )
        
        # Weight by feature activations: feature_acts[i] * (U_i @ V_i @ x)
        weighted_transforms = einops.einsum(
            feature_acts, transformed_x_all,
            '... d_sae, ... d_sae d_model -> ... d_sae d_model'
        )
        
        # Sum over all transforms to get final reconstruction
        reconstruction = einops.reduce(
            weighted_transforms, 
            '... d_sae d_model -> ... d_model', 
            'sum'
        )
        
        if self.cfg.use_decoder_bias:
            reconstruction = reconstruction + self.b_D
            
        return reconstruction

    @override
    def init_parameters(self, **kwargs) -> None:
        super().init_parameters(**kwargs)
        
        # Initialize encoder
        # nn.init.uniform_(self.W_E, -kwargs["encoder_uniform_bound"], kwargs["encoder_uniform_bound"])
        # nn.init.zeros_(self.b_E)

        encoder_bound = 1.0 / math.sqrt(self.cfg.d_sae)
        nn.init.uniform_(self.W_E, -encoder_bound, encoder_bound)
        nn.init.zeros_(self.b_E)
        
        # Initialize U and V matrices for each rank group
        for rank_str in self.U_matrices.keys():
            U = self.U_matrices[rank_str]
            V = self.V_matrices[rank_str]
            # nn.init.uniform_(U, -kwargs["decoder_uniform_bound"], kwargs["decoder_uniform_bound"])
            # nn.init.uniform_(V, -kwargs["decoder_uniform_bound"], kwargs["decoder_uniform_bound"])
            decoder_bound = 1.0 / math.sqrt(self.cfg.d_model * int(rank_str))
            nn.init.uniform_(U, -decoder_bound, decoder_bound)
            nn.init.uniform_(V, -decoder_bound, decoder_bound)

        if self.cfg.use_decoder_bias:
            nn.init.zeros_(self.b_D)

    @override
    def prepare_input(self, batch: dict[str, torch.Tensor], **kwargs) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        x = batch[self.cfg.hook_point_in]
        encoder_kwargs = {}
        decoder_kwargs = {"original_x": x}  # Pass original input to decoder for MoLT
        return x, encoder_kwargs, decoder_kwargs

    @override
    def prepare_label(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        return batch[self.cfg.hook_point_out]

    @override
    def forward(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, "batch d_model"],
        Float[torch.Tensor, "batch seq_len d_model"],
    ]:
        feature_acts = self.encode(x, **kwargs)
        # Pass original x to decode through kwargs for MOLT computation
        reconstructed = self.decode(feature_acts, original_x=x, **kwargs)
        return reconstructed

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, strict_loading: bool = True, **kwargs):
        cfg = MOLTConfig.from_pretrained(pretrained_name_or_path, strict_loading=strict_loading, **kwargs)
        return cls.from_config(cfg)