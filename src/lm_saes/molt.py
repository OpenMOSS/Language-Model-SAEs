import math
from typing import Any, Literal, Union, overload

import torch
import torch.nn as nn
from jaxtyping import Float
from torch.distributed.device_mesh import DeviceMesh
from typing_extensions import override

from .abstract_sae import AbstractSparseAutoEncoder
from .config import MoltConfig


class MixtureOfLinearTransform(AbstractSparseAutoEncoder):
    """Mixture of Linear Transforms (MoLT) model.
    
    MoLT uses d_sae linear transforms, each with its own rank for UtVt decomposition.
    The rank of each transform is determined by the rank_distribution configuration.
    """

    def __init__(self, cfg: MoltConfig, device_mesh: DeviceMesh | None = None) -> None:
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
            self.U_matrices = nn.ModuleDict()
            self.V_matrices = nn.ModuleDict()
            
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
            raise NotImplementedError("Distributed MoLT is not yet implemented")

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
        norms = torch.zeros(self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype)
        
        transform_idx = 0
        for rank in self.cfg.available_ranks:
            rank_str = str(rank)
            if rank_str in self.U_matrices:
                U = self.U_matrices[rank_str]  # (count, d_model, rank)
                V = self.V_matrices[rank_str]  # (count, rank, d_model)
                
                # Compute ||U_i||_F * ||V_i||_F for each transform of this rank
                U_norms = torch.norm(U, p='fro', dim=(1, 2))  # (count,)
                V_norms = torch.norm(V, p='fro', dim=(1, 2))  # (count,)
                UV_norms = U_norms * V_norms  # (count,)
                
                # Assign to the corresponding positions
                count = U.shape[0]
                norms[transform_idx:transform_idx + count] = UV_norms
                transform_idx += count
        
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
        # For simplicity, scale all U and V matrices proportionally
        for U, V in zip(self.U_matrices, self.V_matrices):
            current_norm = torch.norm(U, p='fro') * torch.norm(V, p='fro')
            scale_factor = (value / current_norm) ** 0.5  # Split the scaling between U and V
            if force_exact:
                U.data.mul_(scale_factor)
                V.data.mul_(scale_factor)
            else:
                safe_scale = scale_factor if current_norm >= value else 1.0
                U.data.mul_(safe_scale)
                V.data.mul_(safe_scale)

    @torch.no_grad()
    def set_encoder_to_fixed_norm(self, value: float) -> None:
        self.W_E.mul_(value / self.encoder_norm(keepdim=True))

    @override
    def transform_to_unit_decoder_norm(self) -> None:
        # Set each expert to unit norm
        for U, V in zip(self.U_matrices, self.V_matrices):
            current_norm = torch.norm(U, p='fro') * torch.norm(V, p='fro')
            scale_factor = (1.0 / current_norm) ** 0.5
            U.data.mul_(scale_factor)
            V.data.mul_(scale_factor)

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
        for U, V in zip(self.U_matrices, self.V_matrices):
            U.data.mul_(scale_factor ** 0.5)
            V.data.mul_(scale_factor ** 0.5)
        
        self.cfg.norm_activation = "inference"

    @override
    def init_encoder_with_decoder_transpose(self, factor: float = 1.0) -> None:
        # For MoLT, we can initialize encoder with the first expert's transpose
        if self.cfg.num_experts > 0:
            # Use the first expert (U0 * V0) as initialization
            U0, V0 = self.U_matrices[0], self.V_matrices[0]
            decoder_approx = U0 @ V0  # (d_model, d_model)
            # Take a subset for encoder initialization
            min_dim = min(self.cfg.d_model, self.cfg.d_sae)
            self.W_E.data[:, :min_dim] = decoder_approx[:, :min_dim].T * factor

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
        # MoLT decode: get original input from kwargs
        if "original_x" not in kwargs:
            raise ValueError("MoLT decode requires 'original_x' in kwargs")
        
        x = kwargs["original_x"]
        
        # MoLT decode: sum over i of feature_acts[i] * (U_i @ V_i @ x)
        reconstruction = torch.zeros_like(x)
        
        transform_idx = 0
        ###### TODO: do we need a for loop here?
        for rank in self.cfg.available_ranks:
            rank_str = str(rank)
            if rank_str in self.U_matrices:
                U = self.U_matrices[rank_str]  # (count, d_model, rank)
                V = self.V_matrices[rank_str]  # (count, rank, d_model)
                count = U.shape[0]
                
                # Get feature activations for this rank group
                features = feature_acts[..., transform_idx:transform_idx + count]  # (..., count)
                
                # Compute U_i @ V_i for each transform
                UV = torch.bmm(U, V)  # (count, d_model, d_model)
                
                # Apply each transform to original input x: U_i @ V_i @ x
                # x: (..., d_model), UV: (count, d_model, d_model)
                # Result: (..., count, d_model)
                transformed_x = torch.einsum('...d,cde->...ce', x, UV)  # (..., count, d_model)
                
                # Weight by feature activations: feature_acts[i] * (U_i @ V_i @ x)
                weighted_transforms = features.unsqueeze(-1) * transformed_x  # (..., count, d_model)
                
                # Sum over transforms in this rank group
                reconstruction += weighted_transforms.sum(dim=-2)  # (..., d_model)
                
                transform_idx += count
        
        if self.cfg.use_decoder_bias:
            reconstruction = reconstruction + self.b_D
            
        return reconstruction

    @override
    def init_parameters(self, **kwargs) -> None:
        super().init_parameters(**kwargs)
        
        # Initialize encoder
        nn.init.uniform_(self.W_E, -kwargs["encoder_uniform_bound"], kwargs["encoder_uniform_bound"])
        nn.init.zeros_(self.b_E)
        
        # Initialize U and V matrices for each rank group
        for rank_str in self.U_matrices.keys():
            U = self.U_matrices[rank_str]
            V = self.V_matrices[rank_str]
            nn.init.uniform_(U, -kwargs["decoder_uniform_bound"], kwargs["decoder_uniform_bound"])
            nn.init.uniform_(V, -kwargs["decoder_uniform_bound"], kwargs["decoder_uniform_bound"])
        
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
        # Pass original x to decode through kwargs for MoLT computation
        reconstructed = self.decode(feature_acts, original_x=x, **kwargs)
        return reconstructed

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, strict_loading: bool = True, **kwargs):
        cfg = MoltConfig.from_pretrained(pretrained_name_or_path, strict_loading=strict_loading, **kwargs)
        return cls.from_config(cfg)