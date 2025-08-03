import math
from collections import Counter
from typing import Any, Union

import torch
import torch.distributed.tensor
import torch.nn as nn
from jaxtyping import Float
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from typing_extensions import override

from .abstract_sae import AbstractSparseAutoEncoder
from .config import MOLTConfig
from .utils.distributed import DimMap
from .utils.logging import get_distributed_logger
from .utils.timer import timer

logger = get_distributed_logger("molt")


class MixtureOfLinearTransform(AbstractSparseAutoEncoder):
    """Mixture of Linear Transforms (MOLT) model.
    
    MOLT is a sparse autoencoder variant that uses d_sae linear transforms, each with 
    its own rank for UtVt decomposition. This approach provides more flexibility in 
    modeling complex patterns while maintaining sparsity.
    
    Mathematical Formulation:
    - Encoder: ϕ(et · x - bt) where ϕ is the activation function
    - Decoder: Σᵢ fᵢ · (Uᵢ @ Vᵢ @ x) where fᵢ are feature activations
    - Decoder norm: ||UᵢVᵢ||_F for each transform i
    
    The rank of each transform is determined by the rank_distribution configuration,
    allowing for adaptive model capacity allocation.
    """

    def __init__(self, cfg: MOLTConfig, device_mesh: DeviceMesh | None = None) -> None:
        super().__init__(cfg, device_mesh=device_mesh)
        self.cfg = cfg

        # Generate rank assignment for each linear transform
        if device_mesh is not None:
            # In distributed training, get local rank assignments
            # Use model dimension for tensor parallelism
            model_dim_index = device_mesh.mesh_dim_names.index("model")
            local_rank = device_mesh.get_local_rank(mesh_dim=model_dim_index)
            model_parallel_size = device_mesh.size(mesh_dim=model_dim_index)
            self.rank_assignments = cfg.get_local_rank_assignments(
                local_rank, model_parallel_size
            )
            print(f"[DEBUG] Local rank {local_rank}: {len(self.rank_assignments)} assignments, "
                  f"rank counts: {dict(sorted(Counter(self.rank_assignments).items()))}")
        else:
            # Non-distributed case
            self.rank_assignments = cfg.generate_rank_assignments()

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
                # Always create parameters for all rank types for consistency
                # In non-distributed case, we can skip empty tensors
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
            # Distributed initialization
            w_e_placements = self.dim_maps()["W_E"].placements(device_mesh)
            b_e_placements = self.dim_maps()["b_E"].placements(device_mesh)
            print(f"[DEBUG] W_E placements: {w_e_placements}")
            print(f"[DEBUG] b_E placements: {b_e_placements}")
            print(f"[DEBUG] Device mesh: {device_mesh}")
            print(f"[DEBUG] dim_maps: {self.dim_maps()}")
            
            self.W_E = nn.Parameter(
                torch.distributed.tensor.empty(
                    cfg.d_model,
                    cfg.d_sae,
                    dtype=cfg.dtype,
                    device_mesh=device_mesh,
                    placements=w_e_placements,
                )
            )
            
            self.b_E = nn.Parameter(
                torch.distributed.tensor.empty(
                    cfg.d_sae,
                    dtype=cfg.dtype,
                    device_mesh=device_mesh,
                    placements=b_e_placements,
                )
            )
            
            print(f"[DEBUG] W_E local shape: {self.W_E.to_local().shape}")
            print(f"[DEBUG] b_E local shape: {self.b_E.to_local().shape}")
            
            # Decoder parameters: d_sae linear transforms, each with UtVt decomposition
            # Group by rank for efficient parameter storage
            self.U_matrices = nn.ParameterDict()
            self.V_matrices = nn.ParameterDict()
            
            for rank in cfg.available_ranks:
                count = sum(1 for r in self.rank_assignments if r == rank)
                # Each device has all rank groups with count = global_count / model_parallel_size
                # count should never be 0 due to proper sharding logic
                assert count > 0, f"Rank {rank} has count=0, sharding logic error"
                
                self.U_matrices[str(rank)] = nn.Parameter(
                    torch.distributed.tensor.empty(
                        count,
                        cfg.d_model,
                        rank,
                        dtype=cfg.dtype,
                        device_mesh=device_mesh,
                        placements=self.dim_maps()["U_matrices"].placements(device_mesh),
                    )
                )
                
                self.V_matrices[str(rank)] = nn.Parameter(
                    torch.distributed.tensor.empty(
                        count,
                        rank,
                        cfg.d_model,
                        dtype=cfg.dtype,
                        device_mesh=device_mesh,
                        placements=self.dim_maps()["V_matrices"].placements(device_mesh),
                    )
                )
            
            if cfg.use_decoder_bias:
                self.b_D = nn.Parameter(
                    torch.distributed.tensor.empty(
                        cfg.d_model,
                        dtype=cfg.dtype,
                        device_mesh=device_mesh,
                        placements=self.dim_maps()["b_D"].placements(device_mesh),
                    )
                )

    def dim_maps(self) -> dict[str, DimMap]:
        """Return dimension maps for distributed training.
        
        Encoder and decoder use consistent sharding:
        - W_E sharded along d_sae (output) dimension  
        - U/V matrices sharded along transform count (first) dimension
        This ensures feature_acts from encoder can directly feed decoder without redistribution.
        """
        base_maps = super().dim_maps()
        
        molt_maps = {
            "W_E": DimMap({"model": 1}),  # Shard along d_sae dimension
            "b_E": DimMap({"model": 0}),  # Shard along d_sae dimension
            
            # U and V matrices sharded along transform count dimension
            # This matches the W_E sharding pattern for feature_acts compatibility
            "U_matrices": DimMap({"model": 0}),  # Shard along transform count
            "V_matrices": DimMap({"model": 0}),  # Shard along transform count
            
            "b_D": DimMap({}),  # Replicate decoder bias
        }
        
        return base_maps | molt_maps

    @override
    @timer.time("encoder_norm")
    def encoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Compute the norm of the encoder weight."""
        if not isinstance(self.W_E, DTensor):
            return torch.norm(self.W_E, p=2, dim=0, keepdim=keepdim).to(self.cfg.device)
        else:
            assert self.device_mesh is not None
            return DTensor.from_local(
                torch.norm(self.W_E.to_local(), p=2, dim=0, keepdim=keepdim),
                device_mesh=self.device_mesh,
                placements=DimMap({"model": 1 if keepdim else 0}).placements(self.device_mesh),
            )

    @override
    @timer.time("decoder_norm")
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
    @timer.time("decoder_bias_norm")
    def decoder_bias_norm(self) -> torch.Tensor:
        assert self.cfg.use_decoder_bias, "Decoder bias is not used"
        if not isinstance(self.b_D, DTensor):
            return torch.norm(self.b_D, p=2, dim=0, keepdim=True).to(self.cfg.device)
        else:
            assert self.device_mesh is not None
            return DTensor.from_local(
                torch.norm(self.b_D.to_local(), p=2, dim=0, keepdim=True),
                device_mesh=self.device_mesh,
                placements=DimMap({}).placements(self.device_mesh),
            )

    @override
    @timer.time("set_decoder_to_fixed_norm")
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
    @timer.time("set_encoder_to_fixed_norm")
    def set_encoder_to_fixed_norm(self, value: float) -> None:
        self.W_E.mul_(value / self.encoder_norm(keepdim=True))

    @override
    @timer.time("transform_to_unit_decoder_norm")
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
    @timer.time("standardize_parameters_of_dataset_norm")
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
    @timer.time("init_encoder_with_decoder_transpose")
    def init_encoder_with_decoder_transpose(self, factor: float = 1.0):
        raise NotImplementedError("init_encoder_with_decoder_transpose does not make sense for MOLT")

    @override
    @timer.time("encode")
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

    def _decode_single_gpu(
        self,
        feature_acts: Union[
            Float[torch.Tensor, "batch d_sae"],
            Float[torch.Tensor, "batch seq_len d_sae"],
        ],
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
    ) -> Union[
        Float[torch.Tensor, "batch d_model"],
        Float[torch.Tensor, "batch seq_len d_model"],
    ]:
        """Decode features using single GPU (non-distributed) computation."""
        reconstruction = torch.zeros_like(x)
        feature_idx = 0
        
        for rank in self.cfg.available_ranks:
            rank_str = str(rank)
            if rank_str in self.V_matrices:
                V = self.V_matrices[rank_str]  # (count, rank, d_model)
                U = self.U_matrices[rank_str]  # (count, d_model, rank)
                local_count = V.shape[0]
                
                if local_count > 0:
                    # Extract features for this rank group
                    curr_features = feature_acts[..., feature_idx:feature_idx+local_count]
                    feature_idx += local_count
                    
                    # Compute: f_i * (U_i @ V_i @ x)
                    # Step 1: V @ x (project input through V matrices)
                    Vx = torch.einsum('... d, c r d -> ... c r', x, V)
                    
                    # Step 2: Weight by feature activations
                    weighted_Vx = curr_features.unsqueeze(-1) * Vx
                    
                    # Step 3: U @ (weighted V @ x)
                    reconstruction += torch.einsum('... c r, c d r -> ... d', weighted_Vx, U)
        
        return reconstruction

    def _decode_distributed(
        self,
        feature_acts: Union[
            Float[torch.Tensor, "batch d_sae"],
            Float[torch.Tensor, "batch seq_len d_sae"],
        ],
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
    ) -> Union[
        Float[torch.Tensor, "batch d_model"],
        Float[torch.Tensor, "batch seq_len d_model"],
    ]:
        """Decode features using distributed computation.
        
        With consistent encoder/decoder sharding, feature_acts from encoder
        directly matches decoder sharding without redistribution needed.
        """
        reconstruction = torch.zeros_like(x)
        feature_idx = 0
        
        for rank in self.cfg.available_ranks:
            rank_str = str(rank)
            if rank_str in self.V_matrices:
                V = self.V_matrices[rank_str]  # (local_count, rank, d_model)
                U = self.U_matrices[rank_str]  # (local_count, d_model, rank)
                
                # Get local count from distributed tensor
                local_count = V.to_local().shape[0]
                
                if local_count > 0:
                    # Extract features for this rank group
                    curr_features = feature_acts[..., feature_idx:feature_idx+local_count]
                    feature_idx += local_count
                    
                    # Convert to local tensors for computation
                    V_local = V.to_local()
                    U_local = U.to_local()
                    x_local = x.to_local()
                    curr_features_local = curr_features.to_local()
                    
                    # Compute: f_i * (U_i @ V_i @ x)
                    # Step 1: V @ x (project input through V matrices)
                    Vx = torch.einsum('... d, c r d -> ... c r', x_local, V_local)
                    
                    # Step 2: Weight by feature activations
                    weighted_Vx = curr_features_local.unsqueeze(-1) * Vx
                    
                    # Step 3: U @ (weighted V @ x)
                    local_reconstruction = torch.einsum('... c r, c d r -> ... d', weighted_Vx, U_local)
                    
                    # Convert back to DTensor if needed
                    if isinstance(reconstruction, DTensor):
                        local_reconstruction = DTensor.from_local(
                            local_reconstruction,
                            device_mesh=self.device_mesh,
                            placements=reconstruction.placements,
                        )
                    
                    reconstruction += local_reconstruction
        
        return reconstruction

    @override
    @timer.time("decode")
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
        """Decode feature activations back to model space using MOLT transforms.
        
        Args:
            feature_acts: Feature activations from encode()
            **kwargs: Must contain 'original_x' - the original input tensor
            
        Returns:
            Reconstructed tensor in model space
        """
        assert "original_x" in kwargs, "MOLT decode requires 'original_x' in kwargs"
    
        x = kwargs["original_x"]
        
        # Choose decoding strategy based on distributed setup
        is_distributed = any(
            isinstance(self.U_matrices[str(rank)], DTensor) 
            for rank in self.cfg.available_ranks 
            if str(rank) in self.U_matrices
        )
        
        if is_distributed:
            reconstruction = self._decode_distributed(feature_acts, x)
        else:
            reconstruction = self._decode_single_gpu(feature_acts, x)
        
        # Add decoder bias if configured
        if self.cfg.use_decoder_bias:
            reconstruction = reconstruction + self.b_D
            
        return reconstruction

    @override
    @timer.time("init_parameters")
    def init_parameters(self, **kwargs) -> None:
        super().init_parameters(**kwargs)
        
        # Initialize encoder
        encoder_bound = 1.0 / math.sqrt(self.cfg.d_sae)
        
        if self.device_mesh is None:
            # Non-distributed initialization
            W_E = torch.empty(self.cfg.d_model, self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype).uniform_(-encoder_bound, encoder_bound)
            b_E = torch.zeros(self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype)
        else:
            # Distributed initialization
            W_E_local = torch.empty(self.cfg.d_model, self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype).uniform_(-encoder_bound, encoder_bound)
            W_E = self.dim_maps()["W_E"].distribute(W_E_local, self.device_mesh)
            b_E = self.dim_maps()["b_E"].distribute(torch.zeros(self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype), self.device_mesh)
        
        self.W_E.copy_(W_E)
        self.b_E.copy_(b_E)
        
        # Initialize U and V matrices for each rank group
        for rank_str in self.U_matrices.keys():
            rank = int(rank_str)
            U = self.U_matrices[rank_str]
            V = self.V_matrices[rank_str]
            
            # Xavier initialization considering the rank
            U_bound = 1.0 / math.sqrt(self.cfg.d_model) ## TODO: check this
            V_bound = 1.0 / math.sqrt(self.cfg.d_model) ## TODO: check this
            
            if self.device_mesh is None:
                # Non-distributed initialization
                U_local = torch.empty(U.shape, device=self.cfg.device, dtype=self.cfg.dtype).uniform_(-U_bound, U_bound)
                V_local = torch.empty(V.shape, device=self.cfg.device, dtype=self.cfg.dtype).uniform_(-V_bound, V_bound)
            else:
                # Distributed initialization
                U_local = torch.empty(U.shape, device=self.cfg.device, dtype=self.cfg.dtype).uniform_(-U_bound, U_bound)
                V_local = torch.empty(V.shape, device=self.cfg.device, dtype=self.cfg.dtype).uniform_(-V_bound, V_bound)  ## TODO
                U_local = self.dim_maps()["U_matrices"].distribute(U_local, self.device_mesh)
                V_local = self.dim_maps()["V_matrices"].distribute(V_local, self.device_mesh)
            
            U.copy_(U_local)
            V.copy_(V_local)

        if self.cfg.use_decoder_bias:
            if self.device_mesh is None:
                # Non-distributed initialization
                b_D = torch.zeros(self.cfg.d_model, device=self.cfg.device, dtype=self.cfg.dtype)
            else:
                # Distributed initialization
                b_D = self.dim_maps()["b_D"].distribute(torch.zeros(self.cfg.d_model, device=self.cfg.device, dtype=self.cfg.dtype), self.device_mesh)
            self.b_D.copy_(b_D)

    @override
    @timer.time("prepare_input")
    def prepare_input(self, batch: dict[str, torch.Tensor], **kwargs) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        x = batch[self.cfg.hook_point_in]
        encoder_kwargs = {}
        decoder_kwargs = {"original_x": x}  # Pass original input to decoder for MoLT
        return x, encoder_kwargs, decoder_kwargs

    @override
    @timer.time("prepare_label")
    def prepare_label(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        return batch[self.cfg.hook_point_out]

    @override
    @timer.time("forward")
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

    @override
    def load_distributed_state_dict(
        self, state_dict: dict[str, torch.Tensor], device_mesh: DeviceMesh, prefix: str = ""
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

        # Load U and V matrices for each rank group
        for rank_str in self.U_matrices.keys():
            U_param_name = f"U_matrices.{rank_str}"
            V_param_name = f"V_matrices.{rank_str}"
            self.U_matrices[rank_str] = nn.Parameter(state_dict[f"{prefix}{U_param_name}"].to(self.U_matrices[rank_str].dtype))
            self.V_matrices[rank_str] = nn.Parameter(state_dict[f"{prefix}{V_param_name}"].to(self.V_matrices[rank_str].dtype))

        # Load decoder bias
        if self.cfg.use_decoder_bias:
            self.b_D = nn.Parameter(state_dict[f"{prefix}b_D"].to(self.b_D.dtype))

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, strict_loading: bool = True, **kwargs):
        cfg = MOLTConfig.from_pretrained(pretrained_name_or_path, strict_loading=strict_loading, **kwargs)
        return cls.from_config(cfg)