import math
from typing import Any, Literal, Union, cast, overload

import torch
import torch.distributed.tensor
import torch.nn as nn
from jaxtyping import Float
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from typing_extensions import override

from lm_saes.abstract_sae import (
    AbstractSparseAutoEncoder,
    BaseSAEConfig,
    register_sae_config,
    register_sae_model,
)
from lm_saes.utils.distributed import DimMap, item
from lm_saes.utils.logging import get_distributed_logger
from lm_saes.utils.timer import timer

logger = get_distributed_logger("molt")


@register_sae_config("molt")
class MOLTConfig(BaseSAEConfig):
    """Configuration for Mixture of Linear Transforms (MOLT).

    MOLT is a more efficient alternative to transcoders that sparsely replaces
    MLP computation in transformers. It converts dense MLP layers into sparse,
    interpretable linear transforms.
    """

    sae_type: str = "molt"
    hook_point_in: str
    """Hook point to capture input activations from."""
    hook_point_out: str
    """Hook point to output activations to."""
    rank_counts: dict[int, int]
    """Dictionary mapping rank values to their integer counts.
    Example: {4: 128, 8: 256, 16: 128} means 128 transforms of rank 4, 256 transforms of rank 8, and 128 transforms of rank 16.
    """

    def model_post_init(self, __context):
        super().model_post_init(__context)
        # Validate counts
        assert self.rank_counts, "rank_counts cannot be empty"

        for rank, count in self.rank_counts.items():
            assert rank > 0, f"Rank must be positive, got {rank}"
            assert count > 0, f"Count for rank {rank} must be positive, got {count}"

        # Workaround: expansion_factor is not used in MOLT, but we keep it for consistency with other SAE variants.
        assert abs(self.expansion_factor - self.d_sae / self.d_model) < 0.1, (
            f"Expansion factor {self.expansion_factor} is not close to d_sae / d_model {self.d_sae / self.d_model}"
        )

    def generate_rank_assignments(self) -> list[int]:
        """Generate rank assignment for each of the d_sae linear transforms.

        Returns:
            List of rank assignments for each transform.
            For example: [1, 1, 1, 1, 2, 2, 4].
        """
        assignments = []
        for rank in sorted(self.rank_counts.keys()):
            assignments.extend([rank] * self.rank_counts[rank])
        return assignments

    def get_local_rank_assignments(self, model_parallel_size: int) -> list[int]:
        """Get rank assignments for a specific local device in distributed running.

        Each device gets all rank groups, with each group evenly divided across devices.
        This ensures consistent encoder/decoder sharding without feature_acts redistribution.

        Args:
            model_parallel_size: Number of model parallel devices for training and inference.

        Returns:
            List of rank assignments for this local device
            For example:
            global_rank_assignments = [1, 1, 2, 2], model_parallel_size = 2 -> local_rank_assignments = [1, 2]
        """
        local_assignments = []
        for rank in sorted(self.rank_counts.keys()):
            global_count = self.rank_counts[rank]

            # Verify even division
            assert global_count % model_parallel_size == 0, (
                f"Transform rank {rank} global count {global_count} not divisible by "
                f"model_parallel_size {model_parallel_size}"
            )

            local_count = global_count // model_parallel_size
            local_assignments.extend([rank] * local_count)

        return local_assignments

    @property
    @override
    def d_sae(self) -> int:
        """Calculate d_sae based on total rank counts."""
        return sum(self.rank_counts.values())

    @property
    def available_ranks(self) -> list[int]:
        """Get sorted list of available ranks."""
        return sorted(self.rank_counts.keys())

    @property
    def num_rank_types(self) -> int:
        """Number of different rank types."""
        return len(self.rank_counts)

    @property
    def associated_hook_points(self) -> list[str]:
        return [self.hook_point_in, self.hook_point_out]


@register_sae_model("molt")
class MixtureOfLinearTransform(AbstractSparseAutoEncoder):
    """Mixture of Linear Transforms (MOLT) model.

    MOLT is a sparse autoencoder variant that uses d_sae linear transforms,
    each with its own rank for UtVt decomposition.

    Mathematical Formulation:
    - Encoder: ϕ(et · x - bt) where ϕ is the activation function
    - Decoder: Σᵢ fᵢ · (Uᵢ @ Vᵢ @ x) where fᵢ are feature activations
    - Decoder norm: ||UᵢVᵢ||_F for each transform i

    The rank of each transform is determined by the rank_counts configuration,
    allowing for adaptive model capacity allocation.
    """

    def __init__(self, cfg: MOLTConfig, device_mesh: DeviceMesh | None = None) -> None:
        super().__init__(cfg, device_mesh=device_mesh)
        self.cfg = cfg

        # Generate rank assignment for each linear transform
        if device_mesh is not None:
            # In distributed training/inference, get local rank assignments
            # Use model dimension for tensor parallelism
            mesh_dim_names = device_mesh.mesh_dim_names
            if mesh_dim_names is None:
                model_dim_index = 0
            else:
                model_dim_index = mesh_dim_names.index("model") if "model" in mesh_dim_names else 0
            local_rank = device_mesh.get_local_rank(
                mesh_dim=model_dim_index
            )  # this rank stands for device rank of this process
            model_parallel_size = device_mesh.size(mesh_dim=model_dim_index)

            self.rank_assignments = cfg.get_local_rank_assignments(model_parallel_size)

            for k, v in cfg.rank_counts.items():
                logger.info(
                    f"Rank {k} has {v} global transforms, device rank {local_rank} has {self.rank_assignments.count(k)} transforms"
                )
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

            # Decoder parameters: d_sae linear transforms, each with UtVt decomposition
            # Group by rank for efficient parameter storage
            self.U_matrices = nn.ParameterDict()
            self.V_matrices = nn.ParameterDict()

            for rank in cfg.available_ranks:
                local_count = sum(1 for r in self.rank_assignments if r == rank)
                assert local_count > 0, f"Rank {rank} has local_count=0, sharding logic error"

                # Create DTensor with GLOBAL shape
                self.U_matrices[str(rank)] = nn.Parameter(
                    torch.distributed.tensor.empty(
                        self.cfg.rank_counts[rank],  # GLOBAL count
                        cfg.d_model,
                        rank,
                        dtype=cfg.dtype,
                        device_mesh=device_mesh,
                        placements=self.dim_maps()["U_matrices"].placements(device_mesh),
                    )
                )

                self.V_matrices[str(rank)] = nn.Parameter(
                    torch.distributed.tensor.empty(
                        self.cfg.rank_counts[rank],  # GLOBAL count
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

                assert isinstance(U, DTensor) == isinstance(V, DTensor), "U and V must have the same type"
                # Handle DTensor case - work with local shards
                if isinstance(U, DTensor) and isinstance(V, DTensor):
                    U_local = U.to_local()
                    V_local = V.to_local()

                    # Compute ||U_i @ V_i||_F for each transform (local shard)
                    UV_local = torch.bmm(U_local, V_local)  # (local_count, d_model, d_model)
                    UV_norms_local = torch.norm(UV_local.view(UV_local.shape[0], -1), p="fro", dim=1)  # (local_count,)

                    # Convert back to DTensor with proper placement
                    assert self.device_mesh is not None
                    UV_norms = DTensor.from_local(
                        UV_norms_local,
                        device_mesh=self.device_mesh,
                        placements=self.dim_maps()["U_matrices"].placements(self.device_mesh)[
                            0:1
                        ],  # Only keep first dimension placement
                    )
                    norm_list.append(UV_norms)
                else:
                    # Non-distributed case
                    UV = torch.bmm(U, V)  # (count, d_model, d_model)
                    UV_norms = torch.norm(UV.view(UV.shape[0], -1), p="fro", dim=1)  # (count,)
                    norm_list.append(UV_norms)

        if not norm_list:
            if self.device_mesh is not None:
                # Create replicated DTensor for zero norms
                norms = DTensor.from_local(
                    torch.zeros(self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype),
                    device_mesh=self.device_mesh,
                    placements=self.dim_maps()["b_E"].placements(self.device_mesh),  # Same as b_E sharding
                )
            else:
                norms = torch.zeros(self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype)
        else:
            # Concatenate all norms in correct order
            if isinstance(norm_list[0], DTensor):
                # CRITICAL FIX: Avoid full_tensor() to prevent numerical errors
                # Instead, directly concatenate the DTensors which preserves numerical precision
                assert self.device_mesh is not None

                # Convert each DTensor norm to local tensor and concatenate locally
                local_norms = [norm.to_local() for norm in norm_list]

                # Concatenate local norms and convert back to DTensor
                norms_local = torch.cat(local_norms, dim=0)
                norms = DTensor.from_local(
                    norms_local,
                    device_mesh=self.device_mesh,
                    placements=self.dim_maps()["b_E"].placements(self.device_mesh),  # Same as b_E (d_sae dimension)
                )
            else:
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
    @torch.no_grad()
    def set_decoder_to_fixed_norm(self, value: float, force_exact: bool) -> None:
        # Scale all U and V matrices proportionally
        for rank_str in self.U_matrices.keys():
            U = self.U_matrices[rank_str]  # (count, d_model, rank)
            V = self.V_matrices[rank_str]  # (count, rank, d_model)

            # Compute current norms for each transform: ||U_i @ V_i||_F
            UV = torch.bmm(U, V)  # (count, d_model, d_model)
            current_norms = torch.norm(UV.view(UV.shape[0], -1), p="fro", dim=1)  # (count,)

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
        """Set encoder weights to a fixed norm."""
        self.W_E.mul_(value / self.encoder_norm(keepdim=True))

    @override
    @timer.time("transform_to_unit_decoder_norm")
    @torch.no_grad()
    def transform_to_unit_decoder_norm(self) -> None:
        # Set each transform to unit norm
        for rank_str in self.U_matrices.keys():
            U = self.U_matrices[rank_str]  # (count, d_model, rank)
            V = self.V_matrices[rank_str]  # (count, rank, d_model)

            # Compute current norms for each transform: ||U_i @ V_i||_F
            UV = torch.bmm(U, V)  # (count, d_model, d_model)
            current_norms = torch.norm(UV.view(UV.shape[0], -1), p="fro", dim=1)  # (count,)

            # Scale to unit norm (split equally between U and V)
            scale_factors = (1.0 / current_norms) ** 0.5
            U.mul_(scale_factors.view(-1, 1, 1))
            V.mul_(scale_factors.view(-1, 1, 1))

    @override
    @timer.time("standardize_parameters_of_dataset_norm")
    def standardize_parameters_of_dataset_norm(self) -> None:
        # Similar to SAE standardization
        assert self.cfg.norm_activation == "dataset-wise"
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
            self.U_matrices[rank_str].data.mul_(scale_factor**0.5)
            self.V_matrices[rank_str].data.mul_(scale_factor**0.5)

        self.cfg.norm_activation = "inference"

    @override
    @timer.time("init_encoder_with_decoder_transpose")
    def init_encoder_with_decoder_transpose(self, factor: float = 1.0):
        raise NotImplementedError("init_encoder_with_decoder_transpose does not make sense for MOLT")

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
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        return_hidden_pre: Literal[True],
        **kwargs,
    ) -> tuple[
        Union[
            Float[torch.Tensor, "batch d_sae"],
            Float[torch.Tensor, "batch seq_len d_sae"],
        ],
        Union[
            Float[torch.Tensor, "batch d_sae"],
            Float[torch.Tensor, "batch seq_len d_sae"],
        ],
    ]: ...

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

        # Scale feature activations by decoder norm if configured
        if self.cfg.sparsity_include_decoder_norm:
            hidden_pre = hidden_pre * self.decoder_norm()

        feature_acts = self.activation_function(hidden_pre)

        if self.cfg.sparsity_include_decoder_norm:
            feature_acts = feature_acts / self.decoder_norm()
            hidden_pre = hidden_pre / self.decoder_norm()

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
                    curr_features = feature_acts[..., feature_idx : feature_idx + local_count]
                    feature_idx += local_count

                    # Compute: f_i * (U_i @ V_i @ x)
                    # Step 1: V @ x (project input through V matrices)
                    Vx = torch.einsum("... d, c r d -> ... c r", x, V)

                    # Step 2: Weight by feature activations
                    weighted_Vx = curr_features.unsqueeze(-1) * Vx

                    # Step 3: U @ (weighted V @ x)
                    local_reconstruction = torch.einsum("... c r, c d r -> ... d", weighted_Vx, U)

                    reconstruction += local_reconstruction

        # Add decoder bias if configured
        if self.cfg.use_decoder_bias:
            reconstruction = reconstruction + self.b_D

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
        assert self.device_mesh is not None
        mesh = cast(DeviceMesh, self.device_mesh)
        x_local = x.to_local() if isinstance(x, DTensor) else x
        reconstruction_local = torch.zeros_like(x_local)  # (..., d_model)

        # Convert feature_acts to local tensor for processing
        assert isinstance(feature_acts, DTensor), "feature_acts was expected to be a DTensor"
        feature_acts_local = feature_acts.to_local()

        # Track local feature index within this GPU's shard
        local_feature_idx = 0

        for rank in self.cfg.available_ranks:
            rank_str = str(rank)
            assert rank_str in self.V_matrices and rank_str in self.U_matrices, (
                f"rank_str {rank_str} not in V_matrices or U_matrices"
            )

            V = self.V_matrices[rank_str]  # (local_count, rank, d_model)
            U = self.U_matrices[rank_str]  # (local_count, d_model, rank)

            # Get local count from distributed tensor
            V_local = V.to_local()
            U_local = U.to_local()
            local_count = V_local.shape[0]

            assert local_count > 0, f"local_count is 0 for rank {rank}"

            # Extract features from LOCAL shard (not global indexing!)
            curr_features_local = feature_acts_local[..., local_feature_idx : local_feature_idx + local_count]
            local_feature_idx += local_count

            # Compute: f_i * (U_i @ V_i @ x)
            # Step 1: V @ x (project input through V matrices)
            Vx = torch.einsum("... d, c r d -> ... c r", x_local, V_local)

            # Step 2: Weight by feature activations
            weighted_Vx = curr_features_local.unsqueeze(-1) * Vx

            # Step 3: U @ (weighted V @ x)
            reconstruction_local += torch.einsum("... c r, c d r -> ... d", weighted_Vx, U_local)

        # Add decoder bias prior to creating DTensor to avoid distributed broadcasting issues
        if self.cfg.use_decoder_bias:
            bias_local = self.b_D.to_local() if isinstance(self.b_D, DTensor) else self.b_D
            while bias_local.ndim < reconstruction_local.ndim:
                bias_local = bias_local.unsqueeze(0)  # align dimensions for broadcasting
            reconstruction_local = reconstruction_local + bias_local

        # All-reduce(sum) within model parallel shards
        model_group = mesh.get_group("model")
        torch.distributed.all_reduce(reconstruction_local, op=torch.distributed.ReduceOp.SUM, group=model_group)

        # Convert to DTensor with proper sharding (sharding along data dimension, replicated along model dimension)
        reconstruction_dt = DTensor.from_local(
            reconstruction_local,
            device_mesh=mesh,
            placements=DimMap({"data": 0}).placements(mesh),  # = [Shard(0), Replicate()]
        )

        return reconstruction_dt

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

        return reconstruction

    @override
    @torch.no_grad()
    @timer.time("init_parameters")
    def init_parameters(self, **kwargs) -> None:
        super().init_parameters(**kwargs)
        # Initialize encoder
        encoder_bound = 1.0 / math.sqrt(self.cfg.d_sae)

        if self.device_mesh is None:
            # Non-distributed initialization
            W_E = torch.empty(self.cfg.d_model, self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype).uniform_(
                -encoder_bound, encoder_bound
            )
            b_E = torch.zeros(self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype)
        else:
            # Distributed initialization
            W_E_local = torch.empty(
                self.cfg.d_model, self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype
            ).uniform_(-encoder_bound, encoder_bound)
            W_E = self.dim_maps()["W_E"].distribute(W_E_local, self.device_mesh)
            b_E_zeros = torch.zeros(self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype)
            b_E = self.dim_maps()["b_E"].distribute(b_E_zeros, self.device_mesh)

        self.W_E.copy_(W_E)
        self.b_E.copy_(b_E)

        # Initialize U and V matrices for each rank group
        for rank_str in self.U_matrices.keys():
            U = self.U_matrices[rank_str]
            V = self.V_matrices[rank_str]

            # Xavier initialization considering the rank
            U_bound = 1.0 / math.sqrt(self.cfg.d_sae)  # TODO: initialization should have a better solution
            V_bound = 1.0 / math.sqrt(self.cfg.d_sae)

            if self.device_mesh is None:
                # Non-distributed initialization
                U_local = torch.empty(U.shape, device=self.cfg.device, dtype=self.cfg.dtype).uniform_(-U_bound, U_bound)
                V_local = torch.empty(V.shape, device=self.cfg.device, dtype=self.cfg.dtype).uniform_(-V_bound, V_bound)
                U.copy_(U_local)
                V.copy_(V_local)
            else:
                # Distributed initialization - DTensor already has correct global shape
                # Initialize directly on the DTensor's local shard for proper distribution
                U_local_shard = U.to_local()
                V_local_shard = V.to_local()

                # Initialize local shards with proper uniform distribution
                U_local_shard.uniform_(-U_bound, U_bound)
                V_local_shard.uniform_(-V_bound, V_bound)

                # No need to copy - we initialized in place on the DTensor

        if self.cfg.use_decoder_bias:
            if self.device_mesh is None:
                # Non-distributed initialization
                b_D = torch.zeros(self.cfg.d_model, device=self.cfg.device, dtype=self.cfg.dtype)
            else:
                # Distributed initialization
                b_D = self.dim_maps()["b_D"].distribute(
                    torch.zeros(self.cfg.d_model, device=self.cfg.device, dtype=self.cfg.dtype), self.device_mesh
                )
            self.b_D.copy_(b_D)

    @override
    @timer.time("prepare_input")
    def prepare_input(
        self, batch: dict[str, torch.Tensor], **kwargs
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        x = batch[self.cfg.hook_point_in]
        encoder_kwargs = {}
        decoder_kwargs = {"original_x": x}  # Pass original input to decoder for MoLT
        return x, encoder_kwargs, decoder_kwargs

    @override
    @timer.time("prepare_label")
    def prepare_label(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        return batch[self.cfg.hook_point_out]

    @override
    @torch.no_grad()
    def compute_training_metrics(
        self,
        *,
        l0: torch.Tensor,
        feature_acts: torch.Tensor,
        **kwargs,
    ) -> dict[str, float]:
        """Compute per-rank group training metrics for MOLT."""
        metrics = {}
        feature_idx = 0
        total_rank_sum = 0.0

        for rank in self.cfg.available_ranks:
            rank_str = str(rank)
            if rank_str in self.U_matrices:
                # Extract features for this rank group
                end_idx = (
                    feature_idx + self.cfg.rank_counts[rank]
                )  # rank_counts[rank] is the GLOBAL count of this rank group
                rank_features = feature_acts[..., feature_idx:end_idx]

                # Count active transforms (l0) for this rank group
                rank_l0 = (rank_features > 0).float().sum(-1)
                rank_l0_mean = item(rank_l0.mean())

                # Record metrics
                metrics[f"molt_metrics/l0_rank{rank}"] = rank_l0_mean
                metrics[f"molt_metrics/l0_rank{rank}_ratio"] = rank_l0_mean / self.cfg.rank_counts[rank]
                total_rank_sum += rank_l0_mean * rank

                feature_idx += self.cfg.rank_counts[rank]

        # Record total rank sum
        metrics["molt_metrics/total_rank_sum"] = total_rank_sum
        return metrics

    @override
    @timer.time("forward")
    def forward(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        encoder_kwargs: dict[str, Any] = {},
        decoder_kwargs: dict[str, Any] = {},
    ) -> Union[
        Float[torch.Tensor, "batch d_model"],
        Float[torch.Tensor, "batch seq_len d_model"],
    ]:
        """Forward pass through the autoencoder.
        Ensure that the input activations are normalized by calling `normalize_activations` before calling this method.
        """
        feature_acts = self.encode(x, **encoder_kwargs)
        reconstructed = self.decode(feature_acts, original_x=x)
        return reconstructed
