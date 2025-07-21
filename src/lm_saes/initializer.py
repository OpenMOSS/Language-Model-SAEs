from typing import Dict, Iterable, List

import torch
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from wandb.sdk.wandb_run import Run
import scipy.stats

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.config import BaseSAEConfig, InitializerConfig
from lm_saes.crosscoder import CrossCoder
from lm_saes.sae import SparseAutoEncoder
from lm_saes.utils.logging import get_distributed_logger
from lm_saes.utils.misc import calculate_activation_norm
from lm_saes.utils.tensor_dict import batch_size

logger = get_distributed_logger("initializer")


class Initializer:
    def __init__(self, cfg: InitializerConfig):
        self.cfg = cfg

    @torch.no_grad()
    def initialize_parameters(self, sae: AbstractSparseAutoEncoder):
        """Initialize the parameters of the SAE.
        Only used when the state is "training" to initialize sae.
        """

        sae.init_parameters(
            encoder_uniform_bound=self.cfg.encoder_uniform_bound,
            decoder_uniform_bound=self.cfg.decoder_uniform_bound,
            init_log_jumprelu_threshold_value=self.cfg.init_log_jumprelu_threshold_value,
        )

        if self.cfg.init_decoder_norm:
            sae.set_decoder_to_fixed_norm(self.cfg.init_decoder_norm, force_exact=True)

        if self.cfg.init_encoder_with_decoder_transpose:
            sae.init_encoder_with_decoder_transpose(self.cfg.init_encoder_with_decoder_transpose_factor)

        if self.cfg.init_encoder_norm:
            sae.set_encoder_to_fixed_norm(self.cfg.init_encoder_norm)

        return sae

    @torch.no_grad()
    def initialization_search(
        self,
        sae: AbstractSparseAutoEncoder,
        activation_batch: Dict[str, Tensor],
        wandb_logger: Run | None = None,
    ):
        """
        This function is used to search for the best initialization norm for the SAE decoder.
        """
        batch = sae.normalize_activations(activation_batch)

        if self.cfg.init_decoder_norm is None:
            def grid_search_best_init_norm(search_range: List[float]) -> float:
                losses: Dict[float, float] = {}

                for norm in search_range:
                    sae.set_decoder_to_fixed_norm(norm, force_exact=True)
                    if self.cfg.init_encoder_with_decoder_transpose and self.cfg.init_encoder_norm is None:
                        sae.init_encoder_with_decoder_transpose(self.cfg.init_encoder_with_decoder_transpose_factor)
                    mse = sae.compute_loss(batch)[1][0]["l_rec"].mean().item()
                    losses[norm] = mse
                best_norm = min(losses, key=losses.get)  # type: ignore
                return best_norm

            best_norm_coarse = grid_search_best_init_norm(torch.linspace(0.1, 1, 10).numpy().tolist())  # type: ignore
            best_norm_fine_grained = grid_search_best_init_norm(
                torch.linspace(best_norm_coarse - 0.09, best_norm_coarse + 0.1, 20).numpy().tolist()  # type: ignore
            )

            logger.info(f"The best (i.e. lowest MSE) initialized norm is {best_norm_fine_grained}")
            if wandb_logger is not None:
                wandb_logger.log({"best_norm_fine_grained": best_norm_fine_grained})

            sae.set_decoder_to_fixed_norm(best_norm_fine_grained, force_exact=True)

        if self.cfg.bias_init_method == "geometric_median":
            assert isinstance(sae, SparseAutoEncoder), (
                "SparseAutoEncoder is the only supported SAE type for encoder bias initialization"
            )
            assert sae.b_D is not None, "Decoder bias should exist if use_decoder_bias is True"
            sae.b_D.copy_(
                sae.compute_norm_factor(batch[sae.cfg.hook_point_out], hook_point=sae.cfg.hook_point_out)
                * batch[sae.cfg.hook_point_out]
            ).mean(0)

            if not sae.cfg.apply_decoder_bias_to_pre_encoder:
                normalized_input = (
                    sae.compute_norm_factor(batch[sae.cfg.hook_point_in], hook_point=sae.cfg.hook_point_in)
                    * batch[sae.cfg.hook_point_in]
                )
                normalized_median = normalized_input.mean(0)
                sae.b_E.copy_(-normalized_median @ sae.W_E)

        if self.cfg.init_encoder_with_decoder_transpose and self.cfg.init_encoder_norm is None:
            sae.init_encoder_with_decoder_transpose(self.cfg.init_encoder_with_decoder_transpose_factor)

        return sae

    @torch.no_grad()
    def initialize_jump_relu_threshold(self, sae: AbstractSparseAutoEncoder, activation_batch: Dict[str, Tensor]):
        """
        This function is used to initialize the jump_relu_threshold for the SAE.
        """
        batch = sae.normalize_activations(activation_batch)
        x, kwargs = sae.prepare_input(batch)
        _, hidden_pre = sae.encode(x, **kwargs, return_hidden_pre=True)
        hidden_pre = torch.clamp(hidden_pre, min=0.0)
        hidden_pre = hidden_pre.flatten()
        threshold = hidden_pre.topk(k=batch_size(batch) * sae.cfg.top_k).values[-1]
        sae.cfg.jump_relu_threshold = threshold.item()
        return sae

    def initialize_sae_from_config(
        self,
        cfg: BaseSAEConfig,
        activation_stream: Iterable[dict[str, Tensor]] | None = None,
        activation_norm: dict[str, float] | None = None,
        device_mesh: DeviceMesh | None = None,
        wandb_logger: Run | None = None,
    ):
        """
        Initialize the SAE from the SAE config.
        Args:
            cfg (SAEConfig): The SAE config.
            activation_iter (Iterable[dict[str, Tensor]] | None): The activation iterator. Used for initialization search when self.cfg.init_search is True.
            activation_norm (dict[str, float] | None): The activation normalization. Used for dataset-wise normalization when self.cfg.norm_activation is "dataset-wise".
            device_mesh (DeviceMesh | None): The device mesh.
        """
        if cfg.sae_type == "sae":
            sae: AbstractSparseAutoEncoder = SparseAutoEncoder.from_config(cfg, device_mesh=device_mesh)
        elif cfg.sae_type == "crosscoder":
            sae: AbstractSparseAutoEncoder = CrossCoder.from_config(cfg, device_mesh=device_mesh)
        else:
            raise ValueError(f"SAE type {cfg.sae_type} not supported.")
        if self.cfg.state == "training":
            if cfg.sae_type == "sae" and cfg.proj_data:
                assert activation_stream is not None, (
                    "Activation iterator must be provided for proj_data initialization"
                )
                self.initialize_proj_data(sae, activation_stream, device_mesh)
                
            if cfg.sae_pretrained_name_or_path is None:
                sae = self.initialize_parameters(sae)
            if sae.cfg.norm_activation == "dataset-wise":
                if activation_norm is None:
                    assert activation_stream is not None, (
                        "Activation iterator must be provided for dataset-wise normalization"
                    )

                    activation_norm = calculate_activation_norm(
                        activation_stream, cfg.associated_hook_points, device_mesh=device_mesh
                    )
                sae.set_dataset_average_activation_norm(activation_norm)

            if self.cfg.init_search:
                assert activation_stream is not None, "Activation iterator must be provided for initialization search"
                activation_batch = next(iter(activation_stream))  # type: ignore
                sae = self.initialization_search(sae, activation_batch, wandb_logger=wandb_logger)

        elif self.cfg.state == "inference":
            if sae.cfg.norm_activation == "dataset-wise":
                sae.standardize_parameters_of_dataset_norm(activation_norm)
            if sae.cfg.sparsity_include_decoder_norm:
                sae.transform_to_unit_decoder_norm()
            if "topk" in sae.cfg.act_fn:
                logger.info(
                    "Converting topk activation to jumprelu for inference. Features are set independent to each other."
                )
                if sae.cfg.jump_relu_threshold is None:
                    assert activation_stream is not None, (
                        "Activation iterator must be provided for jump_relu_threshold initialization"
                    )
                    activation_batch = next(iter(activation_stream))
                    self.initialize_jump_relu_threshold(sae, activation_batch)
                sae.cfg.act_fn = "jumprelu"
        return sae

    @torch.no_grad()
    def initialize_proj_data(self, sae: AbstractSparseAutoEncoder, activation_stream: Iterable[dict[str, Tensor]], device_mesh: DeviceMesh | None):
        """
        Initialize the proj data for the SAE.
        Collects 200,000 samples from activation_stream to initialize proj.bias with mean
        and proj.weight using SVD or random orthogonal matrix.
        
        In distributed training, all devices iterate through the same activation_stream to maintain
        synchronization, but only the master rank (rank 0) performs the computation intensive operations
        (SVD, random matrix generation), then broadcasts the results to all other ranks to ensure 
        consistency and avoid distributed tensor creation deadlocks.
        
        Args:
            sae: The SAE model to initialize
            activation_stream: Iterator over activation batches
            device_mesh: Device mesh for distributed training (None for single GPU)
        """
        logger.info("Initializing proj data...")
        
        if device_mesh is not None:
            import torch.distributed as dist
            is_master = dist.get_rank() == 0
            logger.info(f"Distributed training mode: All ranks will iterate data, master rank will compute and broadcast proj parameters")
        else:
            is_master = True
        
        # All devices collect the same data to keep activation_stream synchronized
        target_samples = 100_000
        collected_samples = []
        total_collected = 0
        
        for batch in activation_stream:
            # Use hook_point_in as the data source
            activations = batch[sae.cfg.hook_point_in]  # [batch_size, seq_len, d_model] or [batch_size, d_model]
            
            # In distributed case, ensure we're working with local tensor for computation
            if hasattr(activations, 'to_local'):
                activations = activations.to_local()
            
            # Flatten if needed to get individual samples
            if activations.dim() == 3:
                activations = activations.reshape(-1, activations.size(-1))  # [batch_size * seq_len, d_model]
            
            batch_samples = activations.size(0)
            if total_collected + batch_samples > target_samples:
                # Take only what we need
                needed = target_samples - total_collected
                activations = activations[:needed]
            
            collected_samples.append(activations)
            total_collected += activations.size(0)
            
            if total_collected >= target_samples:
                break
        
        # All devices concatenate the data (to keep memory usage and computation synchronized)
        all_activations = torch.cat(collected_samples, dim=0)  # [100_000, d_model]
        
        # Only master rank performs the actual computation
        if is_master:
            # Initialize proj.bias with the mean of the data
            proj_bias = all_activations.mean(dim=0)  # [d_model]
            
            # Center the data
            centered_data = all_activations - proj_bias  # [100_000, d_model]
            
            # Initialize proj.weight
            if sae.cfg.init_with_svd:
                logger.info("Initializing proj.weight with SVD")
                # Perform SVD decomposition
                U, S, V = torch.svd(centered_data.T.to(torch.float32))  # centered_data.T is [d_model, 100_000]
                # Use the first d_feature principal components
                proj_weight = U[:, :sae.cfg.d_feature].contiguous() # [d_model, d_feature]
                variance_factor = (S[:sae.cfg.d_feature] ** 2).sum() / (S ** 2).sum()
                relative_sigular_value = S / S[0]
                sae.variance_factor.copy_(variance_factor)
                logger.info(f"Variance factor: {variance_factor}")
                del U, S, V, centered_data
                torch.cuda.empty_cache()
            else:
                logger.info("Initializing proj.weight with random orthogonal matrix")
                # Use scipy.stats.ortho_group to get a random orthogonal matrix
                ortho_matrix = scipy.stats.ortho_group.rvs(sae.cfg.d_model)  # [d_model, d_model]
                ortho_matrix = torch.from_numpy(ortho_matrix).to(dtype=all_activations.dtype, device=all_activations.device)
                # Take the first d_feature columns
                proj_weight = ortho_matrix[:, :sae.cfg.d_feature].contiguous()  # [d_model, d_feature]
                del ortho_matrix
                torch.cuda.empty_cache()
        else:
            # Non-master ranks create empty tensors with correct shape and dtype
            device = all_activations.device
            dtype = all_activations.dtype
            proj_bias = torch.empty(sae.cfg.d_model, device=device, dtype=dtype)
            proj_weight = torch.empty(sae.cfg.d_model, sae.cfg.d_feature, device=device, dtype=dtype)
            relative_sigular_value = torch.empty(sae.cfg.d_model, device=device, dtype=dtype)
        # Broadcast from master rank to all other ranks
        if device_mesh is not None:
            # Ensure tensors are on CUDA and contiguous
            proj_bias = proj_bias.cuda().contiguous()
            proj_weight = proj_weight.cuda().contiguous()
            dist.broadcast(proj_bias, src=0)
            dist.broadcast(proj_weight, src=0)
            if sae.cfg.init_with_svd:
                relative_sigular_value = relative_sigular_value.cuda().contiguous()
                dist.broadcast(relative_sigular_value, src=0)
        # Now all ranks have identical tensors, safe to distribute with replicated placement
        if device_mesh is not None:
            logger.info("Converting proj parameters to distributed tensors with replicated placement")
            proj_bias = sae.dim_maps()["proj_bias"].distribute(proj_bias, device_mesh)
            proj_weight = sae.dim_maps()["proj_weight"].distribute(proj_weight, device_mesh)
            if sae.cfg.init_with_svd:
                relative_sigular_value = sae.dim_maps()["relative_sigular_value"].distribute(relative_sigular_value, device_mesh)
        # Set the proj parameters
        sae.proj_bias.copy_(proj_bias)
        sae.proj_weight.copy_(proj_weight)
        if sae.cfg.init_with_svd:
            sae.relative_sigular_value.copy_(relative_sigular_value)
        logger.info("Proj data initialization completed")
        del proj_bias, proj_weight, all_activations
        torch.cuda.empty_cache()
        
