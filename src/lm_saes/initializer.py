import math
from typing import Dict, Iterable, List

import torch
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from wandb.sdk.wandb_run import Run

from lm_saes.abstract_sae import AbstractSparseAutoEncoder, JumpReLU
from lm_saes.clt import CrossLayerTranscoder
from lm_saes.lorsa import LowRankSparseAttention
from lm_saes.config import BaseSAEConfig, InitializerConfig
from lm_saes.crosscoder import CrossCoder
from lm_saes.sae import SparseAutoEncoder
from lm_saes.utils.logging import get_distributed_logger
from lm_saes.utils.misc import calculate_activation_norm
from lm_saes.utils.tensor_dict import batch_size
from lm_saes.backend.language_model import LanguageModel

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

        if self.cfg.init_encoder_with_decoder_transpose:
            sae.init_encoder_with_decoder_transpose(self.cfg.init_encoder_with_decoder_transpose_factor)

        return sae

    # @torch.no_grad()
    # def initialization_search(
    #     self,
    #     sae: AbstractSparseAutoEncoder,
    #     activation_batch: Dict[str, Tensor],
    #     wandb_logger: Run | None = None,
    # ):
    #     """
    #     This function is used to search for the best initialization norm for the SAE decoder.
    #     """
    #     batch = sae.normalize_activations(activation_batch)

    #     def grid_search_best_init_norm(search_range: List[float]) -> float:
    #         losses: Dict[float, float] = {}

    #         for norm in search_range:
    #             sae.set_decoder_to_fixed_norm(norm, force_exact=True)
    #             if self.cfg.init_encoder_with_decoder_transpose:
    #                 sae.init_encoder_with_decoder_transpose(self.cfg.init_encoder_with_decoder_transpose_factor)
    #             mse = sae.compute_loss(batch)[1][0]["l_rec"].mean().item()
    #             losses[norm] = mse
    #         best_norm = min(losses, key=losses.get)  # type: ignore
    #         return best_norm

    #     best_norm_coarse = grid_search_best_init_norm(torch.linspace(0.1, 1, 10).numpy().tolist())  # type: ignore
    #     best_norm_fine_grained = grid_search_best_init_norm(
    #         torch.linspace(best_norm_coarse - 0.09, best_norm_coarse + 0.1, 20).numpy().tolist()  # type: ignore
    #     )

    #     logger.info(f"The best (i.e. lowest MSE) initialized norm is {best_norm_fine_grained}")
    #     if wandb_logger is not None:
    #         wandb_logger.log({"best_norm_fine_grained": best_norm_fine_grained})

    #     sae.set_decoder_to_fixed_norm(best_norm_fine_grained, force_exact=True)

    #     if self.cfg.bias_init_method == "geometric_median":
    #         assert isinstance(sae, SparseAutoEncoder), (
    #             "SparseAutoEncoder is the only supported SAE type for encoder bias initialization"
    #         )
    #         assert sae.b_D is not None, "Decoder bias should exist if use_decoder_bias is True"
    #         # sae.b_D.copy_(
    #         #     sae.compute_norm_factor(batch[sae.cfg.hook_point_out], hook_point=sae.cfg.hook_point_out)
    #         #     * batch[sae.cfg.hook_point_out]
    #         # ).mean(0)
    #         sae.b_D.copy_(
    #             (
    #                 sae.compute_norm_factor(
    #                     batch[sae.cfg.hook_point_out], 
    #                     hook_point=sae.cfg.hook_point_out
    #                 )
    #                 * batch[sae.cfg.hook_point_out]
    #             ).mean(0)
    #         )

    #         normalized_input = (
    #             sae.compute_norm_factor(
    #                 batch[sae.cfg.hook_point_in], 
    #                 hook_point=sae.cfg.hook_point_in
    #             )
    #             * batch[sae.cfg.hook_point_in]
    #         )
    #         normalized_median = normalized_input.mean(0)
    #         sae.b_E.copy_(-normalized_median @ sae.W_E)

    #     if self.cfg.init_encoder_with_decoder_transpose:
    #         sae.init_encoder_with_decoder_transpose(self.cfg.init_encoder_with_decoder_transpose_factor)

    #     return sae

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

        @torch.autocast(device_type=sae.cfg.device, dtype=sae.cfg.dtype)
        def grid_search_best_init_norm(search_range: List[float]) -> float:
            losses: Dict[float, float] = {}

            for norm in search_range:
                sae.set_decoder_to_fixed_norm(norm, force_exact=True)
                if self.cfg.init_encoder_with_decoder_transpose:
                    sae.init_encoder_with_decoder_transpose(self.cfg.init_encoder_with_decoder_transpose_factor)
                mse = sae.compute_loss(batch)[1][0]["l_rec"].mean().item()  # type: ignore
                losses[norm] = mse
            best_norm = min(losses, key=losses.get)  # type: ignore
            return best_norm

        if self.cfg.grid_search_init_norm:
            best_norm_coarse = grid_search_best_init_norm(torch.linspace(0.1, 1, 10).numpy().tolist())  # type: ignore
            best_norm_fine_grained = grid_search_best_init_norm(
                torch.linspace(best_norm_coarse - 0.09, best_norm_coarse + 0.1, 20).numpy().tolist()  # type: ignore
            )

            logger.info(f"The best (i.e. lowest MSE) initialized norm is {best_norm_fine_grained}")
            if wandb_logger is not None:
                wandb_logger.log({"best_norm_fine_grained": best_norm_fine_grained})

            sae.set_decoder_to_fixed_norm(best_norm_fine_grained, force_exact=True)

        if self.cfg.bias_init_method == "geometric_median":
            assert sae.b_D is not None, "Decoder bias should exist if use_decoder_bias is True"
            if isinstance(sae, CrossLayerTranscoder):
                for i in range(sae.cfg.n_layers):
                    hook_point_out = sae.cfg.hook_points_out[i]
                    normalized_mean_activation = batch[hook_point_out].mean(0)
                    if isinstance(sae.b_D[i], DTensor):
                        normalized_mean_activation = DTensor.from_local(
                            normalized_mean_activation,
                            device_mesh=sae.device_mesh,
                            placements=sae.dim_maps()["b_D"].placements(sae.device_mesh),
                        )
                    
                    sae.b_D[i].copy_(normalized_mean_activation)
            else:
                normalized_mean_activation = batch[sae.cfg.hook_point_out].mean(
                    dim=list(range((batch[sae.cfg.hook_point_out].ndim - 1)))
                )
                sae.b_D.copy_(normalized_mean_activation)

        if self.cfg.init_encoder_with_decoder_transpose:
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
        sae.cfg.act_fn = "jumprelu"
        sae.activation_function = sae.activation_function_factory(sae.device_mesh)
        assert isinstance(sae.activation_function, JumpReLU)
        sae.activation_function.log_jumprelu_threshold.data.fill_(math.log(threshold.item()))
        return sae

    # def initialize_sae_from_config(
    #     self,
    #     cfg: BaseSAEConfig,
    #     activation_stream: Iterable[dict[str, Tensor]] | None = None,
    #     activation_norm: dict[str, float] | None = None,
    #     device_mesh: DeviceMesh | None = None,
    #     wandb_logger: Run | None = None,
    # ):
    #     """
    #     Initialize the SAE from the SAE config.
    #     Args:
    #         cfg (SAEConfig): The SAE config.
    #         activation_iter (Iterable[dict[str, Tensor]] | None): The activation iterator. Used for initialization search when self.cfg.init_search is True.
    #         activation_norm (dict[str, float] | None): The activation normalization. Used for dataset-wise normalization when self.cfg.norm_activation is "dataset-wise".
    #         device_mesh (DeviceMesh | None): The device mesh.
    #     """
    #     if cfg.sae_type == "sae":
    #         sae: AbstractSparseAutoEncoder = SparseAutoEncoder.from_config(cfg, device_mesh=device_mesh)
    #     elif cfg.sae_type == "crosscoder":
    #         sae: AbstractSparseAutoEncoder = CrossCoder.from_config(cfg, device_mesh=device_mesh)
    #     elif cfg.sae_type == "clt":
    #         sae: AbstractSparseAutoEncoder = CrossLayerTranscoder.from_config(cfg, device_mesh=device_mesh)
    #     else:
    #         raise ValueError(f"SAE type {cfg.sae_type} not supported.")
    #     if self.cfg.state == "training":
    #         if cfg.sae_pretrained_name_or_path is None:
    #             sae = self.initialize_parameters(sae)
    #         if sae.cfg.norm_activation == "dataset-wise":
    #             if activation_norm is None:
    #                 assert activation_stream is not None, (
    #                     "Activation iterator must be provided for dataset-wise normalization"
    #                 )

    #                 activation_norm = calculate_activation_norm(
    #                     activation_stream, cfg.associated_hook_points, device_mesh=device_mesh
    #                 )
    #             sae.set_dataset_average_activation_norm(activation_norm)

    #         if self.cfg.init_search:
    #             assert activation_stream is not None, "Activation iterator must be provided for initialization search"
    #             activation_batch = next(iter(activation_stream))  # type: ignore
    #             sae = self.initialization_search(sae, activation_batch, wandb_logger=wandb_logger)

    #     elif self.cfg.state == "inference":
    #         if sae.cfg.norm_activation == "dataset-wise":
    #             sae.standardize_parameters_of_dataset_norm(activation_norm)
    #         if sae.cfg.sparsity_include_decoder_norm:
    #             sae.transform_to_unit_decoder_norm()
    #         if "topk" in sae.cfg.act_fn:
    #             logger.info(
    #                 "Converting topk activation to jumprelu for inference. Features are set independent to each other."
    #             )
    #             assert activation_stream is not None, (
    #                 "Activation iterator must be provided for jump_relu_threshold initialization"
    #             )
    #             activation_batch = next(iter(activation_stream))
    #             self.initialize_jump_relu_threshold(sae, activation_batch)
    #     return sae


    def initialize_sae_from_config(
        self,
        cfg: BaseSAEConfig,
        activation_stream: Iterable[dict[str, Tensor]] | None = None,
        activation_norm: dict[str, float] | None = None,
        device_mesh: DeviceMesh | None = None,
        wandb_logger: Run | None = None,
        fold_activation_scale: bool = False,
        model: LanguageModel | None = None
    ):
        """
        Initialize the SAE from the SAE config.
        Args:
            cfg (SAEConfig): The SAE config.
            activation_iter (Iterable[dict[str, Tensor]] | None): The activation iterator.
            activation_norm (dict[str, float] | None): The activation normalization. Used for dataset-wise normalization when self.cfg.norm_activation is "dataset-wise".
            device_mesh (DeviceMesh | None): The device mesh.
        """
        try:
            sae_cls = {
                "sae": SparseAutoEncoder,
                "crosscoder": CrossCoder,
                "clt": CrossLayerTranscoder,
                "lorsa": LowRankSparseAttention,
            }[cfg.sae_type]
        except KeyError:
            raise ValueError(f"SAE type {cfg.sae_type} not supported.")

        sae: AbstractSparseAutoEncoder = sae_cls.from_config(
            cfg,
            device_mesh=device_mesh,
            fold_activation_scale=fold_activation_scale,
        )
        
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
                    print('train from scratch...... set_dataset_average_activation norm')
                    print(f'{activation_norm = }')
                sae.set_dataset_average_activation_norm(activation_norm)
                print('successfully set dataset_average_activation_norm!')
                # sae.standardize_parameters_of_dataset_norm(activation_norm)
            # if self.cfg.init_search:
            #     assert activation_stream is not None, "Activation iterator must be provided for initialization search"
            #     activation_batch = next(iter(activation_stream))  # type: ignore
            #     sae = self.initialization_search(sae, activation_batch, wandb_logger=wandb_logger)
            if cfg.sae_type == 'lorsa' and self.cfg.initialize_lorsa_with_mhsa and sae.cfg.norm_activation == 'dataset-wise':
                sae.init_lorsa_with_mhsa(model.model.blocks[self.cfg.model_layer])
 
            # Initialize LORSA SmolGen from encoder if configured
            if (
                cfg.sae_type == 'lorsa'
                and cfg.use_smolgen
                and self.cfg.initialize_lorsa_smolgen_from_encoder
                and model is not None
            ):
                print(
                    f"====== init_smolgen_from_encoder ======"
                )
                sae.init_smolgen_from_encoder(
                    model.model.blocks[self.cfg.model_layer]
                )
 
            assert activation_stream is not None, "Activation iterator must be provided for initialization search"
            activation_batch = next(iter(activation_stream))  # type: ignore
            sae = self.initialization_search(sae, activation_batch, wandb_logger=wandb_logger)
 
            if self.cfg.initialize_W_D_with_active_subspace:
                activation_batch = sae.normalize_activations(activation_batch)
                if cfg.sae_type == 'lorsa' and self.cfg.initialize_lorsa_with_mhsa and sae.cfg.norm_activation == 'dataset-wise':
                    print(
                        f"====== init_W_D_with_active_subspace_per_head ======"
                    )
                    sae.init_W_D_with_active_subspace_per_head(
                        activation_batch,
                        encoder_layer=model.model.blocks[self.cfg.model_layer],
                    )
                else:
                    sae.init_W_D_with_active_subspace(activation_batch, self.cfg.d_active_subspace)
            if self.cfg.initialize_lorsa_attn_scale_from_encoder:
                print(
                    f"====== init_attn_scale_from_encoder ======"
                )
                sae.init_attn_scale_from_encoder(
                    encoder_layer=model.model.blocks[self.cfg.model_layer],
                )
            if self.cfg.initialize_W_D_with_mhsa:
                print(
                    f"====== init_lorsa_W_D_with_mhsa ======"
                )
                sae.init_lorsa_W_D_with_mhsa(
                    encoder_layer=model.model.blocks[self.cfg.model_layer],
                )
        return sae
