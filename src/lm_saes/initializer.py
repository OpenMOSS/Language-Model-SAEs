from typing import Dict, Iterable, List, Literal, cast

import torch
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from transformer_lens import HookedTransformer
from transformer_lens.components import Attention, GroupedQueryAttention, TransformerBlock
from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
from wandb.sdk.wandb_run import Run

from lm_saes.abstract_sae import AbstractSparseAutoEncoder, BaseSAEConfig
from lm_saes.backend.language_model import LanguageModel, TransformerLensLanguageModel
from lm_saes.clt import CrossLayerTranscoder
from lm_saes.config import BaseConfig
from lm_saes.lorsa import LowRankSparseAttention
from lm_saes.molt import MixtureOfLinearTransform
from lm_saes.sae import SparseAutoEncoder
from lm_saes.utils.distributed.ops import item
from lm_saes.utils.logging import get_distributed_logger
from lm_saes.utils.misc import calculate_activation_norm

logger = get_distributed_logger("initializer")


class InitializerConfig(BaseConfig):
    bias_init_method: Literal["all_zero", "geometric_median"] = "all_zero"
    decoder_uniform_bound: float = 1.0
    encoder_uniform_bound: float = 1.0
    init_encoder_with_decoder_transpose: bool = True
    init_encoder_with_decoder_transpose_factor: float = 1.0
    init_log_jumprelu_threshold_value: float | None = None
    grid_search_init_norm: bool = False
    initialize_W_D_with_active_subspace: bool = False
    d_active_subspace: int | None = None
    initialize_lorsa_with_mhsa: bool | None = None
    initialize_tc_with_mlp: bool | None = None
    model_layer: int | None = None
    init_encoder_bias_with_mean_hidden_pre: bool = False


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

        if self.cfg.bias_init_method == "geometric_median":
            assert sae.b_D is not None, "Decoder bias should exist if use_decoder_bias is True"
            if isinstance(sae, CrossLayerTranscoder):
                for i in range(sae.cfg.n_layers):
                    hook_point_out = sae.cfg.hook_points_out[i]
                    normalized_mean_activation = batch[hook_point_out].mean(0)
                    sae.b_D[i].copy_(normalized_mean_activation)
            elif (
                isinstance(sae, MixtureOfLinearTransform)
                or isinstance(sae, LowRankSparseAttention)
                or isinstance(sae, SparseAutoEncoder)
            ):
                label = sae.prepare_label(batch)
                normalized_mean_activation = label.mean(dim=list(range((batch[sae.cfg.hook_point_out].ndim - 1))))
                sae.b_D.copy_(normalized_mean_activation)
            else:
                raise ValueError(
                    f"Bias initialization method {self.cfg.bias_init_method} is not supported for {sae.cfg.sae_type}"
                )

        if self.cfg.init_encoder_bias_with_mean_hidden_pre:
            sae.init_encoder_bias_with_mean_hidden_pre(batch)

        @torch.autocast(device_type=sae.cfg.device, dtype=sae.cfg.dtype)
        def grid_search_best_init_norm(search_range: List[float]) -> float:
            losses: Dict[float, float] = {}

            for norm in search_range:
                sae.set_decoder_to_fixed_norm(norm, force_exact=True)
                if self.cfg.init_encoder_with_decoder_transpose:
                    sae.init_encoder_with_decoder_transpose(self.cfg.init_encoder_with_decoder_transpose_factor)
                if self.cfg.init_encoder_bias_with_mean_hidden_pre:
                    sae.init_encoder_bias_with_mean_hidden_pre(batch)
                mse = item(sae.compute_loss(batch)["l_rec"].mean())
                losses[norm] = mse
            best_norm = min(losses, key=losses.get)  # type: ignore
            return best_norm

        if self.cfg.grid_search_init_norm:
            best_norm_coarse = grid_search_best_init_norm(torch.linspace(0.1, 1, 10).numpy().tolist())
            best_norm_fine_grained = grid_search_best_init_norm(
                torch.linspace(best_norm_coarse - 0.09, best_norm_coarse + 0.1, 20).numpy().tolist()
            )

            logger.info(f"The best (i.e. lowest MSE) initialized norm is {best_norm_fine_grained}")
            if wandb_logger is not None:
                wandb_logger.log({"best_norm_fine_grained": best_norm_fine_grained})

            sae.set_decoder_to_fixed_norm(best_norm_fine_grained, force_exact=True)

        if self.cfg.init_encoder_with_decoder_transpose:
            sae.init_encoder_with_decoder_transpose(self.cfg.init_encoder_with_decoder_transpose_factor)
        if self.cfg.init_encoder_bias_with_mean_hidden_pre:
            sae.init_encoder_bias_with_mean_hidden_pre(batch)

        return sae

    def initialize_sae_from_config(
        self,
        cfg: BaseSAEConfig,
        activation_stream: Iterable[dict[str, Tensor]] | None = None,
        activation_norm: dict[str, float] | None = None,
        device_mesh: DeviceMesh | None = None,
        wandb_logger: Run | None = None,
        model: LanguageModel | None = None,
    ):
        """
        Initialize the SAE from the SAE config.
        Args:
            cfg (SAEConfig): The SAE config.
            activation_iter (Iterable[dict[str, Tensor]] | None): The activation iterator.
            activation_norm (dict[str, float] | None): The activation normalization. Used for dataset-wise normalization when self.cfg.norm_activation is "dataset-wise".
            device_mesh (DeviceMesh | None): The device mesh.
        """
        sae: AbstractSparseAutoEncoder = AbstractSparseAutoEncoder.from_config(
            cfg,
            device_mesh=device_mesh,
        )

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

        if isinstance(sae, LowRankSparseAttention) and self.cfg.initialize_lorsa_with_mhsa:
            assert sae.cfg.norm_activation == "dataset-wise", (
                "Norm activation must be dataset-wise for Lorsa if use initialize_lorsa_with_mhsa"
            )
            assert isinstance(model, TransformerLensLanguageModel) and model.model is not None, (
                "Only support TransformerLens backend for initializing Lorsa with Original Multi Head Sparse Attention"
            )
            assert self.cfg.model_layer is not None, (
                "Model layer must be provided for initializing Lorsa with Original Multi Head Sparse Attention"
            )
            assert isinstance(model.model, HookedTransformer), "Model must be a TransformerLens model"
            assert isinstance(model.model.blocks[self.cfg.model_layer], TransformerBlock), (
                "Block must be a TransformerBlock"
            )
            assert isinstance(model.model.blocks[self.cfg.model_layer].attn, Attention | GroupedQueryAttention), (
                "Attention must be an Attention or GroupedQueryAttention"
            )
            sae.init_lorsa_with_mhsa(
                cast(
                    Attention | GroupedQueryAttention,
                    model.model.blocks[self.cfg.model_layer].attn,
                )
            )

        assert activation_stream is not None, "Activation iterator must be provided for initialization search"
        activation_batch = next(iter(activation_stream))  # type: ignore

        if (
            isinstance(sae, SparseAutoEncoder)
            and sae.cfg.hook_point_in != sae.cfg.hook_point_out
            and self.cfg.initialize_tc_with_mlp
        ):
            batch = sae.normalize_activations(activation_batch)
            assert sae.cfg.norm_activation == "dataset-wise"
            assert isinstance(model, TransformerLensLanguageModel) and model.model is not None
            assert self.cfg.model_layer is not None
            assert isinstance(model.model, HookedTransformer), "Model must be a TransformerLens model"
            assert isinstance(model.model.blocks[self.cfg.model_layer], TransformerBlock), (
                "Block must be a TransformerBlock"
            )
            assert isinstance(model.model.blocks[self.cfg.model_layer].mlp, CanBeUsedAsMLP)
            sae.init_tc_with_mlp(
                batch=batch,
                mlp=cast(CanBeUsedAsMLP, model.model.blocks[self.cfg.model_layer].mlp),
            )

        if self.cfg.initialize_W_D_with_active_subspace:
            batch = sae.normalize_activations(activation_batch)
            if isinstance(sae, LowRankSparseAttention):
                assert sae.cfg.norm_activation == "dataset-wise", (
                    "Norm activation must be dataset-wise for Lorsa if use initialize_W_D_with_active_subspace"
                )
                assert isinstance(model, TransformerLensLanguageModel) and model.model is not None, (
                    "Only support TransformerLens backend for initializing Lorsa decoder weight with active subspace"
                )
                assert self.cfg.model_layer is not None, (
                    "Model layer must be provided for initializing Lorsa decoder weight with active subspace"
                )
                sae.init_W_V_with_active_subspace_per_head(
                    batch=batch,
                    mhsa=cast(
                        Attention | GroupedQueryAttention,
                        model.model.blocks[self.cfg.model_layer].attn,
                    ),
                )
            else:
                assert self.cfg.d_active_subspace is not None, (
                    "d_active_subspace must be provided for initializing other SAEs with active subspace"
                )
                sae.init_W_D_with_active_subspace(batch=batch, d_active_subspace=self.cfg.d_active_subspace)

        sae = self.initialization_search(sae, activation_batch, wandb_logger=wandb_logger)

        return sae
