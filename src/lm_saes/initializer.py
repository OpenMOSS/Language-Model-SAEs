import warnings
from typing import Any, Dict, Iterable, List

import torch
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from lm_saes.config import BaseSAEConfig, InitializerConfig
from lm_saes.crosscoder import CrossCoder
from lm_saes.mixcoder import MixCoder
from lm_saes.sae import SparseAutoEncoder
from lm_saes.utils.misc import all_reduce_tensor, calculate_activation_norm, get_modality_indices


class Initializer:
    def __init__(self, cfg: InitializerConfig):
        self.cfg = cfg

    @torch.no_grad()
    def initialize_parameters(self, sae: SparseAutoEncoder, mixcoder_settings: dict[str, Any] | None = None):
        """Initialize the parameters of the SAE.
        Only used when the state is "training" to initialize sae.
        """

        if sae.cfg.sae_type == "mixcoder":
            assert mixcoder_settings is not None
            # assert "model_name" in mixcoder_settings and "tokenizer" in mixcoder_settings
            if mixcoder_settings.get("modality_indices", None) is None:
                modality_indices = get_modality_indices(mixcoder_settings["tokenizer"], mixcoder_settings["model_name"])
            else:
                modality_indices = {
                    k: torch.tensor(v, device=sae.cfg.device) for k, v in mixcoder_settings["modality_indices"].items()
                }

            sae.init_parameters(modality_indices=modality_indices)

        else:
            sae.init_parameters(
                encoder_uniform_bound=self.cfg.encoder_uniform_bound,
                decoder_uniform_bound=self.cfg.decoder_uniform_bound,
                init_log_jumprelu_threshold_value=self.cfg.init_log_jumprelu_threshold_value,
            )

        if self.cfg.init_decoder_norm:
            sae.set_decoder_to_fixed_norm(self.cfg.init_decoder_norm, force_exact=True)

        if self.cfg.init_encoder_with_decoder_transpose:
            sae.init_encoder_with_decoder_transpose(self.cfg.init_encoder_with_decoder_transpose_factor)
        else:
            if self.cfg.init_encoder_norm:
                sae.set_encoder_to_fixed_norm(self.cfg.init_encoder_norm)

        return sae

    @torch.no_grad()
    def initialize_tensor_parallel(self, sae: SparseAutoEncoder, device_mesh: DeviceMesh | None = None):
        if not device_mesh or device_mesh["model"].size(0) == 1:
            return sae
        if sae.cfg.sae_type == "sae":
            sae.device_mesh = device_mesh
            plan = {
                "encoder": ColwiseParallel(output_layouts=Replicate()),
                "decoder": RowwiseParallel(input_layouts=Replicate()),
            }
            if sae.cfg.use_glu_encoder:
                plan["encoder_glu"] = ColwiseParallel(output_layouts=Replicate())
            sae = parallelize_module(sae, device_mesh=device_mesh["model"], parallelize_plan=plan)  # type: ignore

        elif sae.cfg.sae_type == "mixcoder":
            warnings.warn("MixCoder is not supported for tensor parallel initialization.")
        return sae

    @torch.no_grad()
    def initialization_search(self, sae: SparseAutoEncoder, activation_batch: Dict[str, Tensor]):
        """
        This function is used to search for the best initialization norm for the SAE decoder.
        """
        activation_in, activation_out = (
            activation_batch[sae.cfg.hook_point_in],
            activation_batch[sae.cfg.hook_point_out],
        )
        tokens = activation_batch["tokens"]
        if self.cfg.init_decoder_norm is None:
            # assert sae.cfg.sparsity_include_decoder_norm, "Decoder norm must be included in sparsity loss"
            if not self.cfg.init_encoder_with_decoder_transpose or sae.cfg.hook_point_in != sae.cfg.hook_point_out:
                return sae

            def grid_search_best_init_norm(search_range: List[float]) -> float:
                losses: Dict[float, float] = {}

                for norm in search_range:
                    sae.set_decoder_to_fixed_norm(norm, force_exact=True)
                    sae.init_encoder_with_decoder_transpose(self.cfg.init_encoder_with_decoder_transpose_factor)
                    if sae.cfg.sae_type == "crosscoder":
                        sae.initialize_with_same_weight_across_layers()
                    mse = sae.compute_loss(activation_batch, tokens=tokens, return_aux_data=False).mean()
                    if sae.cfg.sae_type == "crosscoder":
                        mse = all_reduce_tensor(mse, aggregate="mean")
                    losses[norm] = mse.item()
                best_norm = min(losses, key=losses.get)  # type: ignore
                return best_norm

            if "topk" not in sae.cfg.act_fn:
                assert self.cfg.l1_coefficient is not None
                sae.set_current_l1_coefficient(self.cfg.l1_coefficient)
            best_norm_coarse = grid_search_best_init_norm(torch.linspace(0.1, 1, 10).numpy().tolist())  # type: ignore
            best_norm_fine_grained = grid_search_best_init_norm(
                torch.linspace(best_norm_coarse - 0.09, best_norm_coarse + 0.1, 20).numpy().tolist()  # type: ignore
            )

            print(f"The best (i.e. lowest MSE) initialized norm is {best_norm_fine_grained}")

            sae.set_decoder_to_fixed_norm(best_norm_fine_grained, force_exact=True)

        if self.cfg.bias_init_method == "geometric_median" and sae.cfg.sae_type != "mixcoder":
            # TODO: add support for MixCoder
            sae.decoder.bias.data = (
                sae.compute_norm_factor(activation_out, hook_point=sae.cfg.hook_point_out) * activation_out
            ).mean(0)

            if not sae.cfg.apply_decoder_bias_to_pre_encoder:
                normalized_input = (
                    sae.compute_norm_factor(activation_in, hook_point=sae.cfg.hook_point_in) * activation_in
                )
                normalized_median = normalized_input.mean(0)
                sae.encoder.bias.data = -normalized_median @ sae.encoder.weight.data.T

        if self.cfg.init_encoder_with_decoder_transpose:
            sae.init_encoder_with_decoder_transpose(self.cfg.init_encoder_with_decoder_transpose_factor)

        return sae

    @torch.no_grad()
    def initialize_encoder_bias_for_const_fire_times(self, sae: SparseAutoEncoder, activation_batch: Dict[str, Tensor]):
        """
        This function is used to initialize the encoder bias for constant fire times.
        """
        activation_in = activation_batch[sae.cfg.hook_point_in]
        tokens = activation_batch["tokens"]
        batch_size = activation_in.size(0)
        _, hidden_pre = sae.encode(activation_in, return_hidden_pre=True, tokens=tokens)
        k = int(self.cfg.const_times_for_init_b_e * batch_size / sae.cfg.d_sae)
        encoder_bias, _ = torch.kthvalue(hidden_pre, batch_size - k + 1, dim=0)
        sae.encoder.bias.data.copy_((sae.log_jumprelu_threshold.exp() - encoder_bias).to(dtype=torch.float32))
        # feature_act, hidden_pre = sae.encode(activation_in, return_hidden_pre=True, tokens=tokens)
        # print(hidden_pre.shape)
        # print(torch.sum(hidden_pre > sae.log_jumprelu_threshold.exp(), dim=0))
        # print(torch.sum(feature_act > 0, dim=0))
        # exit()
        return sae

    def initialize_sae_from_config(
        self,
        cfg: BaseSAEConfig,
        activation_stream: Iterable[dict[str, Tensor]] | None = None,
        activation_norm: dict[str, float] | None = None,
        device_mesh: DeviceMesh | None = None,
        mixcoder_settings: dict[str, Any] | None = None,
    ):
        """
        Initialize the SAE from the SAE config.
        Args:
            cfg (SAEConfig): The SAE config.
            activation_iter (Iterable[dict[str, Tensor]] | None): The activation iterator. Used for initialization search when self.cfg.init_search is True.
            activation_norm (dict[str, float] | None): The activation normalization. Used for dataset-wise normalization when self.cfg.norm_activation is "dataset-wise".
            device_mesh (DeviceMesh | None): The device mesh.
        """
        sae = None  # type: ignore
        if cfg.sae_type == "sae":
            sae = SparseAutoEncoder.from_config(cfg)
        elif cfg.sae_type == "mixcoder":
            sae = MixCoder.from_config(cfg)
        elif cfg.sae_type == "crosscoder":
            sae = CrossCoder.from_config(cfg)
        else:
            # TODO: add support for different SAE config types, e.g. MixCoderConfig, CrossCoderConfig, etc.
            raise ValueError(f"SAE type {cfg.sae_type} not supported.")
        if self.cfg.state == "training":
            if cfg.sae_pretrained_name_or_path is None:
                sae: SparseAutoEncoder = self.initialize_parameters(sae, mixcoder_settings=mixcoder_settings)
            if sae.cfg.norm_activation == "dataset-wise":
                if activation_norm is None:
                    assert (
                        activation_stream is not None
                    ), "Activation iterator must be provided for dataset-wise normalization"
                    activation_norm = calculate_activation_norm(
                        activation_stream, [cfg.hook_point_in, cfg.hook_point_out]
                    )
                sae.set_dataset_average_activation_norm(activation_norm)

            if self.cfg.bias_init_method == "init_b_e_for_const_fire_times":
                assert (
                    activation_stream is not None
                ), "Activation iterator must be provided for encoder bias initialization"
                activation_batch = next(iter(activation_stream))
                sae = self.initialize_encoder_bias_for_const_fire_times(sae, activation_batch)

            if self.cfg.init_search:
                assert activation_stream is not None, "Activation iterator must be provided for initialization search"
                activation_batch = next(iter(activation_stream))  # type: ignore
                sae = self.initialization_search(sae, activation_batch)

        elif self.cfg.state == "inference":
            if sae.cfg.norm_activation == "dataset-wise":
                sae.standardize_parameters_of_dataset_norm(activation_norm)
            if sae.cfg.sparsity_include_decoder_norm:
                sae.transform_to_unit_decoder_norm()

        sae = self.initialize_tensor_parallel(sae, device_mesh)
        return sae
