import os
import warnings
from typing import Dict, Iterable, List, cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from lm_saes.abstract_sae import AbstractSparseAutoEncoder, JumpReLU
from lm_saes.config import BaseSAEConfig, InitializerConfig
from lm_saes.crosscoder import CrossCoder
from lm_saes.mixcoder import MixCoder
from lm_saes.sae import SparseAutoEncoder
from lm_saes.utils.misc import calculate_activation_norm
from lm_saes.utils.tensor_dict import batch_size


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
        else:
            if self.cfg.init_encoder_norm:
                sae.set_encoder_to_fixed_norm(self.cfg.init_encoder_norm)

        return sae

    @torch.no_grad()
    def initialize_tensor_parallel(self, sae: AbstractSparseAutoEncoder, device_mesh: DeviceMesh | None = None):
        if not device_mesh:
            return sae

        if isinstance(sae, MixCoder):
            # TODO: add support for MixCoder
            warnings.warn("MixCoder is not supported for tensor parallel initialization.")
            return sae

        if isinstance(sae, SparseAutoEncoder):
            if device_mesh["model"].size(0) == 1:
                return sae
            sae.device_mesh = device_mesh
            plan = {
                "encoder": ColwiseParallel(output_layouts=Replicate()),
                "decoder": RowwiseParallel(input_layouts=Replicate()),
            }
            if sae.cfg.use_glu_encoder:
                plan["encoder_glu"] = ColwiseParallel(output_layouts=Replicate())
            sae = parallelize_module(sae, device_mesh=device_mesh["model"], parallelize_plan=plan)  # type: ignore

        elif isinstance(sae, CrossCoder):
            sae.tensor_parallel(device_mesh)

        return sae

    @torch.no_grad()
    def initialization_search(
        self, sae: AbstractSparseAutoEncoder, activation_batch: Dict[str, Tensor], device_mesh: DeviceMesh | None = None
    ):
        """
        This function is used to search for the best initialization norm for the SAE decoder.
        """
        batch = sae.normalize_activations(activation_batch)

        if (
            isinstance(sae, CrossCoder)
            and device_mesh is not None
            and "head" in cast(tuple[str, ...], device_mesh.mesh_dim_names)
        ):
            object_list = [None] * device_mesh.get_group("head").size()
            dist.all_gather_object(object_list, activation_batch, group=device_mesh.get_group("head"))
            activation_batch = {
                k: v.to(torch.device("cuda", int(os.environ["LOCAL_RANK"]))) if isinstance(v, Tensor) else v
                for d in cast(list[dict[str, Tensor]], object_list)
                for k, v in d.items()
            }

        if self.cfg.init_decoder_norm is None:
            if not self.cfg.init_encoder_with_decoder_transpose:
                return sae

            def grid_search_best_init_norm(search_range: List[float]) -> float:
                losses: Dict[float, float] = {}

                for norm in search_range:
                    sae.set_decoder_to_fixed_norm(norm, force_exact=True)
                    sae.init_encoder_with_decoder_transpose(self.cfg.init_encoder_with_decoder_transpose_factor)
                    mse = sae.compute_loss(activation_batch)[1][0]["l_rec"].mean().item()
                    losses[norm] = mse
                best_norm = min(losses, key=losses.get)  # type: ignore
                return best_norm

            best_norm_coarse = grid_search_best_init_norm(torch.linspace(0.1, 1, 10).numpy().tolist())  # type: ignore
            best_norm_fine_grained = grid_search_best_init_norm(
                torch.linspace(best_norm_coarse - 0.09, best_norm_coarse + 0.1, 20).numpy().tolist()  # type: ignore
            )

            print(f"The best (i.e. lowest MSE) initialized norm is {best_norm_fine_grained}")

            sae.set_decoder_to_fixed_norm(best_norm_fine_grained, force_exact=True)

        if self.cfg.bias_init_method == "geometric_median" and sae.cfg.sae_type != "mixcoder":
            # TODO: add support for MixCoder
            assert isinstance(sae, SparseAutoEncoder), (
                "SparseAutoEncoder is the only supported SAE type for encoder bias initialization"
            )
            sae.decoder.bias.data = (
                sae.compute_norm_factor(batch[sae.cfg.hook_point_out], hook_point=sae.cfg.hook_point_out)
                * batch[sae.cfg.hook_point_out]
            ).mean(0)

            if not sae.cfg.apply_decoder_bias_to_pre_encoder:
                normalized_input = (
                    sae.compute_norm_factor(batch[sae.cfg.hook_point_in], hook_point=sae.cfg.hook_point_in)
                    * batch[sae.cfg.hook_point_in]
                )
                normalized_median = normalized_input.mean(0)
                sae.encoder.bias.data = -normalized_median @ sae.encoder.weight.data.T

        if self.cfg.init_encoder_with_decoder_transpose:
            sae.init_encoder_with_decoder_transpose(self.cfg.init_encoder_with_decoder_transpose_factor)

        return sae

    @torch.no_grad()
    def initialize_encoder_bias_for_const_fire_times(
        self, sae: AbstractSparseAutoEncoder, activation_batch: Dict[str, Tensor]
    ):
        """
        This function is used to initialize the encoder bias for constant fire times.
        """
        assert isinstance(sae, SparseAutoEncoder), (
            "SparseAutoEncoder is the only supported SAE type for encoder bias initialization"
        )
        batch = sae.normalize_activations(activation_batch)
        x, kwargs = sae.prepare_input(batch)
        _, hidden_pre = sae.encode(x, **kwargs, return_hidden_pre=True)
        k = int(self.cfg.const_times_for_init_b_e * batch_size(batch) / sae.cfg.d_sae)
        encoder_bias, _ = torch.kthvalue(hidden_pre, batch_size(batch) - k + 1, dim=0)
        assert isinstance(sae.activation_function, JumpReLU)
        sae.encoder.bias.data.copy_(
            (sae.activation_function.log_jumprelu_threshold.exp() - encoder_bias).to(dtype=torch.float32)
        )
        return sae

    @torch.no_grad()
    def initialize_jump_relu_threshold(self, sae: AbstractSparseAutoEncoder, activation_batch: Dict[str, Tensor]):
        # TODO: add support for MixCoder
        if sae.cfg.sae_type == "mixcoder":
            warnings.warn("MixCoder is not supported for jump_relu_threshold initialization.")
            return sae

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
            sae: AbstractSparseAutoEncoder = SparseAutoEncoder.from_config(cfg)
        elif cfg.sae_type == "mixcoder":
            sae: AbstractSparseAutoEncoder = MixCoder.from_config(cfg)
        elif cfg.sae_type == "crosscoder":
            sae: AbstractSparseAutoEncoder = CrossCoder.from_config(cfg)
        else:
            # TODO: add support for different SAE config types, e.g. MixCoderConfig, CrossCoderConfig, etc.
            raise ValueError(f"SAE type {cfg.sae_type} not supported.")
        if self.cfg.state == "training":
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

            if self.cfg.bias_init_method == "init_b_e_for_const_fire_times":
                assert activation_stream is not None, (
                    "Activation iterator must be provided for encoder bias initialization"
                )
                activation_batch = next(iter(activation_stream))
                sae = self.initialize_encoder_bias_for_const_fire_times(sae, activation_batch)

            if self.cfg.init_search:
                assert activation_stream is not None, "Activation iterator must be provided for initialization search"
                activation_batch = next(iter(activation_stream))  # type: ignore
                sae = self.initialization_search(sae, activation_batch, device_mesh=device_mesh)

        elif self.cfg.state == "inference":
            if sae.cfg.norm_activation == "dataset-wise":
                sae.standardize_parameters_of_dataset_norm(activation_norm)
            if sae.cfg.sparsity_include_decoder_norm:
                sae.transform_to_unit_decoder_norm()
            if "topk" in sae.cfg.act_fn:
                print(
                    "Converting topk activation to jumprelu for inference. Features are set independent to each other."
                )
                if sae.cfg.jump_relu_threshold is None:
                    assert activation_stream is not None, (
                        "Activation iterator must be provided for jump_relu_threshold initialization"
                    )
                    activation_batch = next(iter(activation_stream))
                    self.initialize_jump_relu_threshold(sae, activation_batch)
                if cfg.sae_type != "mixcoder":  # TODO: add support for MixCoder
                    sae.cfg.act_fn = "jumprelu"

        sae = self.initialize_tensor_parallel(sae, device_mesh)
        return sae
