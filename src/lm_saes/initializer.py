from typing import Dict, Iterable, List

import torch
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from lm_saes.config import InitializerConfig, SAEConfig
from lm_saes.sae import SparseAutoEncoder


class Initializer:
    def __init__(self, cfg: InitializerConfig):
        self.cfg = cfg

    @torch.no_grad()
    def initialize_parameters(self, sae: SparseAutoEncoder):
        """Initialize the parameters of the SAE.
        Only used when the state is "training" to initialize sae.
        """
        torch.nn.init.kaiming_uniform_(sae.encoder.weight)
        torch.nn.init.kaiming_uniform_(sae.decoder.weight)
        torch.nn.init.zeros_(sae.encoder.bias)
        if sae.cfg.use_decoder_bias:
            torch.nn.init.zeros_(sae.decoder.bias)
        if sae.cfg.use_glu_encoder:
            torch.nn.init.kaiming_uniform_(sae.encoder_glu.weight)
            torch.nn.init.zeros_(sae.encoder_glu.bias)

        if self.cfg.init_decoder_norm:
            sae.set_decoder_to_fixed_norm(self.cfg.init_decoder_norm, force_exact=True)

        if self.cfg.init_encoder_with_decoder_transpose:
            sae.encoder.weight.data = sae.decoder.weight.data.T.clone().contiguous()
        else:
            if self.cfg.init_encoder_norm:
                sae.set_encoder_to_fixed_norm(self.cfg.init_encoder_norm)

        return sae

    @torch.no_grad()
    def initialize_tensor_parallel(self, sae: SparseAutoEncoder, device_mesh: DeviceMesh | None = None):
        if not device_mesh or device_mesh["model"].size(0) == 1:
            return sae
        sae.device_mesh = device_mesh
        plan = {
            "encoder": ColwiseParallel(output_layouts=Replicate()),
            "decoder": RowwiseParallel(input_layouts=Replicate()),
        }
        if sae.cfg.use_glu_encoder:
            plan["encoder_glu"] = ColwiseParallel(output_layouts=Replicate())
        sae = parallelize_module(sae, device_mesh=device_mesh["model"], parallelize_plan=plan)  # type: ignore
        return sae

    @torch.no_grad()
    def initialization_search(self, sae: SparseAutoEncoder, activation_batch: Dict[str, Tensor]):
        """
        This function is used to search for the best initialization norm for the SAE decoder.
        """
        assert sae.cfg.hook_point_out is not None
        activation_in, activation_out = (
            activation_batch[sae.cfg.hook_point_in],
            activation_batch[sae.cfg.hook_point_out],
        )

        if self.cfg.init_decoder_norm is None:
            assert sae.cfg.sparsity_include_decoder_norm, "Decoder norm must be included in sparsity loss"
            if not self.cfg.init_encoder_with_decoder_transpose or sae.cfg.hook_point_in != sae.cfg.hook_point_out:
                return sae

            def grid_search_best_init_norm(search_range: List[float]) -> float:
                losses: Dict[float, float] = {}

                for norm in search_range:
                    sae.set_decoder_to_fixed_norm(norm, force_exact=True)
                    sae.encoder.weight.data = sae.decoder.weight.data.T.clone().contiguous()
                    mse = (
                        sae.compute_loss(
                            x=activation_in,
                            label=activation_out,
                        )[1][0]["l_rec"]
                        .mean()
                        .item()
                    )
                    losses[norm] = mse
                best_norm = min(losses, key=losses.get)  # type: ignore
                return best_norm

            assert self.cfg.l1_coefficient is not None
            sae.set_current_l1_coefficient(self.cfg.l1_coefficient)
            best_norm_coarse = grid_search_best_init_norm(torch.linspace(0.1, 1, 10).numpy().tolist())  # type: ignore
            best_norm_fine_grained = grid_search_best_init_norm(
                torch.linspace(best_norm_coarse - 0.09, best_norm_coarse + 0.1, 20).numpy().tolist()  # type: ignore
            )

            print(f"The best (i.e. lowest MSE) initialized norm is {best_norm_fine_grained}")

            sae.set_decoder_to_fixed_norm(best_norm_fine_grained, force_exact=True)

        if self.cfg.bias_init_method == "geometric_median":
            sae.decoder.bias.data = (sae.compute_norm_factor(activation_out, hook_point="out") * activation_out).mean(0)

            if not sae.cfg.apply_decoder_bias_to_pre_encoder:
                normalized_input = sae.compute_norm_factor(activation_in, hook_point="in") * activation_in
                normalized_median = normalized_input.mean(0)
                sae.encoder.bias.data = -normalized_median @ sae.encoder.weight.data.T

        if self.cfg.init_encoder_with_decoder_transpose:
            sae.encoder.weight.data = sae.decoder.weight.data.T.clone().contiguous()

        return sae

    def initialize_sae_from_config(
        self,
        cfg: SAEConfig,
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
        sae = None  # type: ignore
        if isinstance(cfg, SAEConfig):
            sae = SparseAutoEncoder.from_config(cfg)
        else:
            # TODO: add support for different SAE config types, e.g. MixCoderConfig, CrossCoderConfig, etc.
            pass
        if self.cfg.state == "training":
            if cfg.sae_pretrained_name_or_path is None:
                sae: SparseAutoEncoder = self.initialize_parameters(sae)
            if sae.cfg.norm_activation == "dataset-wise":
                if self.cfg.is_activation_normalized:
                    print("Activation is normalized, using 1.0 as the activation norm")
                    activation_norm = {
                        "in": 1.0,
                        "out": 1.0,
                    }
                else:
                    assert (
                        activation_norm is not None
                    ), "Activation normalization must be provided if data from activation_iter is not normalized"
                sae.set_dataset_average_activation_norm(activation_norm)

            if self.cfg.init_search:
                assert activation_stream is not None, "Activation iterator must be provided for initialization search"
                activation_batch = next(iter(activation_stream))  # type: ignore
                sae = self.initialization_search(sae, activation_batch)

        elif self.cfg.state == "inference":
            if sae.cfg.norm_activation == "dataset-wise":
                sae.standardize_parameters_of_dataset_norm(activation_norm)
            if sae.cfg.sparsity_include_decoder_norm:
                sae.transform_to_unit_decoder_norm()
            if sae.cfg.act_fn == "topk" and sae.cfg.jump_relu_threshold > 0:
                print(
                    "Converting topk activation to jumprelu for inference. Features are set independent to each other."
                )
                sae.cfg.act_fn = "jumprelu"

        sae = self.initialize_tensor_parallel(sae, device_mesh)
        return sae
