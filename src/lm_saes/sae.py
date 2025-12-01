import math
from typing import Any, Literal, Union, overload

import torch
import torch.distributed.tensor
from jaxtyping import Float
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor
from transformer_lens.hook_points import HookPoint
from typing_extensions import override

from lm_saes.utils.distributed import DimMap
from lm_saes.utils.logging import get_distributed_logger

from .abstract_sae import AbstractSparseAutoEncoder
from .config import SAEConfig

try:
    from sae_lens import JumpReLUSAE, StandardSAE, TopKSAE
    sae_lens_warning = False
except ImportError:
    sae_lens_warning = True

logger = get_distributed_logger("sae")


class SparseAutoEncoder(AbstractSparseAutoEncoder):
    def __init__(self, cfg: SAEConfig, device_mesh: DeviceMesh | None = None):
        super(SparseAutoEncoder, self).__init__(cfg, device_mesh=device_mesh)
        self.cfg = cfg

        if device_mesh is None:
            self.W_E = nn.Parameter(torch.empty(cfg.d_model, cfg.d_sae, device=cfg.device, dtype=cfg.dtype))
            self.b_E = nn.Parameter(torch.empty(cfg.d_sae, device=cfg.device, dtype=cfg.dtype))
            self.W_D = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_model, device=cfg.device, dtype=cfg.dtype))
            if cfg.use_decoder_bias:
                self.b_D = nn.Parameter(torch.empty(cfg.d_model, device=cfg.device, dtype=cfg.dtype))

            if cfg.use_glu_encoder:
                self.W_E_glu = nn.Parameter(torch.empty(cfg.d_model, cfg.d_sae, device=cfg.device, dtype=cfg.dtype))
                self.b_E_glu = nn.Parameter(torch.empty(cfg.d_sae, device=cfg.device, dtype=cfg.dtype))
        else:
            self.W_E = nn.Parameter(
                torch.distributed.tensor.empty(
                    cfg.d_model,
                    cfg.d_sae,
                    dtype=cfg.dtype,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["W_E"].placements(device_mesh),
                )
            )
            self.b_E = nn.Parameter(
                torch.distributed.tensor.empty(
                    cfg.d_sae,
                    dtype=cfg.dtype,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["b_E"].placements(device_mesh),
                )
            )
            self.W_D = nn.Parameter(
                torch.distributed.tensor.empty(
                    cfg.d_sae,
                    cfg.d_model,
                    dtype=cfg.dtype,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["W_D"].placements(device_mesh),
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
            if cfg.use_glu_encoder:
                self.W_E_glu = nn.Parameter(
                    torch.distributed.tensor.empty(
                        cfg.d_model,
                        cfg.d_sae,
                        dtype=cfg.dtype,
                        device_mesh=device_mesh,
                        placements=self.dim_maps()["W_E_glu"].placements(device_mesh),
                    )
                )
                self.b_E_glu = nn.Parameter(
                    torch.distributed.tensor.empty(
                        cfg.d_sae,
                        dtype=cfg.dtype,
                        device_mesh=device_mesh,
                        placements=self.dim_maps()["b_E_glu"].placements(device_mesh),
                    )
                )

        self.hook_hidden_pre = HookPoint()
        self.hook_feature_acts = HookPoint()
        self.hook_reconstructed = HookPoint()

    @override
    def encoder_norm(self, keepdim: bool = False):
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
    def decoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Compute the norm of the decoder weight."""
        if not isinstance(self.W_D, DTensor):
            return torch.norm(self.W_D, p=2, dim=1, keepdim=keepdim).to(self.cfg.device)
        else:
            assert self.device_mesh is not None
            return DTensor.from_local(
                torch.norm(self.W_D.to_local(), p=2, dim=1, keepdim=keepdim),
                device_mesh=self.device_mesh,
                placements=DimMap({"model": 0}).placements(self.device_mesh),
            )

    @override
    def decoder_bias_norm(self) -> torch.Tensor:
        if not self.cfg.use_decoder_bias:
            raise ValueError("Decoder bias is not used")
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
    def set_decoder_to_fixed_norm(self, value: float, force_exact: bool):
        if force_exact:
            self.W_D.mul_(value / self.decoder_norm(keepdim=True))
        else:
            self.W_D.mul_(value / torch.clamp(self.decoder_norm(keepdim=True), min=value))

    @torch.no_grad()
    def set_encoder_to_fixed_norm(self, value: float):
        self.W_E.mul_(value / self.encoder_norm(keepdim=True))

    def dim_maps(self) -> dict[str, DimMap]:
        """Return a dictionary mapping parameter names to dimension maps.

        Returns:
            A dictionary mapping parameter names to DimMap objects.
        """
        parent_maps = super().dim_maps()
        sae_maps = {
            "W_E": DimMap({"model": 1}),
            "W_D": DimMap({"model": 0}),
            "b_E": DimMap({"model": 0}),
        }
        if self.cfg.use_decoder_bias:
            sae_maps["b_D"] = DimMap({})
        if self.cfg.use_glu_encoder:
            sae_maps["W_E_glu"] = DimMap({"model": 1})
            sae_maps["b_E_glu"] = DimMap({"model": 0})
        return parent_maps | sae_maps

    @override
    @torch.no_grad()
    def transform_to_unit_decoder_norm(self):
        self.W_D.mul_(1 / self.decoder_norm(keepdim=False))

    @torch.no_grad()
    def standardize_parameters_of_dataset_norm(self):  # should be overridden by subclasses due to side effects
        """
        Standardize the parameters of the model to account for dataset_norm during inference.
        This function should be called during inference by the Initializer.

        During training, the activations correspond to an input `x` where the norm is sqrt(d_model).
        However, during inference, the norm of the input `x` corresponds to the dataset_norm.
        To ensure consistency between training and inference, the activations during inference
        are scaled by the factor:

            scaled_activation = training_activation * (dataset_norm / sqrt(d_model))

        Args:
            dataset_average_activation_norm (dict[str, float]):
                A dictionary where keys represent in or out and values
                specify the average activation norm of the dataset during inference.

                dataset_average_activation_norm = {
                    self.cfg.hook_point_in: 1.0,
                    self.cfg.hook_point_out: 1.0,
                }

        Returns:
            None: Updates the internal parameters to reflect the standardized activations and change the norm_activation to "inference" mode.
        """
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
            assert self.b_D is not None, "Decoder bias should exist if use_decoder_bias is True"
            self.b_D.div_(output_norm_factor)
        self.W_D.mul_(input_norm_factor / output_norm_factor)
        self.cfg.norm_activation = "inference"

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
        """Encode input tensor through the sparse autoencoder.

        Args:
            x: Input tensor of shape (batch, d_model) or (batch, seq_len, d_model)
            return_hidden_pre: If True, also return the pre-activation hidden states

        Returns:
            If return_hidden_pre is False:
                Feature activations tensor of shape (batch, d_sae) or (batch, seq_len, d_sae)
            If return_hidden_pre is True:
                Tuple of (feature_acts, hidden_pre) where both have shape (batch, d_sae) or (batch, seq_len, d_sae)
        """
        # Pass through encoder
        hidden_pre = x @ self.W_E + self.b_E

        # Apply GLU if configured
        if self.cfg.use_glu_encoder:
            hidden_pre_glu = torch.sigmoid(x @ self.W_E_glu + self.b_E_glu)
            hidden_pre = hidden_pre * hidden_pre_glu

        hidden_pre = self.hook_hidden_pre(hidden_pre)

        # Scale feature activations by decoder norm if configured
        if self.cfg.sparsity_include_decoder_norm:
            hidden_pre = hidden_pre * self.decoder_norm()

        feature_acts = self.activation_function(hidden_pre)
        feature_acts = self.hook_feature_acts(feature_acts)

        if self.cfg.sparsity_include_decoder_norm:
            feature_acts = feature_acts / self.decoder_norm()
            hidden_pre = hidden_pre / self.decoder_norm()

        if return_hidden_pre:
            return feature_acts, hidden_pre
        return feature_acts

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
    ]:  # may be overridden by subclasses
        max_l0_in_batch = feature_acts.gt(0).to(feature_acts).sum(dim=-1).max()
        sparsity_threshold = self.cfg.d_sae * (1 - self.cfg.sparsity_threshold_for_triton_spmm_kernel)
        if (
            self.cfg.use_triton_kernel and 0 < max_l0_in_batch < sparsity_threshold
        ):  # triton kernel cannot handle empty feature_acts
            from .kernels import decode_with_triton_spmm_kernel

            reconstructed = decode_with_triton_spmm_kernel(feature_acts, self.W_D.T.contiguous())
        else:
            reconstructed = feature_acts @ self.W_D

        assert reconstructed is not None, "Reconstructed cannot be None"
        if self.cfg.use_decoder_bias:
            reconstructed = reconstructed + self.b_D
        reconstructed = self.hook_reconstructed(reconstructed)

        if isinstance(reconstructed, DTensor):
            reconstructed = DimMap({"data": 0}).redistribute(reconstructed)

        return reconstructed

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
        reconstructed = self.decode(feature_acts, **kwargs)
        return reconstructed

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, strict_loading: bool = True, fold_activation_scale:bool = True,  **kwargs):
        cfg = SAEConfig.from_pretrained(pretrained_name_or_path, strict_loading=strict_loading, **kwargs)
        return cls.from_config(cfg, fold_activation_scale=fold_activation_scale)

    @torch.no_grad()
    def _init_encoder_with_decoder_transpose(
        self, encoder: torch.nn.Linear, decoder: torch.nn.Linear, factor: float = 1.0
    ):
        encoder.weight.data = decoder.weight.data.T.clone().contiguous() * factor

    @override
    @torch.no_grad()
    def init_encoder_with_decoder_transpose(self, factor: float = 1.0):
        self.W_E.copy_(self.W_D.contiguous().T * factor)

    @override
    @torch.no_grad()
    def init_parameters(self, **kwargs):
        super().init_parameters(**kwargs)

        W_E = torch.empty(self.cfg.d_model, self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype).uniform_(
            -kwargs["encoder_uniform_bound"], kwargs["encoder_uniform_bound"]
        )
        W_D = torch.empty(self.cfg.d_sae, self.cfg.d_model, device=self.cfg.device, dtype=self.cfg.dtype).uniform_(
            -kwargs["decoder_uniform_bound"], kwargs["decoder_uniform_bound"]
        )
        b_E = torch.zeros(self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype)

        if self.device_mesh is not None:
            W_E = self.dim_maps()["W_E"].distribute(W_E, self.device_mesh)
            W_D = self.dim_maps()["W_D"].distribute(W_D, self.device_mesh)
            b_E = self.dim_maps()["b_E"].distribute(b_E, self.device_mesh)

        self.W_E.copy_(W_E)
        self.W_D.copy_(W_D)
        self.b_E.copy_(b_E)

        if self.cfg.use_decoder_bias:
            b_D = torch.zeros(self.cfg.d_model, device=self.cfg.device, dtype=self.cfg.dtype)

            if self.device_mesh is not None:
                b_D = self.dim_maps()["b_D"].distribute(b_D, self.device_mesh)

            self.b_D.copy_(b_D)

        if self.cfg.use_glu_encoder:
            W_E_glu = torch.empty(
                self.cfg.d_model, self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype
            ).uniform_(-kwargs["encoder_uniform_bound"], kwargs["encoder_uniform_bound"])
            if self.device_mesh is not None:
                W_E_glu = self.dim_maps()["W_E_glu"].distribute(W_E_glu, self.device_mesh)
            self.W_E_glu.copy_(W_E_glu)

    @override
    def prepare_input(
        self, batch: dict[str, torch.Tensor], **kwargs
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        x = batch[self.cfg.hook_point_in]
        return x, {}, {}

    @override
    def prepare_label(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        label = batch[self.cfg.hook_point_out]
        return label

    @override
    @torch.no_grad()
    def init_W_D_with_active_subspace(self, activation_batch: dict[str, torch.Tensor], d_active_subspace: int):
        """Initialize W_D with the active subspace.

        Args:
            activation_batch: The activation batch.
            d_active_subspace: The dimension of the active subspace.
        """
        label = self.prepare_label(activation_batch)
        demeaned_label = label - label.mean(dim=0)
        U, S, V = torch.svd(demeaned_label.T.to(torch.float32))
        proj_weight = U[:, :d_active_subspace]  # [d_model, d_active_subspace]
        self.W_D.copy_(self.W_D.data[:, :d_active_subspace] @ proj_weight.T.to(self.cfg.dtype))

    @torch.no_grad()
    def init_encoder_bias_with_mean_hidden_pre(self, activation_batch: dict[str, torch.Tensor]):
        x = self.prepare_input(activation_batch)[0]
        _, hidden_pre = self.encode(x, return_hidden_pre=True)
        self.b_E.copy_(-hidden_pre.mean(dim=0))
    
    @classmethod
    def from_saelens(cls, sae_saelens):
        
        # Check env
        assert not sae_lens_warning, "Warning: sae_lens library not found."
        
        # Check Configuration
        assert sae_saelens.cfg.reshape_activations == 'none', f"The 'reshape_activations' should be 'none' but get {sae_saelens.cfg.reshape_activations}."
        assert not sae_saelens.cfg.apply_b_dec_to_input, f"The 'apply_b_dec_to_input' should be 'False' but get {sae_saelens.cfg.apply_b_dec_to_input}."
        assert sae_saelens.cfg.normalize_activations == 'none', f"The 'normalize_activations' should be 'false' but get {sae_saelens.cfg.normalize_activations}."
        
        # Parse
        d_model = sae_saelens.cfg.d_in
        d_sae = sae_saelens.cfg.d_sae
        hook_name = sae_saelens.cfg.metadata.hook_name
        dtype = sae_saelens.W_enc.dtype
        
        rescale_acts_by_decoder_norm = False
        jumprelu_threshold_window = 0
        k = 0
        if isinstance(sae_saelens, StandardSAE):
            activation_fn = 'relu'
        elif isinstance(sae_saelens, TopKSAE):
            activation_fn = 'topk'
            k = sae_saelens.cfg.k
            rescale_acts_by_decoder_norm = sae_saelens.cfg.rescale_acts_by_decoder_norm
        elif isinstance(sae_saelens, JumpReLUSAE):
            activation_fn = 'jumprelu'
            jumprelu_threshold_window = 2
        
        print('act_fn', activation_fn)
        # create cfg
        cfg = SAEConfig(
            sae_type = "sae",
            hook_point_in = hook_name,
            hook_point_out = hook_name,
            dtype = dtype,
            d_model = d_model,
            act_fn = activation_fn,
            jumprelu_threshold_window=jumprelu_threshold_window,
            top_k = k,
            expansion_factor = d_sae / d_model,
            sparsity_include_decoder_norm = rescale_acts_by_decoder_norm,
        )
        
        model = cls.from_config(cfg, None)
        
        with torch.no_grad():
            model.W_D.data.copy_(sae_saelens.W_dec)
            model.W_E.data.copy_(sae_saelens.W_enc)
            model.b_D.data.copy_(sae_saelens.b_dec)
            model.b_E.data.copy_(sae_saelens.b_enc)
            
            if isinstance(sae_saelens, JumpReLUSAE):
                model.activation_function.log_jumprelu_threshold.copy_(torch.log(sae_saelens.threshold.clone().detach()))
        
        return model
