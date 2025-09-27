import math
import os
from abc import ABC, abstractmethod
from importlib.metadata import version
from pathlib import Path
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Self,
    Union,
    cast,
    overload,
)

import einops
import safetensors.torch as safe
import torch
import torch.distributed.checkpoint as dcp
from jaxtyping import Float
from safetensors import safe_open
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from transformer_lens.hook_points import HookedRootModule

from lm_saes.database import MongoClient
from lm_saes.utils.distributed import DimMap
from lm_saes.utils.huggingface import parse_pretrained_name_or_path
from lm_saes.utils.logging import get_distributed_logger
from lm_saes.utils.misc import is_primary_rank
from lm_saes.utils.timer import timer

from lm_saes.activation_functions import JumpReLU
from lm_saes.config import BaseSAEConfig

logger = get_distributed_logger("abstract_sae")


class AbstractSparseAutoEncoder(HookedRootModule, ABC):
    """Abstract base class for sparse autoencoder models.

    This class defines the public interface for all sparse autoencoder implementations.
    Concrete implementations should inherit from this class and implement the required methods.
    """

    def __init__(self, cfg: BaseSAEConfig, device_mesh: Optional[DeviceMesh] = None):
        super(AbstractSparseAutoEncoder, self).__init__()
        self.cfg = cfg

        # should be set by Trainer during training
        self.current_k = cfg.top_k

        # if cfg.norm_activation == "dataset-wise", the dataset average activation norm should be
        # calculated by the initializer before training starts and set by standardize_parameters_of_dataset_activation_scaling
        self.dataset_average_activation_norm: dict[str, float] | None = None

        self.device_mesh: DeviceMesh | None = device_mesh

        self.activation_function: Callable[[torch.Tensor], torch.Tensor] = self.activation_function_factory(device_mesh)
        self.circuit_tracing_mode: bool = cfg.circuit_tracing_mode

    @torch.no_grad()
    def set_dataset_average_activation_norm(self, dataset_average_activation_norm: dict[str, float]):
        """Set the dataset average activation norm for training or inference.
        dataset_average_activation_norm is set by the Initializer and only used when cfg.norm_activation == "dataset-wise".
        """
        self.dataset_average_activation_norm = dataset_average_activation_norm

    @torch.no_grad()
    def set_current_k(self, current_k: int):
        """Set the current k for topk activation function.
        This should be set by the Trainer during training.
        """
        self.current_k = current_k

    @abstractmethod
    @torch.no_grad()
    def set_decoder_to_fixed_norm(self, value: float, force_exact: bool):
        """Set the decoder to a fixed norm."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    @torch.no_grad()
    def set_encoder_to_fixed_norm(self, value: float):
        """Set the encoder to a fixed norm."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    @torch.no_grad()
    def transform_to_unit_decoder_norm(self):
        """Transform the model to have unit decoder norm."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    @torch.no_grad()
    def standardize_parameters_of_dataset_norm(self):
        """Standardize the parameters of the model to account for dataset_norm during inference."""
        raise NotImplementedError("Subclasses must implement this method")

    @torch.no_grad()
    def full_state_dict(self):  # should be overridden by subclasses
        state_dict = self.state_dict()

        # Add dataset_average_activation_norm to state dict
        if self.dataset_average_activation_norm is not None:
            for hook_point, value in self.dataset_average_activation_norm.items():
                state_dict[f"dataset_average_activation_norm.{hook_point}"] = torch.tensor(
                    value, device=self.cfg.device, dtype=self.cfg.dtype
                )
        else:
            for hook_point in self.cfg.associated_hook_points:
                state_dict[f"dataset_average_activation_norm.{hook_point}"] = torch.empty(
                    (), device=self.cfg.device, dtype=self.cfg.dtype
                )

        return cast(dict[str, torch.Tensor], state_dict)

    @torch.no_grad()
    def save_checkpoint(self, ckpt_path: Path | str) -> None:
        # TODO: save the config to MongoDB
        """
        {
            "name": sae_name,
            "config": sae_config,
            "path": final_ckpt_path,
        }
        """
        state_dict = self.full_state_dict()
        if Path(ckpt_path).suffix == ".safetensors":
            state_dict = {k: v.full_tensor() if isinstance(v, DTensor) else v for k, v in state_dict.items()}
            if self.device_mesh is None or is_primary_rank(self.device_mesh):
                safe.save_file(state_dict, ckpt_path, {"version": version("lm-saes")})
        elif Path(ckpt_path).suffix == ".pt":
            state_dict = {k: v.full_tensor() if isinstance(v, DTensor) else v for k, v in state_dict.items()}
            if self.device_mesh is None or is_primary_rank(self.device_mesh):
                torch.save({"sae": state_dict, "version": version("lm-saes")}, ckpt_path)
        elif Path(ckpt_path).suffix == ".dcp":
            fs_writer = FileSystemWriter(ckpt_path)
            dcp.save(state_dict, storage_writer=fs_writer)
        else:
            raise ValueError(
                f"Invalid checkpoint path {ckpt_path}. Currently only supports .safetensors and .pt formats."
            )

    @torch.no_grad()
    def save_pretrained(
        self,
        save_path: Path | str,
        sae_name: str | None = None,
        sae_series: str | None = None,
        mongo_client: MongoClient | None = None,
    ) -> None:
        os.makedirs(Path(save_path), exist_ok=True)

        if self.device_mesh is None:
            self.save_checkpoint(Path(save_path) / "sae_weights.safetensors")
        else:
            self.save_checkpoint(Path(save_path) / "sae_weights.dcp")
        if is_primary_rank(self.device_mesh):
            if mongo_client is not None:
                assert sae_name is not None and sae_series is not None, (
                    "sae_name and sae_series must be provided when saving to MongoDB"
                )
                mongo_client.create_sae(
                    name=sae_name, series=sae_series, path=str(Path(save_path).absolute()), cfg=self.cfg
                )

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

    @abstractmethod
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
        Ensure that the input activations are normalized by calling `normalize_activations` before calling this method.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
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
        """Decode feature activations to reconstructed input."""
        raise NotImplementedError("Subclasses must implement this method")

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
        """Forward pass through the autoencoder.
        Ensure that the input activations are normalized by calling `normalize_activations` before calling this method.
        """
        feature_acts = self.encode(x, **kwargs)
        reconstructed = self.decode(feature_acts, **kwargs)
        return reconstructed

    def compute_norm_factor(self, x: torch.Tensor, hook_point: str) -> torch.Tensor:
        """Compute the normalization factor for the activation vectors.
        This should be called during forward pass.
        There are four modes for norm_activation:
        - "token-wise": normalize by the token-wise norm calculated with the input x
        - "batch-wise": normalize by the batch-wise norm calculated with the input x
        - "dataset-wise": normalize by the dataset-wise norm initialized by the Initializer with sae.set_dataset_average_activation_norm method
        - "inference": no normalization, the weights are already normalized by the Initializer with sae.standardize_parameters_of_dataset_norm method
        """
        assert self.cfg.norm_activation in [
            "token-wise",
            "batch-wise",
            "dataset-wise",
            "inference",
        ], f"Not implemented norm_activation {self.cfg.norm_activation}"
        if self.cfg.norm_activation == "token-wise":
            return math.sqrt(self.cfg.d_model) / torch.norm(x, 2, dim=-1, keepdim=True)
        if self.cfg.norm_activation == "batch-wise":
            return math.sqrt(self.cfg.d_model) / torch.norm(x, 2, dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        if self.cfg.norm_activation == "dataset-wise":
            assert self.dataset_average_activation_norm is not None, (
                "dataset_average_activation_norm must be provided from Initializer"
            )
            return torch.tensor(
                math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[hook_point],
                device=x.device,
                dtype=x.dtype,
            )
        if self.cfg.norm_activation == "inference":
            return torch.tensor(1.0, device=x.device, dtype=x.dtype)
        raise ValueError(f"Not implemented norm_activation {self.cfg.norm_activation}")

    @timer.time("normalize_activations")
    def normalize_activations(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Normalize the input activations.
        This should be called before calling `encode` or `compute_loss`.
        """

        def normalize_hook_point(hook_point: str, original_tensor: torch.Tensor):
            norm_factor = self.compute_norm_factor(original_tensor, hook_point=hook_point)
            return original_tensor * norm_factor

        return {k: normalize_hook_point(k, v) if k in self.cfg.associated_hook_points else v for k, v in batch.items()}

    @abstractmethod
    @torch.no_grad()
    def init_encoder_with_decoder_transpose(self, factor: float = 1.0):
        """Initialize the encoder with the transpose of the decoder."""
        raise NotImplementedError("Subclasses must implement this method")

    def get_parameters(self) -> list[dict[str, Any]]:
        """Get the parameters of the model for optimization."""
        return [{"params": self.parameters()}]

    def load_full_state_dict(self, state_dict: dict[str, torch.Tensor], device_mesh: DeviceMesh | None = None) -> None:
        # Extract and set dataset_average_activation_norm if present
        norm_keys = [k for k in state_dict.keys() if k.startswith("dataset_average_activation_norm.")]
        if norm_keys:
            dataset_norm = {key.split(".", 1)[1]: state_dict[key].item() for key in norm_keys}
            self.set_dataset_average_activation_norm(dataset_norm)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("dataset_average_activation_norm.")}
        if device_mesh is None or any(isinstance(v, DTensor) for v in state_dict.values()):
            # Non-distributed checkpoint or DCP checkpoint
            # Load the state dict through torch API
            for k, v in state_dict.items():
                if any(isinstance(v, DTensor) for v in state_dict.values()) and not isinstance(v, DTensor):
                    state_dict[k] = DimMap({}).distribute(v, device_mesh)
            self.load_state_dict(state_dict, strict=self.cfg.strict_loading)
        else:
            # Full checkpoint (in .safetensors or .pt format) to be loaded distributedly
            self.load_distributed_state_dict(state_dict, device_mesh)

    @classmethod
    def from_config(cls, cfg: BaseSAEConfig, device_mesh: DeviceMesh | None = None, fold_activation_scale: bool = True) -> Self:
        model = cls(cfg, device_mesh)
        if cfg.sae_pretrained_name_or_path is None:
            total_params = sum(param.numel() for param in model.parameters()) / 1e9
            logger.info(f"Initializing {cfg.sae_type} from scratch with {total_params:.2f} B parameters")
            return model

        path = parse_pretrained_name_or_path(cfg.sae_pretrained_name_or_path)
        if path.endswith(".pt") or path.endswith(".safetensors") or path.endswith(".dcp"):
            ckpt_path = path
        else:
            ckpt_prioritized_paths = [
                f"{path}/sae_weights.safetensors",
                f"{path}/sae_weights.pt",
                f"{path}/sae_weights.dcp",
                f"{path}/checkpoints/pruned.safetensors",
                f"{path}/checkpoints/pruned.pt",
                f"{path}/checkpoints/pruned.dcp",
                f"{path}/checkpoints/final.safetensors",
                f"{path}/checkpoints/final.pt",
                f"{path}/checkpoints/final.dcp",
            ]
            for ckpt_path in ckpt_prioritized_paths:
                if os.path.exists(ckpt_path):
                    break
            else:
                raise FileNotFoundError(f"Pretrained model not found at {cfg.sae_pretrained_name_or_path}")

        if ckpt_path.endswith(".safetensors"):
            if device_mesh is None:
                state_dict: dict[str, torch.Tensor] = safe.load_file(ckpt_path, device=cfg.device)
            else:
                with safe_open(ckpt_path, device=int(os.environ["LOCAL_RANK"]), framework="pt") as f:

                    def load_tensor(key: str) -> DTensor:
                        tensor_slice = f.get_slice(key)
                        shape = tensor_slice.get_shape()
                        dim_map = model.dim_maps()[key]
                        indices = dim_map.local_slices(shape, device_mesh)
                        local_tensor = tensor_slice[indices].to(model.override_dtypes().get(key, cfg.dtype))
                        return DTensor.from_local(
                            local_tensor,
                            device_mesh=device_mesh,
                            placements=dim_map.placements(device_mesh),
                        )

                    state_dict = {k: load_tensor(k) if k in model.dim_maps() else f.get_tensor(k) for k in f.keys()}
        elif ckpt_path.endswith(".pt"):
            state_dict: dict[str, torch.Tensor] = torch.load(
                ckpt_path,
                map_location=cfg.device,
                weights_only=True,
            )["sae"]
        elif ckpt_path.endswith(".dcp"):
            # DCP checkpoint
            fs_reader = FileSystemReader(ckpt_path)
            state_dict = model.full_state_dict()
            dcp.load(state_dict, storage_reader=fs_reader)
        else:
            raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")                

        model.load_full_state_dict(state_dict, device_mesh)
        if fold_activation_scale:
            model.standardize_parameters_of_dataset_norm()
        return model

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, strict_loading: bool = True, **kwargs):
        """Load a pretrained model."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def encoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Compute the norm of the encoder."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def decoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Compute the norm of the decoder."""
        raise NotImplementedError("Subclasses must implement this method")

    def decoder_norm_full(self, keepdim: bool = False) -> torch.Tensor:
        """Compute the full norm of the decoder."""
        decoder_norm = self.decoder_norm(keepdim=keepdim)
        if not isinstance(decoder_norm, DTensor):
            return decoder_norm
        else:
            return decoder_norm.full_tensor()

    @abstractmethod
    def decoder_bias_norm(self) -> torch.Tensor:
        """Compute the norm of the decoder bias."""
        if self.cfg.use_decoder_bias:
            raise NotImplementedError("Subclasses must implement this method")
        else:
            raise ValueError("Decoder bias norm is not supported for models without decoder bias")

    @torch.no_grad()
    def log_statistics(self):
        log_dict = {
            "metrics/encoder_norm": self.encoder_norm().mean().item(),
            "metrics/decoder_norm": self.decoder_norm().mean().item(),
        }
        if self.cfg.use_decoder_bias:
            log_dict["metrics/decoder_bias_norm"] = self.decoder_bias_norm().mean().item()
        if "topk" in self.cfg.act_fn:
            log_dict["sparsity/k"] = self.current_k
        if isinstance(self.activation_function, JumpReLU):
            log_dict["metrics/mean_jumprelu_threshold"] = (
                self.activation_function.log_jumprelu_threshold.exp().mean().item()
            )
        return log_dict

    def activation_function_factory(
        self, device_mesh: DeviceMesh | None = None
    ) -> Callable[[torch.Tensor], torch.Tensor] | JumpReLU:
        assert self.cfg.act_fn.lower() in [
            "relu",
            "topk",
            "jumprelu",
            "batchtopk",
        ], f"Not implemented activation function {self.cfg.act_fn}"
        if self.cfg.act_fn.lower() == "relu":
            return lambda x: x * x.gt(0).to(x.dtype)
        elif self.cfg.act_fn.lower() == "jumprelu":
            return JumpReLU(
                self.cfg.jumprelu_threshold_window,
                shape=(self.cfg.d_sae,),
                device=self.cfg.device,
                dtype=self.cfg.dtype if self.cfg.promote_act_fn_dtype is None else self.cfg.promote_act_fn_dtype,
                device_mesh=device_mesh,
            )

        elif self.cfg.act_fn.lower() == "topk":

            if self.device_mesh is not None:
                from lm_saes.utils.distributed import distributed_topk
                
                def topk_activation(x: Union[
                        Float[torch.Tensor, "batch d_sae"],
                        Float[torch.Tensor, "batch seq_len d_sae"],
                    ],
                ):
                    return distributed_topk(
                        x, 
                        k=self.current_k, 
                        device_mesh=self.device_mesh, 
                        dim=-1, 
                        mesh_dim_name="model",
                    )
            else:
                def topk_activation(
                    x: Union[
                        Float[torch.Tensor, "batch d_sae"],
                        Float[torch.Tensor, "batch seq_len d_sae"],
                    ],
                ):
                    from lm_saes.utils.math import topk
                    return topk(
                        x,
                        k=self.current_k,
                        dim=-1,
                    )

            return topk_activation

        elif self.cfg.act_fn.lower() == "batchtopk":

            if device_mesh is not None:
                from lm_saes.utils.distributed import distributed_topk
                
                def batch_topk(x: Union[
                        Float[torch.Tensor, "batch d_sae"],
                        Float[torch.Tensor, "batch seq_len d_sae"],
                    ],
                ):
                    x = x * x.gt(0).to(x.dtype)
                    original_shape = None
                    if x.ndim == 3:
                        original_shape = (x.size(0), x.size(1))
                        x = x.flatten(end_dim=1)
                    result = distributed_topk(
                        x,
                        k=self.current_k * x.size(0),
                        device_mesh=device_mesh,
                        dim=(-2, -1),
                        mesh_dim_name="model",
                    )
                    if original_shape is not None:
                        result = result.unflatten(dim=0, sizes=original_shape)
                    return result
            else:
                # single-GPU batchtopk
                from lm_saes.utils.math import topk
                def batch_topk(x: Union[
                        Float[torch.Tensor, "batch d_sae"],
                        Float[torch.Tensor, "batch seq_len d_sae"],
                    ],
                ):
                    x = x * x.gt(0).to(x.dtype)
                    original_shape = None
                    if x.ndim == 3:
                        original_shape = (x.size(0), x.size(1))
                        x = x.flatten(end_dim=1)
                    result = topk(
                        x,
                        k=self.current_k * x.size(0),
                        dim=(-2, -1),
                    )
                    if original_shape is not None:
                        result = result.unflatten(dim=0, sizes=original_shape)
                    return result
            
            return batch_topk

        raise ValueError(f"Not implemented activation function {self.cfg.act_fn}")

    @torch.no_grad()
    def init_parameters(self, **kwargs):
        if self.cfg.act_fn.lower() == "jumprelu":
            assert isinstance(self.activation_function, JumpReLU)
            self.activation_function.log_jumprelu_threshold.data.fill_(kwargs["init_log_jumprelu_threshold_value"])

    @overload
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        use_batch_norm_mse: bool = False,
        sparsity_loss_type: Literal["power", "tanh", "tanh-quad", None] = None,
        tanh_stretch_coefficient: float = 4.0,
        p: int = 1,
        l1_coefficient: float = 1.0,
        lp_coefficient: float = 0.0,
        return_aux_data: Literal[True] = True,
        **kwargs,
    ) -> tuple[
        Float[torch.Tensor, " batch"],
        tuple[dict[str, Optional[torch.Tensor]], dict[str, torch.Tensor]],
    ]: ...

    @overload
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        use_batch_norm_mse: bool = False,
        sparsity_loss_type: Literal["power", "tanh", "tanh-quad", None] = None,
        tanh_stretch_coefficient: float = 4.0,
        p: int = 1,
        l1_coefficient: float = 1.0,
        lp_coefficient: float = 0.0,
        return_aux_data: Literal[False],
        **kwargs,
    ) -> Float[torch.Tensor, " batch"]: ...

    @timer.time("compute_loss")
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        label: (
            Union[
                Float[torch.Tensor, "batch d_model"],
                Float[torch.Tensor, "batch seq_len d_model"],
            ]
            | None
        ) = None,
        *,
        use_batch_norm_mse: bool = False,
        sparsity_loss_type: Literal["power", "tanh", "tanh-quad", None] = None,
        tanh_stretch_coefficient: float = 4.0,
        frequency_scale: float = 0.01,
        p: int = 1,
        l1_coefficient: float = 1.0,
        lp_coefficient: float = 0.0,
        return_aux_data: bool = True,
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, " batch"],
        tuple[
            Float[torch.Tensor, " batch"],
            tuple[dict[str, Optional[torch.Tensor]], dict[str, torch.Tensor]],
        ],
    ]:
        """Compute the loss for the autoencoder.
        Ensure that the input activations are normalized by calling `normalize_activations` before calling this method.
        """
        x, encoder_kwargs = self.prepare_input(batch)
        label = self.prepare_label(batch, **kwargs)

        with timer.time("encode"):
            feature_acts, hidden_pre = self.encode(x, return_hidden_pre=True, **encoder_kwargs)
        with timer.time("decode"):
            reconstructed = self.decode(feature_acts, **kwargs)
        
        with timer.time("loss_calculation"):
            l_rec = (reconstructed - label).pow(2)
            if use_batch_norm_mse:
                l_rec = (
                    l_rec
                    / (label - label.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-8).sqrt()
                )
            l_rec = l_rec.sum(dim=-1)
            if isinstance(l_rec, DTensor):
                l_rec = l_rec.full_tensor()
            loss_dict: dict[str, Optional[torch.Tensor]] = {
                "l_rec": l_rec,
            }
            loss = l_rec.mean()

            if sparsity_loss_type is not None:
                with timer.time("sparsity_loss_calculation"):
                    if sparsity_loss_type == "power":
                        l_s = torch.norm(feature_acts * self.decoder_norm(), p=p, dim=-1)
                    elif sparsity_loss_type == "tanh":
                        l_s = torch.tanh(tanh_stretch_coefficient * feature_acts * self.decoder_norm()).sum(dim=-1)
                    elif sparsity_loss_type == "tanh-quad":
                        approx_frequency = einops.reduce(
                            torch.tanh(tanh_stretch_coefficient * feature_acts * self.decoder_norm()),
                            "... d_sae -> d_sae",
                            "mean",
                        )
                        l_s = (approx_frequency * (1 + approx_frequency / frequency_scale)).sum(dim=-1)
                    else:
                        raise ValueError(f"sparsity_loss_type f{sparsity_loss_type} not supported.")
                    if isinstance(l_s, DTensor):
                        l_s = l_s.full_tensor()
                    l_s = l1_coefficient * l_s
                    # WARNING: Some DTensor bugs make if l1_coefficient * l_s goes before full_tensor, the l1_coefficient value will be internally cached. Furthermore, it will cause the backward pass to fail with redistribution error. See https://github.com/pytorch/pytorch/issues/153603 and https://github.com/pytorch/pytorch/issues/153615 .
                    loss_dict["l_s"] = l_s
                    loss = loss + l_s.mean()
            else:
                loss_dict["l_s"] = None

            # Lp loss calculation: λ_P * Σ_i ReLU(exp(t) - f_i(x)) ||W_{d,i}||_2
            if lp_coefficient > 0.0 and isinstance(self.activation_function, JumpReLU):
                with timer.time("lp_loss_calculation"):
                    # ReLU(exp(lp_threshold) - feature_acts) * decoder_norm
                    jumprelu_threshold = self.activation_function.get_jumprelu_threshold()
                    l_p = (
                        torch.nn.functional.relu(jumprelu_threshold - feature_acts) 
                        * self.decoder_norm()
                    )
                    l_p = l_p.sum(dim=-1)
                    if isinstance(l_p, DTensor):
                        l_p = l_p.full_tensor()
                    l_p = lp_coefficient * l_p
                    loss_dict["l_p"] = l_p
                    loss = loss + l_p.mean()
            else:
                loss_dict["l_p"] = None

        if return_aux_data:
            aux_data = {
                "feature_acts": feature_acts,
                "reconstructed": reconstructed,
                "hidden_pre": hidden_pre,
            }
            return loss, (loss_dict, aux_data)
        return loss

    @abstractmethod
    def prepare_input(self, batch: dict[str, torch.Tensor], **kwargs) -> tuple[torch.Tensor, dict[str, Any]]:
        """Prepare the input for the encoder.
        Returns a tuple of (input, kwargs) where kwargs is a dictionary of additional arguments for the encoder computation.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def prepare_label(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Prepare the label for the loss computation."""
        raise NotImplementedError("Subclasses must implement this method")
    
    # @abstractmethod
    # def init_W_D_with_active_subspace(self, activation_batch: dict[str, torch.Tensor], d_active_subspace: int):
    #     """Initialize the W and D parameters with the active subspace."""
    #     raise NotImplementedError("Subclasses must implement this method")

    def dim_maps(self) -> dict[str, DimMap]:
        """Return a dictionary mapping parameter names to dimension maps.

        Returns:
            A dictionary mapping parameter names to DimMap objects.
        """
        if isinstance(self.activation_function, JumpReLU):
            return {f"activation_function.{k}": v for k, v in self.activation_function.dim_maps().items()}
        return {}

    def override_dtypes(self) -> dict[str, torch.dtype]:
        if isinstance(self.activation_function, JumpReLU):
            return {f"activation_function.{k}": v for k, v in self.activation_function.override_dtypes().items()}
        return {}

    def load_distributed_state_dict(
        self, state_dict: dict[str, torch.Tensor], device_mesh: DeviceMesh, prefix: str = ""
    ) -> None:
        self.device_mesh = device_mesh
        if isinstance(self.activation_function, JumpReLU):
            self.activation_function.load_distributed_state_dict(
                state_dict, device_mesh, f"{prefix}activation_function."
            )
