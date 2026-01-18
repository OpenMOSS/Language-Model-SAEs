import json
import os
import re
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from itertools import accumulate
from typing import Any, Callable, Literal, Optional, Union, cast

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from huggingface_hub import hf_hub_download
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map
from transformer_lens import HookedTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BatchFeature,
    Qwen2_5_VLForConditionalGeneration,
)

from lm_saes.config import BaseModelConfig
from lm_saes.utils.auto import PretrainedSAEType, auto_infer_pretrained_sae_type
from lm_saes.utils.distributed import DimMap
from lm_saes.utils.misc import (
    pad_and_truncate_tokens,
)
from lm_saes.utils.timer import timer


def to_tokens(tokenizer, text, max_length, device="cpu", prepend_bos=True):
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"].to(device)
    has_bos_prepended = torch.all(tokens[:, 0] == tokenizer.bos_token_id)
    if prepend_bos and not has_bos_prepended:
        tokens = torch.cat(
            [torch.tensor([tokenizer.bos_token_id]).expand(tokens.shape[0]).unsqueeze(-1).to(device), tokens], dim=1
        )
    elif not prepend_bos and has_bos_prepended:
        tokens = tokens[:, 1:]
    return tokens


def set_tokens(tokenizer, bos_token_id, eos_token_id, pad_token_id):
    if tokenizer.eos_token is None:
        if eos_token_id is None:
            tokenizer.eos_token = "<|endoftext|>"
        else:
            tokenizer.eos_token = tokenizer.decode(eos_token_id)
    if tokenizer.pad_token is None:
        if pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.decode(pad_token_id)
    if tokenizer.bos_token is None:
        if bos_token_id is None:
            tokenizer.bos_token = tokenizer.eos_token
        else:
            tokenizer.bos_token = tokenizer.decode(bos_token_id)
    return tokenizer


def _match_str_tokens_to_input(text: str, str_tokens: list[str]) -> list[Optional[dict[str, Any]]]:
    """Match the tokens to the input text, returning a list of tuples of the form (start_idx, end_idx) for each token."""
    # Initialize list to store token positions
    token_positions = []

    # Keep track of current position in text
    curr_pos = 0

    # For each token, try to find its position in the input text
    for token in str_tokens:
        # Optimization: Check if the token appears immediately at the current position
        if text.startswith(token, curr_pos):
            pos = curr_pos
        else:
            # Search for token in remaining text
            pos = text.find(token, curr_pos)

        if pos != -1:
            # Found a match, store position and update curr_pos
            token_positions.append({"key": "text", "range": (pos, pos + len(token))})
            curr_pos = pos + len(token)
        else:
            # No match found. This is only allowed if the token is a special token
            # that doesn't appear in the input text, or if the token is a subword token
            # which cannot be decoded separately.
            # TODO: Deal with subword tokens properly
            if not ((token.startswith("<") and token.endswith(">")) or "�" in token):
                raise ValueError(f"Token {token} not found in input text `{text}`")
            token_positions.append(None)

    return token_positions


def _get_layer_indices_from_hook_points(hook_points: list[str]) -> list[int]:
    residual_pattern = r"^blocks\.(\d+)\.hook_resid_post$"
    matches = [re.match(residual_pattern, hook_point) for hook_point in hook_points]
    assert all(match is not None for match in matches), "hook_points must be residual stream hook points"
    layer_indices = [int(cast(re.Match[str], match).group(1)) for match in matches]
    return layer_indices


class LanguageModelConfig(BaseModelConfig):
    model_name: str = "gpt2"
    """ The name of the model to use. """
    model_from_pretrained_path: str | None = None
    """ The path to the pretrained model. If `None`, will use the model from HuggingFace. """
    use_flash_attn: bool = False
    """ Whether to use Flash Attention. """
    cache_dir: str | None = None
    """ The directory of the HuggingFace cache. Should have the same effect as `HF_HOME`. """
    local_files_only: bool = False
    """ Whether to only load the model from the local files. Should have the same effect as `HF_HUB_OFFLINE=1`. """
    max_length: int = 2048
    """ The maximum length of the input. """
    backend: Literal["huggingface", "transformer_lens", "auto"] = "auto"
    """ The backend to use for the language model. """
    load_ckpt: bool = True
    tokenizer_only: bool = False
    """ Whether to only load the tokenizer. """
    prepend_bos: bool = True
    """ Whether to prepend the BOS token to the input. """
    bos_token_id: int | None = None
    """ The ID of the BOS token. If `None`, will use the default BOS token. """
    eos_token_id: int | None = None
    """ The ID of the EOS token. If `None`, will use the default EOS token. """
    pad_token_id: int | None = None
    """ The ID of the padding token. If `None`, will use the default padding token. """

    @staticmethod
    def from_pretrained_sae(pretrained_name_or_path: str, **kwargs):
        """Load the LanguageModelConfig from a pretrained SAE name or path. Config is read from <pretrained_name_or_path>/lm_config.json (for local storage), <repo_id>/<name>/lm_config.json (for HuggingFace Hub), or constructed from model name (for SAELens).

        Args:
            pretrained_name_or_path (str): The path to the pretrained SAE.
            **kwargs: Additional keyword arguments to pass to the LanguageModelConfig constructor.
        """
        sae_type = auto_infer_pretrained_sae_type(pretrained_name_or_path.split(":")[0])
        if sae_type == PretrainedSAEType.LOCAL:
            path = os.path.join(os.path.dirname(pretrained_name_or_path), "lm_config.json")
        elif sae_type == PretrainedSAEType.HUGGINGFACE:
            repo_id, name = pretrained_name_or_path.split(":")
            path = hf_hub_download(repo_id=repo_id, filename=f"{name}/lm_config.json")
        elif sae_type == PretrainedSAEType.SAELENS:
            from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

            repo_id, name = pretrained_name_or_path.split(":")
            lookups = get_pretrained_saes_directory()
            assert lookups.get(repo_id) is not None and lookups[repo_id].saes_map.get(name) is not None, (
                f"Pretrained SAE {pretrained_name_or_path} not found in SAELens. This might indicate bugs in `auto_infer_pretrained_sae_type`."
            )
            model_name = lookups[repo_id].model
            return LanguageModelConfig(model_name=model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported pretrained type: {sae_type}")
        with open(os.path.join(path, "lm_config.json"), "r") as f:
            lm_config = json.load(f)
        return LanguageModelConfig.model_validate(lm_config, **kwargs)

    def save_lm_config(self, sae_path: str):
        assert os.path.exists(sae_path), f"{sae_path} does not exist. Unable to save LanguageModelConfig."

        d = self.model_dump()
        with open(os.path.join(sae_path, "lm_config.json"), "w") as f:
            json.dump(d, f, indent=4)


class LLaDAConfig(LanguageModelConfig):
    mask_ratio: float = 0.0
    mdm_mask_token_id: int = 126336
    prepend_bos: bool = False
    calculate_logits: bool = False


class LanguageModel(ABC):
    @abstractmethod
    def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
        """Trace how raw data is eventually aligned with tokens.

        Args:
            raw (dict[str, Any]): The raw data to trace.

        Returns:
            list[list[Any]]: The origins of the tokens in the raw data. Shape: (batch_size, n_tokens)
        """
        pass

    @abstractmethod
    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        """Convert raw data to activations.

        Args:
            raw (dict[str, Any]): The raw data to convert to activations.
            hook_points (list[str]): The hook points to use for activations.

        Returns:
            dict[str, torch.Tensor]: The activations. Shape: (batch_size, n_tokens, n_activations)
        """
        pass

    @abstractmethod
    def preprocess_raw_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Preprocess the raw data.

        Args:
            raw (dict[str, Any]): The raw data to preprocess.

        Returns:
            dict[str, Any]: The preprocessed raw data.
        """
        pass

    @property
    @abstractmethod
    def eos_token_id(self) -> int | None:
        """The ID of the end-of-sequence token."""
        pass

    @property
    @abstractmethod
    def bos_token_id(self) -> int | None:
        """The ID of the beginning-of-sequence token."""
        pass

    @property
    @abstractmethod
    def pad_token_id(self) -> int | None:
        """The ID of the padding token."""
        pass


class TransformerLensLanguageModel(LanguageModel):
    def __init__(self, cfg: LanguageModelConfig, device_mesh: DeviceMesh | None = None):
        self.cfg = cfg
        self.device_mesh = device_mesh
        if cfg.device == "cuda":
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        elif cfg.device == "npu":
            self.device = torch.device(f"npu:{torch.npu.current_device()}")  # type: ignore[reportAttributeAccessIssue]
        else:
            self.device = torch.device(cfg.device)

        hf_model = (
            AutoModelForCausalLM.from_pretrained(
                (cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path),
                cache_dir=cfg.cache_dir,
                local_files_only=cfg.local_files_only,
                dtype=cfg.dtype,
                trust_remote_code=True,
            )
            if cfg.load_ckpt and not cfg.tokenizer_only
            else None
        )
        hf_tokenizer = AutoTokenizer.from_pretrained(
            (cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path),
            cache_dir=cfg.cache_dir,
            trust_remote_code=True,
            use_fast=True,
            add_bos_token=True,
            local_files_only=cfg.local_files_only,
        )
        self.tokenizer = set_tokens(
            hf_tokenizer,
            cfg.bos_token_id,
            cfg.eos_token_id,
            cfg.pad_token_id,
        )
        self.model = (
            HookedTransformer.from_pretrained_no_processing(
                cfg.model_name,
                use_flash_attn=cfg.use_flash_attn,
                device=self.device,
                cache_dir=cfg.cache_dir,
                hf_model=hf_model,
                hf_config=hf_model.config,
                tokenizer=hf_tokenizer,
                dtype=cfg.dtype,  # type: ignore ; issue with transformer_lens
            )
            if hf_model and not cfg.tokenizer_only
            else None
        )

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        return self.tokenizer.bos_token_id

    @property
    def pad_token_id(self) -> int | None:
        return self.tokenizer.pad_token_id

    def preprocess_raw_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw

    def forward(self, *args, **kwargs):
        assert self.model is not None, "model must be initialized"
        # Collect all tensor arguments
        tensors = [arg for arg in args if isinstance(arg, torch.Tensor)] + [
            v for v in kwargs.values() if isinstance(v, torch.Tensor)
        ]
        # Check if all tensors are DTensors
        is_distributed = len(tensors) > 0 and all(isinstance(t, DTensor) for t in tensors)
        if self.device_mesh is not None:
            assert is_distributed, "All tensor inputs must be DTensor when device_mesh is not None"
            return local_map(
                self.model.forward,
                out_placements=DimMap({"data": 0}).placements(self.device_mesh),
            )(*args, prepend_bos=self.cfg.prepend_bos, **kwargs)  # type: ignore
        else:
            assert not is_distributed, "Input should not contain DTensor when device_mesh is None"
            return self.model.forward(*args, prepend_bos=self.cfg.prepend_bos, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _to_tensor(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, DTensor):
            assert input.placements == tuple(DimMap({"data": 0}).placements(cast(DeviceMesh, self.device_mesh)))
            return input.to_local()
        else:
            return input

    def _to_dtensor(self, input: torch.Tensor) -> torch.Tensor:
        return (
            DTensor.from_local(
                input,
                device_mesh=self.device_mesh,
                placements=DimMap({"data": 0}).placements(cast(DeviceMesh, self.device_mesh)),
            )
            if isinstance(input, torch.Tensor)
            else input
        )

    def _wrap_hook_for_local(self, hook_fn):
        def wrapped_hook_fn(*args, **kwargs):
            args = pytree.tree_map(self._to_dtensor, args)
            kwargs = pytree.tree_map(self._to_dtensor, kwargs)
            return pytree.tree_map(self._to_tensor, hook_fn(*args, **kwargs))

        return wrapped_hook_fn

    def run_with_hooks(
        self,
        *args,
        fwd_hooks: list[tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: list[tuple[Union[str, Callable], Callable]] = [],
        **kwargs,
    ) -> Any:
        assert self.model is not None, "model must be initialized"

        if self.device_mesh is None:
            return self.model.run_with_hooks(*args, fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks, **kwargs)

        wrapped_fwd_hooks = [(name, self._wrap_hook_for_local(hook)) for name, hook in fwd_hooks]
        wrapped_bwd_hooks = [(name, self._wrap_hook_for_local(hook)) for name, hook in bwd_hooks]

        return self.model.run_with_hooks(*args, fwd_hooks=wrapped_fwd_hooks, bwd_hooks=wrapped_bwd_hooks, **kwargs)

    def run_with_cache(self, *args, **kwargs) -> Any:
        assert self.model is not None, "model must be initialized"
        if self.device_mesh is None:
            return self.model.run_with_cache(*args, **kwargs)

        args = pytree.tree_map(self._to_tensor, args)
        kwargs = pytree.tree_map(self._to_tensor, kwargs)
        return pytree.tree_map(self._to_dtensor, self.model.run_with_cache(*args, **kwargs))

    def run_with_cache_until(self, *args, **kwargs) -> Any:
        assert self.model is not None, "model must be initialized"
        if self.device_mesh is None:
            return self.model.run_with_cache_until(*args, **kwargs)

        args = pytree.tree_map(self._to_tensor, args)
        kwargs = pytree.tree_map(self._to_tensor, kwargs)
        return pytree.tree_map(self._to_dtensor, self.model.run_with_cache_until(*args, **kwargs))

    @contextmanager
    def hooks(
        self,
        fwd_hooks: list[tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: list[tuple[Union[str, Callable], Callable]] = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
    ):
        assert self.model is not None, "model must be initialized"
        wrapped_fwd_hooks = fwd_hooks
        wrapped_bwd_hooks = bwd_hooks
        if self.device_mesh is not None:
            wrapped_fwd_hooks = [(name, self._wrap_hook_for_local(hook)) for name, hook in fwd_hooks]
            wrapped_bwd_hooks = [(name, self._wrap_hook_for_local(hook)) for name, hook in bwd_hooks]

        with self.model.hooks(
            fwd_hooks=wrapped_fwd_hooks,
            bwd_hooks=wrapped_bwd_hooks,
            reset_hooks_end=reset_hooks_end,
            clear_contexts=clear_contexts,
        ):
            yield self

    def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
        if any(key in ["images", "videos"] for key in raw):
            warnings.warn(
                "Tracing with modalities other than text is not implemented for TransformerLensLanguageModel. Only text fields will be used."
            )
        encoding = self.tokenizer(
            raw["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_offsets_mapping=True,
        )
        offsets = encoding["offset_mapping"].tolist()
        tokens = encoding["input_ids"]
        has_bos_prepended = torch.all(tokens[:, 0] == self.bos_token_id)
        if self.cfg.prepend_bos and not has_bos_prepended:
            offsets = [[None] + offset_ for offset_ in offsets]
        elif not self.cfg.prepend_bos and has_bos_prepended:
            offsets = [offset_[1:] for offset_ in offsets]
        if n_context is not None:
            offsets = [offset_[:n_context] for offset_ in offsets]
            offsets = [offset_ + [None] * (n_context - len(offset_)) for offset_ in offsets]
        return [
            [{"key": "text", "range": offset} if offset is not None else None for offset in offset_]
            for offset_ in offsets
        ]

    @timer.time("to_activations")
    @torch.no_grad()
    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        assert self.model is not None
        if any(key in ["images", "videos"] for key in raw):
            warnings.warn(
                "Activations with modalities other than text is not implemented for TransformerLensLanguageModel. Only text fields will be used."
            )
        tokens = to_tokens(
            self.tokenizer,
            raw["text"],
            max_length=self.cfg.max_length,
            device=self.cfg.device,
            prepend_bos=self.cfg.prepend_bos,
        )
        if self.device_mesh is not None and n_context is None:
            num_token_list = [None for _ in range(dist.get_world_size(group=self.device_mesh.get_group("data")))]
            dist.all_gather_object(num_token_list, tokens.shape[1], group=self.device_mesh.get_group("data"))
            n_context = max(cast(list[int], num_token_list))
        if n_context is not None:
            assert self.pad_token_id is not None, (
                "Pad token ID must be set for TransformerLensLanguageModel when n_context is provided"
            )
            tokens = pad_and_truncate_tokens(tokens, n_context, pad_token_id=self.pad_token_id)
        tokens = tokens.contiguous()
        with timer.time("run_with_cache_until"):
            # _, activations = self.model.run_with_cache_until(tokens, names_filter=hook_points, use_flash_attn=False)
            _, activations = self.model.run_with_cache_until(tokens, names_filter=hook_points)

        # we do not want to filter out eos. It might be end of chats and include useful information
        mask = torch.isin(
            tokens,
            torch.tensor([self.pad_token_id, self.bos_token_id]).to(tokens.device),
            invert=True,
        ).int()
        attention_mask = torch.isin(tokens, torch.tensor([self.pad_token_id]).to(tokens.device), invert=True).int()

        return {hook_point: activations[hook_point] for hook_point in hook_points} | {
            "tokens": tokens
            if self.device_mesh is None
            else DTensor.from_local(
                tokens, device_mesh=self.device_mesh, placements=DimMap({"data": 0}).placements(self.device_mesh)
            ),
            "mask": mask
            if self.device_mesh is None
            else DTensor.from_local(
                mask, device_mesh=self.device_mesh, placements=DimMap({"data": 0}).placements(self.device_mesh)
            ),
            "attention_mask": attention_mask
            if self.device_mesh is None
            else DTensor.from_local(
                attention_mask,
                device_mesh=self.device_mesh,
                placements=DimMap({"data": 0}).placements(self.device_mesh),
            ),
        }


class HuggingFaceLanguageModel(LanguageModel):
    def __init__(self, cfg: LanguageModelConfig):
        self.cfg = cfg
        if cfg.device == "cuda":
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        elif cfg.device == "npu":
            self.device = torch.device(f"npu:{torch.npu.current_device()}")  # type: ignore[reportAttributeAccessIssue]
        else:
            self.device = torch.device(cfg.device)

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path,
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
            dtype=cfg.dtype,
            trust_remote_code=True,
        ).to(self.device)  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path,
            cache_dir=cfg.cache_dir,
            trust_remote_code=True,
            use_fast=True,
            add_bos_token=True,
            local_files_only=cfg.local_files_only,
        )
        self.model.eval()

    def preprocess_raw_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        return self.tokenizer.bos_token_id  # should be None

    @property
    def pad_token_id(self) -> int | None:
        return self.tokenizer.pad_token_id

    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        layer_indices = _get_layer_indices_from_hook_points(hook_points)
        tokens = to_tokens(
            self.tokenizer,
            raw["text"],
            max_length=self.cfg.max_length,
            device=self.cfg.device,
            prepend_bos=self.cfg.prepend_bos,
        )
        if n_context is not None:
            assert self.pad_token_id is not None, "Pad token ID must be set when n_context is provided"
            tokens = pad_and_truncate_tokens(tokens, n_context, pad_token_id=self.pad_token_id)
        outputs = self.model(tokens, output_hidden_states=True)
        activations = {
            hook_points[i]: outputs.hidden_states[layer_index + 1] for i, layer_index in enumerate(layer_indices)
        }
        activations["tokens"] = tokens
        return activations

    def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
        encoding = self.tokenizer(
            raw["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_offsets_mapping=True,
        )
        offsets = encoding["offset_mapping"]
        tokens = encoding["input_ids"]
        has_bos_prepended = torch.all(tokens[:, 0] == self.bos_token_id)
        if self.cfg.prepend_bos and not has_bos_prepended:
            offsets = [[None] + offset_ for offset_ in offsets]
        elif not self.cfg.prepend_bos and has_bos_prepended:
            offsets = [offset_[1:] for offset_ in offsets]
        if n_context is not None:
            offsets = [offset_[:n_context] for offset_ in offsets]
            offsets = [offset_ + [None] * (n_context - len(offset_)) for offset_ in offsets]
        return [[{"key": "text", "range": offset} for offset in offset_] for offset_ in offsets]


class LLaDALanguageModel(TransformerLensLanguageModel):
    cfg: LLaDAConfig  # Explicitly specify the type to avoid linter errors

    @property
    def mdm_mask_token_id(self) -> int:
        return self.cfg.mdm_mask_token_id

    @torch.no_grad()
    def _get_masked_tokens(self, tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply random masking to non-pad tokens based on mask_ratio. The random seed should be determined by the raw data.
        Args:
            tokens (torch.Tensor): The tokens to mask.
            pad_mask (torch.Tensor): The mask of the padding tokens.

        Returns:
            torch.Tensor: The tokens after masking. Pad tokens are included.
        """
        if self.cfg.mask_ratio <= 0:
            return tokens

        # Calculate random seeds based on non-pad tokens
        random_seeds = (tokens * ~pad_mask).sum(dim=1).tolist()
        masked_tokens = tokens.clone()

        # For each sequence in the batch
        for i, seed in enumerate(random_seeds):
            # Create random generator with deterministic seed
            random_generator = torch.Generator(device=tokens.device)
            random_generator.manual_seed(seed)

            # Generate random values for all positions
            random_values = torch.rand(tokens[i].shape, device=tokens[i].device, generator=random_generator)

            # Count non-pad tokens
            non_pad_mask = ~pad_mask[i]
            non_pad_count = non_pad_mask.sum()

            # Calculate number of tokens to mask
            num_to_mask = int(non_pad_count.item() * self.cfg.mask_ratio)
            if num_to_mask == 0:
                continue

            # Set random values for pad positions to infinity so they won't be selected
            random_values = torch.where(non_pad_mask, random_values, torch.full_like(random_values, float("inf")))

            # Get indices of positions to mask (lowest random values)
            mask_indices = torch.topk(random_values, k=num_to_mask, largest=False).indices

            # Apply masking
            masked_tokens[i][mask_indices] = self.mdm_mask_token_id

        return masked_tokens

    @torch.no_grad()
    def preprocess_raw_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        assert self.model is not None
        tokens = to_tokens(
            self.tokenizer,
            raw["text"],
            max_length=self.cfg.max_length,
            device=self.cfg.device,
            prepend_bos=self.cfg.prepend_bos,
        )
        pad_mask = tokens == self.pad_token_id
        masked_tokens = self._get_masked_tokens(tokens, pad_mask)
        raw["text"] = self.tokenizer.batch_decode(masked_tokens)
        pad_token = self.tokenizer.pad_token
        raw["text"] = [text.replace(pad_token, "") for text in raw["text"]]
        raw_meta = raw.get("meta", [])
        raw["meta"] = [
            (raw_meta[i] if i < len(raw_meta) else {}) | {"mask_ratio": self.cfg.mask_ratio}
            for i in range(len(raw["text"]))
        ]
        return raw

    @torch.no_grad()
    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        assert self.model is not None
        tokens = to_tokens(
            self.tokenizer,
            raw["text"],
            max_length=self.cfg.max_length,
            device=self.cfg.device,
            prepend_bos=self.cfg.prepend_bos,
        )
        if n_context is not None:
            assert self.pad_token_id is not None, (
                "Pad token ID must be set for LLaDALanguageModel when n_context is provided"
            )
            tokens = pad_and_truncate_tokens(tokens, n_context, pad_token_id=self.pad_token_id)
        pad_mask = tokens == self.pad_token_id
        predicted_tokens = None
        if self.cfg.calculate_logits:
            logits, activations = self.model.run_with_cache(tokens, names_filter=hook_points)
            assert isinstance(logits, torch.Tensor)
            predicted_token_ids = logits.argmax(dim=-1)
            predicted_tokens = torch.where(pad_mask, -1, predicted_token_ids)
        else:
            _, activations = self.model.run_with_cache_until(tokens, names_filter=hook_points)
        predicted_tokens = {"predicted_tokens": predicted_tokens} if predicted_tokens is not None else {}
        return (
            {hook_point: activations[hook_point] for hook_point in hook_points} | {"tokens": tokens} | predicted_tokens
        )


class QwenVLLanguageModel(HuggingFaceLanguageModel):
    def __init__(self, cfg: LanguageModelConfig):
        super().__init__(cfg)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.model_name,
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
            dtype=cfg.dtype,
            attn_implementation="flash_attention_2" if cfg.use_flash_attn else None,
        ).to(self.device)  # type: ignore

        self.processor = AutoProcessor.from_pretrained(
            cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path,
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
            padding_side="left",
            max_pixels=1800 * 28 * 28,
        )
        self.tokenizer = self.processor.tokenizer
        self.model.eval()

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        return self.tokenizer.bos_token_id  # should be None

    @property
    def pad_token_id(self) -> int | None:
        return self.tokenizer.pad_token_id

    def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
        assert self.tokenizer is not None, "tokenizer must be initialized"
        assert self.processor is not None, "processor must be initialized"
        inputs, _ = self.process_raw_data(raw)
        input_ids = inputs["input_ids"]
        if n_context is not None:
            assert self.pad_token_id is not None, (
                "Pad token ID must be set for QwenVLLanguageModel when n_context is provided"
            )
            input_ids = pad_and_truncate_tokens(input_ids, n_context, pad_token_id=self.pad_token_id)
        batch_str_tokens: list[list[str]] = [
            self.tokenizer.batch_decode(input_id, clean_up_tokenization_spaces=False) for input_id in input_ids
        ]

        def split_number(n: int, m: float) -> list[int]:
            # split n into m parts, the parts are as even as possible
            assert m.is_integer()
            quotient = n // int(m)
            remainder = n % int(m)
            return [quotient + 1 if i < remainder else quotient for i in range(int(m))]

        batch_token_positions: list[list[Any]] = [
            _match_str_tokens_to_input(text, str_tokens) for (text, str_tokens) in zip(raw["text"], batch_str_tokens)
        ]

        if "images" in raw and raw["images"] is not None:
            assert "image_grid_thw" in inputs
            assert "pixel_values" in inputs
            resized_shape_list = (inputs["image_grid_thw"][:, 1:] * 14).tolist()
            for str_tokens, images in zip(batch_str_tokens, raw["images"]):
                # str_tokens: list[str], tokens for each text in the batch
                # images: list[torch.Tensor], images for each text in the batch
                # resized_shape: [total_image_number, 2]
                # token_positions: list[Any], positions of the tokens in the input text
                start_id_list = [id for id, str_token in enumerate(str_tokens) if str_token == "<|vision_start|>"]
                end_id_list = [id for id, str_token in enumerate(str_tokens) if str_token == "<|vision_end|>"]
                assert len(start_id_list) == len(end_id_list)
                images_num = len(start_id_list)  # number of images in this text
                resized_shapes = resized_shape_list[:images_num]  # get the resized shapes for images in this text
                resized_shape_list = resized_shape_list[images_num:]

                for i, (start_id, end_id, image, resized_shape) in enumerate(
                    zip(start_id_list, end_id_list, images, resized_shapes)
                ):
                    resized_height, resized_width = int(resized_shape[0]), int(resized_shape[1])
                    original_height, original_width = image.shape[-2], image.shape[-1]
                    image_token_num = end_id - start_id - 1
                    assert image_token_num == resized_height * resized_width / 14 / 14 / 4

                    split_height = split_number(original_height, resized_height / 28)
                    split_width = split_number(original_width, resized_width / 28)

                    prefix_sum_height = [0] + list(accumulate(split_height))
                    prefix_sum_width = [0] + list(accumulate(split_width))

                    prefix_sum_height = [i / original_height for i in prefix_sum_height]
                    prefix_sum_width = [i / original_width for i in prefix_sum_width]
                    grid_coords = [
                        (
                            id // (resized_width // 28),
                            id % (resized_width // 28),
                        )
                        for id in range(image_token_num)
                    ]
                    original_coords = [
                        (
                            prefix_sum_width[grid_coords[id][1]],
                            prefix_sum_height[grid_coords[id][0]],
                            prefix_sum_width[grid_coords[id][1] + 1],
                            prefix_sum_height[grid_coords[id][0] + 1],
                        )
                        for id in range(image_token_num)
                    ]

                    batch_token_positions[i][start_id + 1 : end_id] = original_coords
                    for j in range(len(original_coords)):
                        batch_token_positions[i][start_id + 1 + j] = {
                            "key": "image",
                            "rect": original_coords[j],
                            "image_index": i,
                        }
        return batch_token_positions

    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        layer_indices = _get_layer_indices_from_hook_points(hook_points)
        inputs = self.process_raw_data(raw)[0].to(self.device)
        if n_context is not None:
            assert self.pad_token_id is not None, (
                "Pad token ID must be set for QwenVLLanguageModel when n_context is provided"
            )
            inputs["input_ids"] = pad_and_truncate_tokens(
                inputs["input_ids"], n_context, pad_token_id=self.pad_token_id
            )
        outputs = self.model(**inputs, output_hidden_states=True)
        activations = {
            hook_points[i]: outputs.hidden_states[layer_index + 1] for i, layer_index in enumerate(layer_indices)
        }
        activations["tokens"] = inputs["input_ids"]
        return activations

    def process_raw_data(
        self, raw: dict[str, Any], padding: str | bool = "max_length"
    ) -> tuple[BatchFeature, dict[str, Any]]:
        # raw is a dict with keys "text", "images", "videos", etc.
        IMAGE_PAD_TOKEN: str = raw.get("image_pad_token", "<image>")
        inputs = cast(BatchFeature, {})
        processed_raw = {}

        # use process_vision_info to resize images
        assert self.processor is not None, "processor must be initialized"
        if "images" in raw:
            processed_raw["images"] = raw["images"]

        processed_raw["text"] = [
            text.replace(IMAGE_PAD_TOKEN, f"<|vision_start|>{self.processor.image_token}<|vision_end|>")
            for text in raw["text"]
        ]
        inputs: BatchFeature = self.processor(
            text=processed_raw["text"],
            images=raw.get("images", None),
            return_tensors="pt",
            max_length=self.cfg.max_length,
            padding=padding,
            truncation=True,
        )

        return inputs, processed_raw
