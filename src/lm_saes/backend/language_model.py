from __future__ import annotations

import json
import os
import re
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from itertools import accumulate
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union, cast

import einops
import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from huggingface_hub import hf_hub_download
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BatchFeature,
    Qwen2_5_VLForConditionalGeneration,
)

from lm_saes.backend.run_with_cache_until import run_with_cache_until
from lm_saes.config import BaseModelConfig
from lm_saes.utils.auto import PretrainedSAEType, auto_infer_pretrained_sae_type
from lm_saes.utils.distributed import DimMap
from lm_saes.utils.misc import ensure_tokenized, item, pad_and_truncate_tokens
from lm_saes.utils.timer import timer

if TYPE_CHECKING:
    from lm_saes.lorsa import LowRankSparseAttention
    from lm_saes.sae import SparseDictionary


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
            **kwargs (Any): Additional keyword arguments to pass to the LanguageModelConfig constructor.
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


def hook_in_fn_builder(replacement_module: SparseDictionary) -> Callable:
    from lm_saes.lorsa import LowRankSparseAttention
    from lm_saes.sae import SparseAutoEncoder

    if isinstance(replacement_module, SparseAutoEncoder):

        def hook_in_fn(
            x: torch.Tensor,
            hook: HookPoint,
            replacement_module: SparseDictionary,
            cache_activations: dict[str, torch.Tensor],
        ):
            assert hook.name in replacement_module.cfg.hooks_in, "Hook point must be in hook points in"
            cache_activations[hook.name + ".x"] = x
            cache_activations[hook.name + ".feature_acts.up"] = replacement_module.encode(x)
            cache_activations[hook.name + ".feature_acts.down"] = (
                cache_activations[hook.name + ".feature_acts.up"].detach().requires_grad_(True)
            )
            return x.detach()

        return hook_in_fn

    elif isinstance(replacement_module, LowRankSparseAttention):

        def hook_in_fn(
            x: torch.Tensor,
            hook: HookPoint,
            replacement_module: LowRankSparseAttention,
            cache_activations: dict[str, torch.Tensor],
        ):
            assert hook.name in replacement_module.cfg.hooks_in, "Hook point must be in hook points in"
            cache_activations[hook.name + ".x"] = x
            encode_result = replacement_module.encode(
                x,
                return_hidden_pre=False,
                return_attention_pattern=True,
                return_attention_score=True,
            )
            cache_activations[hook.name + ".attn_pattern"] = encode_result[1].detach()
            cache_activations[hook.name + ".feature_acts.up"] = encode_result[0]  # batch, seq_len, d_sae
            cache_activations[hook.name + ".feature_acts.down"] = (
                cache_activations[hook.name + ".feature_acts.up"].detach().requires_grad_(True)
            )

            return x.detach()

        return hook_in_fn

    else:
        # TODO: handle other replacement modules such as CLT
        raise ValueError(f"Unsupported replacement module type: {type(replacement_module)}")


def hook_out_fn_builder(replacement_module: SparseDictionary) -> Callable:
    def hook_out_fn(
        x: torch.Tensor,
        hook: HookPoint,
        replacement_module: SparseDictionary,
        cache_activations: dict[str, torch.Tensor],
        update_error_cache: bool = False,
    ):
        assert hook.name in replacement_module.cfg.hooks_out, "Hook point must be in hook points out"
        hook_in = replacement_module.cfg.hooks_in[0]  # TODO: handle multiple hook points in for CLT
        reconstructed = replacement_module.decode(cache_activations[hook_in + ".feature_acts.down"])
        cache_activations[hook.name + ".reconstructed"] = reconstructed
        assert hook.name + ".error" in cache_activations or update_error_cache, (
            "There must be an error cache for the hook point"
        )
        if update_error_cache:
            error = (x - reconstructed).detach().requires_grad_(True)
            cache_activations[hook.name + ".error"] = error
        else:
            error = cache_activations[hook.name + ".error"]

        return reconstructed + error

    return hook_out_fn


class EdgeMatrix(torch.Tensor):
    """A Tensor wrapper that preserves ``cache_activations`` through operations.

    Args:
        elem: The underlying tensor data.
        cache_activations: Cached activation tensors keyed by hook point name.
    """

    matrix: torch.Tensor
    cache_activations: dict[str, torch.Tensor]
    n_tokens: int
    n_logits: int
    n_active_features: int
    n_error: int
    max_features: int

    _wrap_enabled: bool = True

    @classmethod
    @contextmanager
    def no_wrap(cls):
        """Context manager that disables automatic wrapping in ``__torch_dispatch__``.

        Any torch operation on an ``EdgeMatrix`` inside this context will return
        a plain ``torch.Tensor`` instead of being re-wrapped.
        """
        prev = cls._wrap_enabled
        cls._wrap_enabled = False
        try:
            yield
        finally:
            cls._wrap_enabled = prev

    @classmethod
    def _wrap(
        cls,
        matrix: torch.Tensor,
        cache_activations: dict[str, torch.Tensor],
        n_tokens: int,
        n_logits: int,
        n_active_features: int,
        n_error: int,
        max_features: int | None = None,
    ) -> "EdgeMatrix":
        """Low-level constructor that assembles an EdgeMatrix from pre-computed parts."""
        obj = torch.Tensor._make_wrapper_subclass(
            cls,
            size=matrix.shape,
            dtype=matrix.dtype,
            layout=matrix.layout,
            device=matrix.device,
            requires_grad=matrix.requires_grad,
        )
        obj.matrix = matrix
        obj.cache_activations = cache_activations
        obj.n_tokens = n_tokens
        obj.n_logits = n_logits
        obj.n_active_features = n_active_features
        obj.n_error = n_error
        obj.max_features = max_features or n_active_features
        return obj

    @staticmethod
    def __new__(cls, cache_activations: dict[str, torch.Tensor], max_features: int | None = None) -> "EdgeMatrix":
        n_tokens = cache_activations["hook_embed"].shape[1]
        n_logits = cache_activations["logits"].shape[-1]
        n_active_features = int(
            item(
                cast(
                    torch.Tensor,
                    sum([v[0].gt(0).sum() for k, v in cache_activations.items() if k.endswith(".feature_acts.up")]),
                )
            )
        )
        max_features = min(max_features or n_active_features, n_active_features)
        n_error = n_tokens * len([v for k, v in cache_activations.items() if k.endswith(".error")])
        matrix = torch.zeros(max_features + n_logits, n_tokens + n_active_features + n_error)
        return cls._wrap(matrix, cache_activations, n_tokens, n_logits, n_active_features, n_error, max_features)

    @classmethod
    def __torch_dispatch__(cls, func: Any, types: list[type], args: Any = (), kwargs: Any = None) -> Any:
        from torch.utils._pytree import tree_map

        if kwargs is None:
            kwargs = {}

        source: EdgeMatrix | None = None
        for a in tree_map(lambda x: x, args):
            if isinstance(a, EdgeMatrix):
                source = a
                break

        def unwrap(x: Any) -> Any:
            return x.matrix if isinstance(x, EdgeMatrix) else x

        out = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        if not cls._wrap_enabled:
            return out

        def wrap(x: Any) -> Any:
            if isinstance(x, torch.Tensor) and not isinstance(x, EdgeMatrix) and source is not None:
                return cls._wrap(
                    x,
                    source.cache_activations,
                    source.n_tokens,
                    source.n_logits,
                    source.n_active_features,
                    source.n_error,
                    source.max_features,
                )
            return x

        return tree_map(wrap, out)

    def __init__(self, cache_activations: dict[str, torch.Tensor], max_features: int | None = None):
        self.row_info: list[tuple[int, str, int]] = self._build_row_info()
        self.column_info: list[tuple[int, str, int | str]] = self._build_column_info()
        self.row2column = [-1] * self.n_logits

    def _build_column_info(self) -> list[tuple[int, str, int | str]]:
        """Build column metadata for the edge matrix.

        Each entry is ``(position, hook_name, feature_id)`` where *feature_id*
        is ``str("embed")`` for embed and ``str("error")`` for error columns.

        Returns:
            A list of ``(pos, hook_name, feature_id)`` tuples, one per column,
            ordered as: embed tokens, active features, error terms.
        """
        result: list[tuple[int, str, int | str]] = []
        if "hook_embed" in self.cache_activations:
            n_pos = self.cache_activations["hook_embed"].shape[1]
            result.extend([(p, "hook_embed", "embed") for p in range(n_pos)])
        for key in self.cache_activations:
            if key.endswith(".feature_acts.down"):
                active_mask = self.cache_activations[key][0] > 0  # [n_pos, n_features]
                pos_ids, feat_ids = active_mask.nonzero(as_tuple=True)
                result.extend(
                    [
                        (p, key.removesuffix(".feature_acts.down"), i)
                        for p, i in zip(pos_ids.tolist(), feat_ids.tolist())
                    ]
                )
        for key in self.cache_activations:
            if key.endswith(".error"):
                n_pos = self.cache_activations[key].shape[1]
                result.extend([(p, key, "error") for p in range(n_pos)])
        return result

    def _build_row_info(self) -> list[tuple[int, str, int]]:
        return [(self.n_tokens, "logits", i) for i in range(self.n_logits)]

    def get_row_values(self, rows: int | list[int], feature_type: Literal["up", "down"] = "up") -> list[torch.Tensor]:
        """Return activation values for the given row(s)."""
        rows = [rows] if isinstance(rows, int) else rows
        values: list[torch.Tensor] = []
        for idx, row_idx in enumerate(rows):
            row_info = self.row_info[row_idx]
            if row_info[1] == "logits":
                values.append(self.cache_activations["logits"][idx, -1, row_info[2]])
            else:
                values.append(
                    self.cache_activations[row_info[1] + f".feature_acts.{feature_type}"][idx, row_info[0], row_info[2]]
                )
        return values

    def get_column_values(
        self, columns: int | list[int], feature_type: Literal["up", "down"] = "down"
    ) -> list[torch.Tensor]:
        columns = [columns] if isinstance(columns, int) else columns
        values = []
        for idx, column_idx in enumerate(columns):
            column_info = self.column_info[column_idx]
            if isinstance(column_info[2], str):
                values.append(self.cache_activations[column_info[1]][idx, column_info[0]])
            else:
                values.append(
                    self.cache_activations[column_info[1] + f".feature_acts.{feature_type}"][
                        idx, column_info[0], column_info[2]
                    ]
                )
        return values

    def update_row_info_from_columns(self, column_ids: int | list[int]) -> None:
        column_ids = [column_ids] if isinstance(column_ids, int) else column_ids
        self.row_info += [cast(tuple[int, str, int], self.column_info[column_id]) for column_id in column_ids]
        self.row2column += column_ids


def _get_cache_activations(cache_activations: dict[str, torch.Tensor], suffix: str) -> list[torch.Tensor]:
    return [v for k, v in cache_activations.items() if k.endswith(suffix)]


def _clear_cache_activation_grads(cache_activations: dict[str, torch.Tensor]) -> None:
    """Clear gradients for all cached activations in-place."""
    for activation in cache_activations.values():
        if activation.grad is not None:
            activation.grad = None


def update_edge_matrix_rows(edge_matrix: EdgeMatrix) -> None:
    cur_batch_size = edge_matrix.shape[0]
    column_start = 0
    assert edge_matrix.cache_activations["hook_embed"].grad is not None, "hook_embed must have a gradient"
    edge_matrix[:, column_start : column_start + edge_matrix.n_tokens] = (
        einops.einsum(
            edge_matrix.cache_activations["hook_embed"][:cur_batch_size],
            edge_matrix.cache_activations["hook_embed"].grad[:cur_batch_size],
            "b pos d_model, b pos d_model -> b pos",
        )
        .detach()
        .cpu()
    )
    column_start += edge_matrix.n_tokens

    for feature_acts_down in _get_cache_activations(edge_matrix.cache_activations, ".feature_acts.down"):
        assert feature_acts_down.grad is not None, "feature_acts_down must have a gradient"
        if feature_acts_down.grad is not None:
            attribution = (
                feature_acts_down[:cur_batch_size] * feature_acts_down.grad[:cur_batch_size]
            )  # [b, pos, d_sae]
            active_mask = feature_acts_down[0] > 0  # [pos, d_sae]
            pos_ids, feat_ids = active_mask.nonzero(as_tuple=True)
            n_active = pos_ids.shape[0]
            edge_matrix[:, column_start : column_start + n_active] = (
                attribution[:, pos_ids, feat_ids].detach().cpu()  # [b, n_active]
            )
        else:
            n_active = item((feature_acts_down[0] > 0).sum())
            edge_matrix[:, column_start : column_start + n_active] = (
                torch.zeros_like(edge_matrix[:, column_start : column_start + n_active]).detach().cpu()
            )
        column_start += n_active

    for error in _get_cache_activations(edge_matrix.cache_activations, ".error"):
        edge_matrix[:, column_start : column_start + error.shape[1]] = (
            (
                einops.einsum(
                    error[:cur_batch_size],
                    error.grad[:cur_batch_size],
                    "b pos ..., b pos ... -> b pos",
                )
                .detach()
                .cpu()
            )
            if error.grad is not None
            else torch.zeros_like(edge_matrix[:, column_start : column_start + error.shape[1]]).detach().cpu()
        )
        column_start += error.shape[1]

    assert column_start == edge_matrix.shape[1], "column_start must be equal to the number of columns"


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

    def attribute(
        self,
        inputs: torch.Tensor | str,
        replacement_modules: list[SparseDictionary],
        max_n_logits: int = 10,
        desired_logit_prob: float = 0.95,
        batch_size: int = 512,
        max_features: int | None = None,
    ):
        assert self.model is not None, "model must be initialized"
        cache_activations: dict[str, torch.Tensor] = {}
        tokens = ensure_tokenized(inputs, self.tokenizer, device=self.device)
        fwd_hooks_in: list[tuple[Union[str, Callable], Callable]] = [
            (
                hook_in,
                partial(
                    hook_in_fn_builder(replacement_module),
                    replacement_module=replacement_module,
                    cache_activations=cache_activations,
                ),
            )
            for replacement_module in replacement_modules
            for hook_in in replacement_module.cfg.hooks_in
        ]
        fwd_hooks_out: list[tuple[Union[str, Callable], Callable]] = [
            (
                hook_out,
                partial(
                    hook_out_fn_builder(replacement_module),
                    replacement_module=replacement_module,
                    cache_activations=cache_activations,
                    update_error_cache=True,
                ),
            )
            for replacement_module in replacement_modules
            for hook_out in replacement_module.cfg.hooks_out
        ]

        def token_fwd_hook_fn(
            x: torch.Tensor,
            hook: Any,
            cache_activations: dict[str, torch.Tensor],
        ):
            cache_activations["hook_embed"] = x
            cache_activations["hook_embed"].retain_grad()
            return x

        token_fwd_hooks: list[tuple[Union[str, Callable], Callable]] = [
            (
                "hook_embed",
                partial(token_fwd_hook_fn, cache_activations=cache_activations),
            )
        ]
        with self.hooks(fwd_hooks=fwd_hooks_in + fwd_hooks_out + token_fwd_hooks):
            batch_logits = self.forward(einops.repeat(tokens, "n -> b n", b=batch_size))

        with torch.no_grad():
            probs = torch.softmax(batch_logits[0, -1], dim=-1)
            top_p, top_idx = torch.topk(probs, max_n_logits)
            cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
            top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]
        cache_activations["logits"] = batch_logits[:, :, top_idx]

        edge_matrix = EdgeMatrix(
            cache_activations, max_features=max_features
        )  # [downstream_nodes, upstream_nodes], on cpu, [logits+max_features, tokens+features+errors]

        for i in range(0, edge_matrix.n_logits, batch_size):
            cur_batch_size = min(batch_size, edge_matrix.n_logits - i)
            batch_nodes: torch.Tensor = torch.stack(edge_matrix.get_row_values(list(range(i, i + cur_batch_size))))
            batch_nodes -= torch.mean(batch_logits[:cur_batch_size, -1, :], dim=-1).squeeze()  # [cur_batch_size, ]
            _clear_cache_activation_grads(cache_activations)
            batch_nodes.sum().backward(retain_graph=True)
            update_edge_matrix_rows(cast(EdgeMatrix, edge_matrix[i : i + cur_batch_size]))

        def get_normalize_edge_matrix(matrix: torch.Tensor):
            return torch.abs(matrix) / torch.abs(matrix).sum(dim=1, keepdim=True).clamp(min=1e-8)

        with EdgeMatrix.no_wrap():
            features_attributions = einops.einsum(
                get_normalize_edge_matrix(
                    edge_matrix[
                        : edge_matrix.n_logits,
                        edge_matrix.n_tokens : edge_matrix.n_tokens + edge_matrix.n_active_features,
                    ]
                ).to(self.device),
                top_p,
                "logits features, logits -> features",
            )
        visited_features: torch.Tensor | None = None

        for i in tqdm(range(0, edge_matrix.max_features, batch_size)):
            cur_batch = min(batch_size, edge_matrix.max_features - i)
            if visited_features is not None:
                features_attributions[visited_features] = float("-inf")
            _, batch_feature_ids = torch.topk(features_attributions, cur_batch)
            batch_column_ids = batch_feature_ids + edge_matrix.n_tokens
            _clear_cache_activation_grads(cache_activations)
            bwd = edge_matrix.get_column_values(batch_column_ids.tolist(), "up")
            torch.stack(bwd).sum().backward(retain_graph=True)
            update_edge_matrix_rows(
                cast(EdgeMatrix, edge_matrix[i + edge_matrix.n_logits : i + cur_batch + edge_matrix.n_logits])
            )
            edge_matrix.update_row_info_from_columns(batch_column_ids.tolist())
            visited_features = (
                batch_feature_ids if visited_features is None else torch.cat([visited_features, batch_feature_ids])
            )
            with EdgeMatrix.no_wrap():
                features_attributions += einops.einsum(
                    get_normalize_edge_matrix(edge_matrix[: edge_matrix.n_logits, batch_column_ids]).to(self.device),
                    get_normalize_edge_matrix(
                        edge_matrix[
                            i + edge_matrix.n_logits : i + cur_batch + edge_matrix.n_logits,
                            edge_matrix.n_tokens : edge_matrix.n_tokens + edge_matrix.n_active_features,
                        ]
                    ).to(self.device),
                    top_p,
                    "logits batch_features, batch_features features, logits -> features",
                )

        return edge_matrix

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
        """Run with activation caching, stopping at a given hook for efficiency."""
        assert self.model is not None, "model must be initialized"
        if self.device_mesh is None:
            return run_with_cache_until(self.model, *args, **kwargs)

        args = pytree.tree_map(self._to_tensor, args)
        kwargs = pytree.tree_map(self._to_tensor, kwargs)
        return pytree.tree_map(self._to_dtensor, run_with_cache_until(self.model, *args, **kwargs))

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

        _, activations = self.run_with_cache_until(tokens, names_filter=hook_points, until=hook_points[-1])

        # we do not want to filter out eos. It might be end of chats and include useful information
        assert self.pad_token_id is not None and self.bos_token_id is not None, "Pad and BOS token IDs must be set"
        mask = torch.logical_and(tokens.ne(self.pad_token_id), tokens.ne(self.bos_token_id)).int()
        attention_mask = torch.logical_and(tokens.ne(self.pad_token_id), tokens.ne(self.bos_token_id)).int()

        return {hook_point: activations[hook_point] for hook_point in hook_points} | {
            "tokens": tokens,
            "mask": mask,
            "attention_mask": attention_mask,
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
