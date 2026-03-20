from __future__ import annotations

import json
import os
import re
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from itertools import accumulate
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Literal,
    Optional,
    Self,
    Sequence,
    Union,
    cast,
    overload,
)

import einops
import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from huggingface_hub import hf_hub_download
from torch._tensor import Tensor
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map
from torch.types import Number
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
from transformer_lens.utilities import devices as tl_devices

from lm_saes.backend.run_with_cache_until import run_with_cache_until
from lm_saes.config import BaseModelConfig
from lm_saes.utils.auto import PretrainedSAEType, auto_infer_pretrained_sae_type
from lm_saes.utils.discrete import DiscreteMapper
from lm_saes.utils.distributed import DimMap
from lm_saes.utils.misc import ensure_tokenized, pad_and_truncate_tokens
from lm_saes.utils.timer import timer

if TYPE_CHECKING:
    from lm_saes.models.clt import CrossLayerTranscoder
    from lm_saes.models.lorsa import LowRankSparseAttention
    from lm_saes.models.sparse_dictionary import SparseDictionary


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


def replacement_hookin_fn_builder(replacement_module: SparseDictionary) -> Callable:
    from lm_saes.models.lorsa import LowRankSparseAttention
    from lm_saes.models.sae import SparseAutoEncoder

    if isinstance(replacement_module, SparseAutoEncoder):

        def sae_hookin_fn(
            x: torch.Tensor,
            hook: HookPoint,
            replacement_module: SparseDictionary,
            cache_activations: dict[str, torch.Tensor],
        ):
            assert hook.name in replacement_module.cfg.hooks_in, "Hook point must be in hook points in"
            cache_activations[hook.name + ".x"] = x
            cache_activations[hook.name + ".feature_acts.up"] = replacement_module.encode(
                cache_activations[hook.name + ".x"]
            )
            cache_activations[hook.name + ".feature_acts.down"] = (
                cache_activations[hook.name + ".feature_acts.up"].detach().requires_grad_(True)
            )
            return x

        return sae_hookin_fn

    elif isinstance(replacement_module, LowRankSparseAttention):

        def lorsa_hookin_fn(
            x: torch.Tensor,
            hook: HookPoint,
            replacement_module: LowRankSparseAttention,
            cache_activations: dict[str, torch.Tensor],
        ):
            assert hook.name in replacement_module.cfg.hooks_in, "Hook point must be in hook points in"
            cache_activations[hook.name + ".x"] = x
            encode_result = replacement_module.encode(
                cache_activations[hook.name + ".x"],
                return_hidden_pre=False,
                return_attention_pattern=True,
                return_attention_score=True,
            )
            cache_activations[hook.name + ".attn_pattern"] = encode_result[1].detach()
            cache_activations[hook.name + ".feature_acts.up"] = encode_result[0]  # batch, seq_len, d_sae
            cache_activations[hook.name + ".feature_acts.down"] = (
                cache_activations[hook.name + ".feature_acts.up"].detach().requires_grad_(True)
            )

            return x

        return lorsa_hookin_fn

    else:
        # TODO: handle other replacement modules such as CLT
        raise ValueError(f"Unsupported replacement module type: {type(replacement_module)}")


def replacement_hookout_fn_builder(replacement_module: SparseDictionary) -> Callable:
    # if isinstance(replacement_module, CrossLayerTranscoder):
    #     raise NotImplementedError("CLT hookout function builder not implemented")

    # else:

    def hookout_fn(
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
        return error + reconstructed

    return hookout_fn


def detach_hook_builder(models: TransformerLensLanguageModel) -> list[tuple[str, Callable]]:
    def detach_hook_fn(x: torch.Tensor, hook: HookPoint):
        return x.detach()

    assert models.model is not None, "model must be initialized"
    detach_hooks = []
    for block in models.model.blocks:
        for module_name in ["ln1", "ln2", "ln1_post", "ln2_post"]:
            if hasattr(block, module_name) and isinstance(getattr(block, module_name), torch.nn.Module):
                detach_hooks.append((getattr(block, module_name).hook_scale.name, detach_hook_fn))
        if hasattr(block, "attn") and isinstance(block.attn, torch.nn.Module):
            assert isinstance(block.attn.hook_pattern, HookPoint), "attn must be an instance of AbstractAttention"
            detach_hooks.append((block.attn.hook_pattern.name, detach_hook_fn))

    detach_hooks.append(("ln_final.hook_scale", detach_hook_fn))
    return detach_hooks


@dataclass
class Node:
    key: Any
    """Key of the node. Should be a hashable object."""

    indices: torch.Tensor
    """Indices of elements in the node's data. Should be of shape `(n_elements, d_index)`."""

    offsets: torch.Tensor
    """In-tensor offsets of elements. Should be of shape `(n_elements,)`."""


@dataclass
class NodeInfo:
    """Node identifier with key and indices into the node's data."""

    key: Any
    """Key of the node. Should be a hashable object."""

    indices: torch.Tensor
    """Indices of elements in the node's data. Should be of shape `(n_elements, d_index)`."""

    def __len__(self) -> int:
        return self.indices.shape[0]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NodeInfo):
            return False
        return self.key == other.key and torch.equal(self.indices, other.indices)

    def __getitem__(self, index: slice) -> Self:
        return replace(self, indices=self.indices[index])

    def split(self, batch_size: int) -> tuple[Self, Self]:
        return self[:batch_size], self[batch_size:]


class NodeIndexedTensor:
    def __init__(
        self,
        n_dims: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.n_dims = n_dims
        self.data = torch.zeros([0] * self.n_dims, dtype=dtype, device=device)
        self.mappers = tuple(DiscreteMapper() for _ in range(n_dims))

        self.node_infos: tuple[list[NodeInfo], ...] = tuple([] for _ in range(n_dims))

        # NodeIndexedTensor maintains 3 mappings for efficient node retrieval (sometimes with specific index slices), respectively from node key, node indices and in-data offsets.

        # Mapping node keys to nodes.
        self.node_mappings: tuple[dict[Any, Node], ...] = tuple({} for _ in range(n_dims))

        # Mapping node keys to node inverse indices.
        self.inv_indices_mappings: tuple[dict[Any, torch.Tensor], ...] = tuple({} for _ in range(n_dims))

        # Mapping in-data offsets to node elements (each element is represented by its node key and index).
        # - `keys` are discretely encoded node keys.
        # - `indices` are the indices of the indices of the node elements. (A second-order indices). This second order indices are introduced to handle arbitrary index dimensions.
        self.offset_mapping: tuple[dict[str, Tensor], ...] = tuple(
            {
                "keys": torch.empty(0, device=device, dtype=torch.long),
                "indices": torch.empty(0, device=device, dtype=torch.long),
            }
            for _ in range(n_dims)
        )

    @classmethod
    def from_data(cls, data: torch.Tensor, node_infos: tuple[Sequence[NodeInfo], ...]) -> Self:
        self = cls.__new__(cls)
        self.n_dims = data.ndim
        self.data = data
        self.mappers = tuple(DiscreteMapper() for _ in range(self.n_dims))

        self.node_infos = tuple(list(node_infos_per_dim) for node_infos_per_dim in node_infos)

        # Mapping node keys to nodes.
        self.node_mappings = tuple({} for _ in range(self.n_dims))

        # Mapping node keys to node inverse indices.
        self.inv_indices_mappings = tuple({} for _ in range(self.n_dims))

        # Mapping in-data offsets to node elements (each element is represented by its node key and index).
        # - `keys` are discretely encoded node keys.
        # - `indices` are the indices of the indices of the node elements. (A second-order indices). This second order indices are introduced to handle arbitrary index dimensions.
        self.offset_mapping = tuple(
            {
                "keys": torch.empty(0, device=self.data.device, dtype=torch.long),
                "indices": torch.empty(0, device=self.data.device, dtype=torch.long),
            }
            for _ in range(self.n_dims)
        )

        for dim, node_infos_per_dim in enumerate(node_infos):
            self._register_nodes(node_infos_per_dim, dim, 0)
        return self

    @classmethod
    def from_node_infos(
        cls,
        node_infos: tuple[Sequence[NodeInfo], ...],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Self:
        shape = [
            sum([node_info.indices.shape[0] for node_info in node_infos_per_dim]) for node_infos_per_dim in node_infos
        ]
        return cls.from_data(torch.zeros(shape, dtype=dtype, device=device), node_infos)

    def _register_nodes(self, node_infos: Sequence[NodeInfo], dim: int, start: int):
        for node_info in node_infos:
            node_key_id = self.mappers[dim].encode([node_info.key])[0]
            node_length = node_info.indices.shape[0]
            node = Node(
                node_info.key,
                node_info.indices,
                torch.arange(start, start + node_length, device=self.data.device),
            )

            # Append node to node mapping
            if node.key not in self.node_mappings[dim]:
                n_original_elements = 0
                self.node_mappings[dim][node.key] = node
            else:
                n_original_elements = self.node_mappings[dim][node.key].indices.shape[0]
                self.node_mappings[dim][node.key].offsets = torch.cat(
                    [self.node_mappings[dim][node.key].offsets, node.offsets],
                    dim=0,
                )
                self.node_mappings[dim][node.key].indices = torch.cat(
                    [self.node_mappings[dim][node.key].indices, node.indices],
                    dim=0,
                )

            # Update inverse indices mapping based on new node data in node mapping
            inv_indices = torch.empty(
                [
                    int(self.node_mappings[dim][node.key].indices[:, i].max() + 1)
                    for i in range(self.node_mappings[dim][node.key].indices.shape[1])
                ],
                device=self.data.device,
                dtype=torch.long,
            )
            inv_indices[self.node_mappings[dim][node.key].indices.unbind(dim=1)] = torch.arange(
                self.node_mappings[dim][node.key].indices.shape[0], device=self.data.device
            )
            self.inv_indices_mappings[dim][node.key] = inv_indices

            # Update offset mapping
            self.offset_mapping[dim]["indices"] = torch.cat(
                [
                    self.offset_mapping[dim]["indices"],
                    torch.arange(
                        n_original_elements,
                        n_original_elements + node_length,
                        device=self.data.device,
                        dtype=torch.long,
                    ),
                ],
                dim=0,
            )
            self.offset_mapping[dim]["keys"] = torch.cat(
                [
                    self.offset_mapping[dim]["keys"],
                    torch.full(
                        (node_length,),
                        node_key_id,
                        device=self.data.device,
                        dtype=torch.long,
                    ),
                ],
                dim=0,
            )
            start += node_length

        if start != self.data.shape[dim]:
            raise ValueError(
                f"Registered nodes size {start} does not match data size {self.data.shape[dim]} along dimension {dim}"
            )

    def _add_elements(self, node_infos: Sequence[NodeInfo], dim: int):
        start = self.data.shape[dim]
        new_data_shape = [
            self.data.shape[i] if i != dim else sum([node_info.indices.shape[0] for node_info in node_infos])
            for i in range(self.n_dims)
        ]

        self.data = torch.cat(
            [self.data, torch.zeros(new_data_shape, dtype=self.data.dtype, device=self.data.device)],
            dim=dim,
        )

        self.node_infos = tuple(n if i != dim else n + node_infos for i, n in enumerate(self.node_infos))

        self._register_nodes(node_infos, dim, start)

    def _nodes_to_offsets(self, node_infos: Sequence[NodeInfo], dim: int) -> torch.Tensor:
        offsets = torch.empty(
            sum([node_info.indices.shape[0] for node_info in node_infos]), device=self.data.device, dtype=torch.long
        )
        start = 0
        for i, node_info in enumerate(node_infos):
            key, indices = node_info.key, node_info.indices
            offsets[start : start + indices.shape[0]] = self.node_mappings[dim][key].offsets[
                self.inv_indices_mappings[dim][key][indices.unbind(dim=1)]
            ]
            start += indices.shape[0]
        return offsets

    def _offsets_to_nodes(self, offsets: torch.Tensor, dim: int) -> Sequence[NodeInfo]:
        """Order-preserving inverse of :meth:`_nodes_to_offsets` at
        the offset level. Adjacent offsets with the same node key are
        grouped into a single ``NodeInfo`` entry.
        """

        keys_encoded = self.offset_mapping[dim]["keys"][offsets]
        indices = self.offset_mapping[dim]["indices"][offsets]

        # Group consecutive keys and their indices
        unique_keys_encoded, inverse_indices = torch.unique_consecutive(keys_encoded, return_inverse=True)
        unique_keys = self.mappers[dim].decode(unique_keys_encoded.tolist())
        node_infos: Sequence[NodeInfo] = [
            NodeInfo(
                key=unique_keys[i],
                indices=self.node_mappings[dim][unique_keys[i]].indices[indices[inverse_indices == i]],
            )
            for i in range(len(unique_keys))
        ]
        return node_infos

    def __getitem__(self, key: tuple[Sequence[NodeInfo] | None, ...] | Sequence[NodeInfo] | None) -> Self:
        """Index the tensor with NodeInfo selections for each dimension.

        Each dimension accepts a ``Sequence[NodeInfo]`` to select specific node
        elements, or ``None`` to select all elements along that dimension.

        For a 1-D tensor a single ``Sequence[NodeInfo]`` (or ``None``) can be
        passed directly; for higher-rank tensors, pass a tuple with one entry
        per dimension.

        Args:
            key: Dimension selectors. A tuple of ``(Sequence[NodeInfo] | None)``
                with length equal to :attr:`n_dims`, or a bare
                ``Sequence[NodeInfo] | None`` for 1-D tensors.

        Returns:
            A new :class:`NodeIndexedTensor` (or subclass) containing the
            selected sub-tensor with updated node mappings.
        """
        if not isinstance(key, tuple):
            key = (key,)
        key = cast(tuple[Sequence[NodeInfo] | None, ...], key)
        if len(key) != self.n_dims:
            raise ValueError(f"Expected {self.n_dims} dimension selectors, got {len(key)}")

        data = self.data
        for dim in range(self.n_dims):
            node_infos = key[dim]

            if node_infos is None:
                continue

            offsets = self._nodes_to_offsets(node_infos, dim)
            data = data.index_select(dim, offsets)

        node_infos = tuple(
            node_infos_per_dim if node_infos_per_dim is not None else self.node_infos[dim]
            for dim, node_infos_per_dim in enumerate(key)
        )

        return self.__class__.from_data(data=data, node_infos=node_infos)

    def __setitem__(self, key: tuple[Sequence[NodeInfo] | None, ...] | Sequence[NodeInfo] | None, value: Self | Tensor):
        """Assign to a NodeInfo-selected sub-tensor.

        Args:
            key: Dimension selectors. A tuple of ``(Sequence[NodeInfo] | None)``
                with length equal to :attr:`n_dims`, or a bare
                ``Sequence[NodeInfo] | None`` for 1-D tensors.
            value: Tensor values to write to the selected region. When a
                :class:`NodeIndexedTensor` is provided, its underlying
                :attr:`data` tensor is assigned.

        Returns:
            The updated tensor instance.
        """
        if not isinstance(key, tuple):
            key = (key,)
        key = cast(tuple[Sequence[NodeInfo] | None, ...], key)
        if len(key) != self.n_dims:
            raise ValueError(f"Expected {self.n_dims} dimension selectors, got {len(key)}")

        value = cast(Tensor, value.data if isinstance(value, NodeIndexedTensor) else value)

        indexers: list[torch.Tensor] = []
        for dim in range(self.n_dims):
            node_infos = key[dim]
            offsets = (
                self._nodes_to_offsets(node_infos, dim)
                if node_infos is not None
                else torch.arange(self.data.shape[dim], device=self.data.device, dtype=torch.long)
            )
            view_shape = [1] * self.n_dims
            view_shape[dim] = offsets.shape[0]
            indexers.append(offsets.view(*view_shape))

        self.data[tuple(indexers)] = value
        return self

    def __add__(self, other: Self):
        data = self.data + other.data
        return self.__class__.from_data(data, self.node_infos)


class NodeIndexedVector(NodeIndexedTensor):
    def __init__(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            n_dims=1,
            device=device,
            dtype=dtype,
        )

    def add_nodes(self, node_infos: Sequence[NodeInfo]):
        self._add_elements(node_infos, 0)

    def topk(self, k: int, ignore_node_infos: Sequence[NodeInfo] | None = None):
        ignore_indices = (
            self._nodes_to_offsets(ignore_node_infos, 0)
            if ignore_node_infos is not None and len(ignore_node_infos) > 0
            else None
        )
        if ignore_indices is not None:
            self.data[ignore_indices] = float("-inf")
        topk_values, topk_indices = torch.topk(self.data, k=k, dim=0)
        return topk_values, self._offsets_to_nodes(topk_indices, 0)

    @overload
    def matmul(self, other: NodeIndexedMatrix, _check_node_matching: bool = False) -> NodeIndexedVector: ...
    @overload
    def matmul(self, other: NodeIndexedVector, _check_node_matching: bool = False) -> Number: ...

    def matmul(self, other: NodeIndexedMatrix | NodeIndexedVector, _check_node_matching: bool = False):
        if _check_node_matching:
            a, b = self.node_infos[0], other.node_infos[0]
            if len(a) != len(b) or any(not ai == bi for ai, bi in zip(a, b)):
                raise ValueError(f"Node matching failed: {a} != {b}")

        data = self.data @ other.data

        if isinstance(other, NodeIndexedMatrix):
            return NodeIndexedVector.from_data(data, node_infos=(other.node_infos[1],))
        elif isinstance(other, NodeIndexedVector):
            return data.item()
        else:
            raise ValueError(
                f"Invalid type as right operand in NodeIndexedVector.matmul: {type(other)}. Expected NodeIndexedMatrix or NodeIndexedVector."
            )

    def __matmul__(self, other: NodeIndexedMatrix):
        return self.matmul(other)


class NodeIndexedMatrix(NodeIndexedTensor):
    def __init__(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            n_dims=2,
            device=device,
            dtype=dtype,
        )

    @property
    def target(self) -> dict[Any, Node]:
        return self.node_mappings[0]

    @property
    def source(self) -> dict[Any, Node]:
        return self.node_mappings[1]

    def add_target(self, node_infos: Sequence[NodeInfo] | NodeInfo):
        node_infos = node_infos if isinstance(node_infos, Sequence) else [node_infos]
        self._add_elements(node_infos, 0)

    def add_source(self, node_infos: Sequence[NodeInfo] | NodeInfo):
        node_infos = node_infos if isinstance(node_infos, Sequence) else [node_infos]
        self._add_elements(node_infos, 1)

    @overload
    def matmul(self, other: NodeIndexedVector, _check_node_matching: bool = False) -> NodeIndexedVector: ...
    @overload
    def matmul(self, other: NodeIndexedMatrix, _check_node_matching: bool = False) -> NodeIndexedMatrix: ...

    def matmul(self, other: NodeIndexedVector | NodeIndexedMatrix, _check_node_matching: bool = False):
        if _check_node_matching:
            a, b = self.node_infos[1], other.node_infos[0]
            if len(a) != len(b) or any(not ai == bi for ai, bi in zip(a, b)):
                raise ValueError(f"Node matching failed: {a} != {b}")

        data = self.data @ other.data

        if isinstance(other, NodeIndexedVector):
            return NodeIndexedVector.from_data(data, node_infos=(self.node_infos[0],))
        elif isinstance(other, NodeIndexedMatrix):
            return NodeIndexedMatrix.from_data(data, node_infos=(self.node_infos[0], other.node_infos[1]))
        else:
            raise ValueError(f"Invalid type as right operand in NodeIndexedMatrix.matmul: {type(other)}")

    @overload
    def __matmul__(self, other: NodeIndexedVector) -> NodeIndexedVector: ...
    @overload
    def __matmul__(self, other: NodeIndexedMatrix) -> NodeIndexedMatrix: ...

    def __matmul__(self, other: NodeIndexedVector | NodeIndexedMatrix):
        return self.matmul(other)


# class AttributionGraph:
#     def __init__(
#         self,
#         cache_activations: dict[str, torch.Tensor],
#         max_features: int | None = None,
#         device: torch.device | str = "cpu",
#     ):
#         self.device = torch.device(device) if isinstance(device, str) else device
#         self.cache_activations = cache_activations
#         self.n_tokens = cache_activations["hook_embed"].shape[1]
#         self.n_logits = cache_activations["logits"].shape[-1]
#         self.n_active_features = int(
#             item(
#                 cast(
#                     torch.Tensor,
#                     sum([v[0].gt(0).sum() for k, v in cache_activations.items() if k.endswith(".feature_acts.up")]),
#                 )
#             )
#         )
#         self.max_features = min(max_features or self.n_active_features, self.n_active_features)
#         self.n_error = self.n_tokens * len([v for k, v in cache_activations.items() if k.endswith(".error")])
#         self.matrix = self._create_matrix_with_targets()
#         self.selected_features: torch.Tensor | None = None

#     def _create_matrix_with_sources(self) -> NodeIndexedMatrix:
#         source_infos: list[NodeInfo] = (
#             [
#                 ((pos, "hook_embed"), torch.tensor(0, dtype=torch.long, device=self.device))
#                 for pos in range(self.n_tokens)
#             ]
#             + [
#                 ((pos, key), torch.nonzero(self.cache_activations[key][0, pos]))
#                 for pos in range(self.n_tokens)
#                 for key in self.cache_activations
#                 if key.endswith(".feature_acts.down")
#             ]
#             + [
#                 ((pos, key), torch.tensor(0, dtype=torch.long, device=self.device))
#                 for pos in range(self.n_tokens)
#                 for key in self.cache_activations
#                 if key.endswith(".error")
#             ]
#         )
#         return NodeIndexedMatrix.from_node_infos(node_infos=([], source_infos), device=self.device)

#     def bwd_values(self, target_node_infos: list[NodeInfo]) -> torch.Tensor:
#         values = []
#         batch_idx = 0
#         for (pos_id, hook_key), indices in target_node_infos:
#             local_batch = torch.arange(indices.shape[0], device=self.device) + batch_idx
#             values.append(
#                 self.cache_activations[hook_key.replace(".feature_acts.down", ".feature_acts.up")][
#                     local_batch, pos_id, indices
#                 ]
#             )
#             batch_idx += indices.shape[0]
#         return torch.cat(values, dim=0)

#     def clear_cache_activation_grads(self) -> None:
#         """Clear gradients for all cached activations in-place."""
#         for activation in self.cache_activations.values():
#             if activation.requires_grad and activation.grad is not None:
#                 activation.grad = None

#     def calculate_attribution(
#         self,
#         target_node_infos: list[NodeInfo],
#     ):
#         self.matrix.add_target(target_node_infos)
#         submatrix = self.matrix.get_submatrix(target_node_infos=target_node_infos)
#         attribution = torch.zeros_like(submatrix.matrix)  # (batch_size / n_target, n_source)
#         for source_node in submatrix.source_nodes:
#             (pos_id, key), indices, matrix_indices = source_node.key, source_node.indices, source_node.matrix_indices
#             tensor = (
#                 self.cache_activations[key][:, pos_id, indices]
#                 if key.endswith(".feature_acts.down")
#                 else self.cache_activations[key][:, pos_id]
#             )
#             grad = torch.zeros_like(tensor) if tensor.grad is None else tensor.grad
#             contribution = (
#                 (tensor[: len(target_node_infos)] * grad[: len(target_node_infos)])
#                 if key.endswith(".feature_acts.down")
#                 else tensor[: len(target_node_infos)] * grad[: len(target_node_infos)].sum(dim=1, keepdim=True)
#             ).to(self.device)
#             attribution[:, matrix_indices] += contribution

#         submatrix.update_matrix(attribution)
#         return submatrix


# def compute_feature_influences(
#     batch_size: int,
#     feature_infos: list[NodeInfo],
#     logits_infos: list[NodeInfo],
#     selected_node_infos: dict[Any, NodeInfo],
#     ignored_indices: torch.Tensor | None = None,
#     max_iter: int = 100,
# ) -> tuple[list[NodeInfo], torch.Tensor]:
#     pass


@dataclass
class NodeInfoRef(NodeInfo):
    """NodeInfo with reference to node (tensor) in computation graph."""

    ref: torch.Tensor


class NodeInfoQueue[T: NodeInfo]:
    def __init__(self, node_infos: Sequence[T]):
        self.queue = list(node_infos)

    def enqueue(self, node_info: Sequence[T]):
        self.queue.extend(node_info)

    def dequeue(self, batch_size: int) -> Sequence[T]:
        accumulated = 0
        results = []
        while accumulated < batch_size and len(self.queue) > 0:
            if accumulated + len(self.queue[0]) > batch_size:
                results.append(self.queue[0][: batch_size - accumulated])
                self.queue[0] = self.queue[0][batch_size - accumulated :]
                accumulated = batch_size
            else:
                results.append(self.queue.pop(0))
                accumulated += len(results[-1])
        return results

    def iter(self, batch_size: int) -> Iterator[Sequence[T]]:
        while len(self.queue) > 0:
            yield self.dequeue(batch_size)


class NodeInfoSet[T: NodeInfo]:
    def __init__(self, node_infos: Sequence[T] = []):
        self.node_dict: dict[Any, T] = {}
        self.extend(node_infos)

    def extend(self, node_infos: Sequence[T]):
        for node_info in node_infos:
            if node_info.key not in self.node_dict:
                self.node_dict[node_info.key] = replace(node_info)
            else:
                self.node_dict[node_info.key].indices = torch.cat(
                    [self.node_dict[node_info.key].indices, node_info.indices],
                    dim=0,
                )

    def __len__(self) -> int:
        return sum(len(node_info) for node_info in self.node_dict.values())

    def to_list(self) -> list[T]:
        return list(self.node_dict.values())


def get_normalized_matrix(matrix: NodeIndexedMatrix) -> NodeIndexedMatrix:
    return NodeIndexedMatrix.from_data(
        torch.abs(matrix.data) / torch.abs(matrix.data).sum(dim=1, keepdim=True).clamp(min=1e-8), matrix.node_infos
    )


def compute_intermediates_attribution(
    attribution: NodeIndexedMatrix,
    targets: Sequence[NodeInfoRef],
    intermediates: Sequence[tuple[NodeInfoRef, NodeInfoRef]],
    max_iter: int,
) -> NodeIndexedMatrix:
    attribution = get_normalized_matrix(attribution)
    influence = attribution[targets, None]
    if len(intermediates) == 0:
        return influence
    t2i: NodeIndexedMatrix = attribution[targets, [intermediate[1] for intermediate in intermediates]]
    for _ in range(max_iter):
        cur_influence = t2i @ attribution[[intermediate[0] for intermediate in intermediates], None]
        if not torch.any(cur_influence.data):
            break
        influence += cur_influence
        t2i = (
            t2i
            @ attribution[
                [intermediate[0] for intermediate in intermediates], [intermediate[1] for intermediate in intermediates]
            ]
        )
    return influence


def greedily_collect_attribution(
    targets: Sequence[NodeInfoRef],
    sources: Sequence[NodeInfoRef],
    intermediates: Sequence[tuple[NodeInfoRef, NodeInfoRef]],  # [up as target, down as source]
    max_intermediates: int,
    reduction_weight: torch.Tensor,
    max_iter: int = 100,
) -> NodeIndexedMatrix:
    """
    Greedily collect attribution from targets to sources through intermediates.
    """

    all_sources = list(sources) + [intermediate[1] for intermediate in intermediates]

    attribution = NodeIndexedMatrix.from_node_infos(
        node_infos=(targets, all_sources),
        device=targets[0].ref.device,
        dtype=targets[0].ref.dtype,
    )

    batch_size = targets[0].ref.shape[0]

    queue = NodeInfoQueue(targets)

    def values(node_infos: Sequence[NodeInfoRef]) -> list[torch.Tensor]:
        return [node_info.ref[:, *node_info.indices.unbind(dim=1)] for node_info in node_infos]

    def grads(node_infos: Sequence[NodeInfoRef]) -> list[torch.Tensor]:
        return [
            node_info.ref.grad[:, *node_info.indices.unbind(dim=1)]
            if node_info.ref.grad is not None
            else torch.zeros_like(node_info.ref[:, *node_info.indices.unbind(dim=1)])
            for node_info in node_infos
        ]

    def clear_grads(node_infos: Sequence[NodeInfoRef]) -> None:
        for node_info in node_infos:
            node_info.ref.grad = None

    for target_batch in queue.iter(batch_size):
        clear_grads(all_sources)
        root = torch.diag(torch.cat(values(target_batch), dim=1))
        root.sum().backward(retain_graph=True)
        attribution[target_batch, None] = torch.cat(
            [
                einops.einsum(
                    value[: root.shape[0]],
                    grad[: root.shape[0]],
                    "batch n_elements ..., batch n_elements ... -> batch n_elements",
                )
                for value, grad in zip(values(all_sources), grads(all_sources))
            ],
            dim=1,
        )

    def retrieval_from_intermediates(node_infos: Sequence[NodeInfo]):
        node_refs = []
        for node_info in node_infos:
            for intermediate in intermediates:
                if node_info.key == intermediate[0].key:
                    node_refs.append(
                        (
                            NodeInfoRef(key=node_info.key, indices=node_info.indices, ref=intermediate[0].ref),
                            NodeInfoRef(key=node_info.key, indices=node_info.indices, ref=intermediate[1].ref),
                        )
                    )

        return node_refs

    intermediates_set = NodeInfoSet()
    reduction_weight: NodeIndexedVector = NodeIndexedVector.from_data(reduction_weight, node_infos=(targets,))
    for i in tqdm(range(0, max_intermediates, batch_size)):
        cur_batch_size = min(batch_size, max_intermediates - i)
        intermediates_attribution = compute_intermediates_attribution(
            attribution, targets, retrieval_from_intermediates(intermediates_set.to_list()), max_iter
        )
        influence = (
            reduction_weight @ intermediates_attribution[None, [intermediate[1] for intermediate in intermediates]]
        )

        _, selected_node_infos = influence.topk(k=cur_batch_size, ignore_node_infos=intermediates_set.to_list())

        intermediates_set.extend(selected_node_infos)
        attribution.add_target(selected_node_infos)

        clear_grads(all_sources)
        node_refs = retrieval_from_intermediates(selected_node_infos)
        root = torch.diag(torch.cat(values([node_ref[0] for node_ref in node_refs]), dim=1))
        root.sum().backward(retain_graph=True)
        attribution[selected_node_infos, None] = torch.cat(
            [
                einops.einsum(
                    value[: root.shape[0]],
                    grad[: root.shape[0]],
                    "batch n_elements ..., batch n_elements ... -> batch n_elements",
                )
                for value, grad in zip(values(all_sources), grads(all_sources))
            ],
            dim=1,
        )

    return attribution


# def compute_feature_influences(
#     graph: AttributionGraph,
#     max_iter: int = 100,
# ) -> torch.Tensor:

#     with graph.adjacency_matrix.unwrap():
#         influences = einops.einsum(
#             graph.cache_activations["top_p"].to(graph.adjacency_matrix.device),
#             get_normalized_matrix(
#                 graph.adjacency_matrix[
#                     :,
#                     graph.n_tokens : graph.n_tokens + graph.n_active_features,
#                 ]
#             ),
#             "logits, logits features -> features",
#         )
#     if graph.selected_features is None:
#         return influences

#     with graph.adjacency_matrix.unwrap():
#         feature_to_feature = get_normalized_matrix(
#             graph.adjacency_matrix[
#                 graph.n_logits : graph.selected_features.shape[0] + graph.n_logits,
#                 graph.n_tokens : graph.n_tokens + graph.n_active_features,
#             ]
#         )
#     prod = influences
#     for _ in range(max_iter):
#         prod = prod[graph.selected_features] @ feature_to_feature
#         if not torch.any(prod):
#             break
#         influences += prod
#     return influences


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
                n_devices=cfg.n_devices,
                cache_dir=cfg.cache_dir,
                hf_model=hf_model,
                hf_config=hf_model.config,
                tokenizer=hf_tokenizer,
                dtype=cfg.dtype,  # type: ignore ; issue with transformer_lens
            )
            if hf_model and not cfg.tokenizer_only
            else None
        )
        if self.model is not None and cfg.n_devices > 1:
            self._move_model_modules_to_pipeline_devices()

    def _move_model_modules_to_pipeline_devices(self) -> None:
        assert self.model is not None
        if self.model.cfg.n_layers == 0:
            return

        first_device = tl_devices.get_device_for_block_index(0, self.model.cfg)
        last_device = tl_devices.get_device_for_block_index(self.model.cfg.n_layers - 1, self.model.cfg)

        self.model.embed.to(first_device)
        self.model.hook_embed.to(first_device)
        if self.model.cfg.positional_embedding_type != "rotary":
            self.model.pos_embed.to(first_device)
            self.model.hook_pos_embed.to(first_device)

        for layer, block in enumerate(self.model.blocks):
            block.to(tl_devices.get_device_for_block_index(layer, self.model.cfg))

        if hasattr(self.model, "ln_final"):
            self.model.ln_final.to(last_device)
        self.model.unembed.to(last_device)

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
        fwd_hooks_in: list[tuple[str, Callable]] = [
            (
                hook_in,
                partial(
                    replacement_hookin_fn_builder(replacement_module),
                    replacement_module=replacement_module,
                    cache_activations=cache_activations,
                ),
            )
            for replacement_module in replacement_modules
            for hook_in in replacement_module.cfg.hooks_in
        ]
        fwd_hooks_out: list[tuple[str, Callable]] = [
            (
                hook_out,
                partial(
                    replacement_hookout_fn_builder(replacement_module),
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

        token_fwd_hooks: list[tuple[str, Callable]] = [
            (
                "hook_embed",
                partial(token_fwd_hook_fn, cache_activations=cache_activations),
            )
        ]
        detach_hooks: list[tuple[str, Callable]] = detach_hook_builder(self)
        with self.hooks(
            fwd_hooks=cast(
                list[tuple[str | Callable, Callable]],
                fwd_hooks_in + fwd_hooks_out + token_fwd_hooks + detach_hooks,
            )
        ):
            batch_logits = self.forward(einops.repeat(tokens, "n -> b n", b=batch_size))

        with torch.no_grad():
            probs = torch.softmax(batch_logits[0, -1], dim=-1)
            top_p, top_idx = torch.topk(probs, max_n_logits)
            cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
            top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

        # cache_activations["logits"] = batch_logits[:, :, top_idx]
        # cache_activations["top_p"] = top_p

        # graph = AttributionGraph(cache_activations, max_features)

        # for i in range(0, graph.n_logits, batch_size):
        #     cur_batch_size = min(batch_size, graph.n_logits - i)
        #     node_infos = [
        #         (
        #             (graph.n_tokens - 1, "logits"),
        #             torch.arange(i, i + cur_batch_size, dtype=torch.long, device=graph.device),
        #         )
        #     ]
        #     bwd_values = graph.bwd_values(node_infos)
        #     bwd_values -= torch.mean(batch_logits[:cur_batch_size, -1, :], dim=-1).squeeze()
        #     graph.clear_cache_activation_grads()
        #     bwd_values.sum().backward(retain_graph=True)
        #     graph.calculate_attribution(node_infos)

        # logits_infos = [
        #     ((graph.n_tokens - 1, "logits"), torch.arange(graph.n_logits, dtype=torch.long, device=graph.device))
        # ]

        # ignored_indices = None
        # selected_node_infos = {}
        # for i in tqdm(range(0, graph.max_features, batch_size)):
        #     cur_batch = min(batch_size, graph.max_features - i)
        #     features_attributions = compute_feature_influences(
        #         graph,
        #         cur_batch,
        #         logits_infos,
        #         selected_node_infos,
        #         ignored_indices,
        #         max_iter=len(replacement_modules) + 10,
        #     )
        #     if graph.selected_features is not None:
        #         features_attributions[graph.selected_features] = float("-inf")
        #     _, batch_feature_ids = torch.topk(features_attributions, cur_batch)
        #     features_target_metadata = graph.get_target_nodes(
        #         (batch_feature_ids + graph.n_tokens).tolist(), mapping_from="source"
        #     )
        #     batch_target_values = torch.stack(
        #         [tensor[i] for i, (tensor, _) in enumerate(features_target_metadata)]
        #     ).sum()
        #     graph.clear_cache_activation_grads()
        #     batch_target_values.backward(retain_graph=True)

        #     # graph.update_target(features_target_metadata, calculate_attribution(graph.adjacency_matrix, cur_batch))
        #     graph.selected_features = (
        #         batch_feature_ids
        #         if graph.selected_features is None
        #         else torch.cat([graph.selected_features, batch_feature_ids])
        #     )

        # return graph

        seq_len = cache_activations["hook_embed"].shape[1]

        targets: list[NodeInfoRef] = [
            NodeInfoRef(
                key="logits",
                ref=batch_logits[:, -1, :] - batch_logits[:, -1, :].mean(dim=-1, keepdim=True),
                indices=top_idx.unsqueeze(-1),
            )
        ]

        sources: list[NodeInfoRef] = [
            NodeInfoRef(
                key="hook_embed",
                ref=cache_activations["hook_embed"],
                indices=torch.arange(seq_len, device=self.device).unsqueeze(-1),
            )
        ] + [
            NodeInfoRef(
                key=hook_out + ".error",
                ref=cache_activations[hook_out + ".error"],
                indices=torch.arange(seq_len, device=self.device).unsqueeze(-1),
            )
            for replacement_module in replacement_modules
            for hook_out in replacement_module.cfg.hooks_out
        ]

        intermediates: list[tuple[NodeInfoRef, NodeInfoRef]] = [
            (
                NodeInfoRef(
                    key=hook_in + ".feature_acts",
                    ref=cache_activations[hook_in + ".feature_acts.up"],
                    indices=cache_activations[hook_in + ".feature_acts.up"][0].nonzero(),
                ),
                NodeInfoRef(
                    key=hook_in + ".feature_acts",
                    ref=cache_activations[hook_in + ".feature_acts.down"],
                    indices=cache_activations[hook_in + ".feature_acts.up"][0].nonzero(),
                ),
            )
            for replacement_module in replacement_modules
            for hook_in in replacement_module.cfg.hooks_in
        ]

        max_intermediates = max_features if max_features is not None else len(intermediates)
        max_iter = len(replacement_modules) + 10

        attribution = greedily_collect_attribution(
            targets=targets,
            sources=sources,
            intermediates=intermediates,
            max_intermediates=max_intermediates,
            reduction_weight=top_p,
            max_iter=max_iter,
        )

        return attribution

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
