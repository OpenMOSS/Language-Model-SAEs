from __future__ import annotations

import functools
import json
import os
import re
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
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
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BatchFeature,
    Qwen2_5_VLForConditionalGeneration,
)

from lm_saes.backend.hooks import apply_saes, detach_at, replace_biases_with_leaves
from lm_saes.backend.tl_addons import run_with_cache_until, run_with_ref_cache
from lm_saes.config import BaseModelConfig
from lm_saes.utils.auto import PretrainedSAEType, auto_infer_pretrained_sae_type
from lm_saes.utils.discrete import DiscreteMapper
from lm_saes.utils.distributed import DimMap
from lm_saes.utils.misc import ensure_tokenized, pad_and_truncate_tokens, tensor_id
from lm_saes.utils.timer import timer

if TYPE_CHECKING:
    from lm_saes.models.lorsa import LowRankSparseAttention
    from lm_saes.models.molt import MixtureOfLinearTransform
    from lm_saes.models.sae import SparseAutoEncoder
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


@dataclass(frozen=True)
class Node:
    key: Any
    """Key of the node. Should be a hashable object."""

    indices: torch.Tensor
    """Indices of elements in the node's data. Should be of shape `(n_elements, d_index)`."""

    offsets: torch.Tensor
    """In-tensor offsets of elements. Should be of shape `(n_elements,)`."""

    _inv_indices: torch.Tensor | None = field(default=None, repr=False, compare=False)
    """Inverse indices of elements."""

    def __hash__(self) -> int:
        return hash((self.key, tensor_id(self.indices)))

    @property
    def inv_indices(self) -> torch.Tensor:
        if self._inv_indices is None:
            object.__setattr__(self, "_inv_indices", compute_inv_indices(self.indices))
        return cast(torch.Tensor, self._inv_indices)

    def __len__(self) -> int:
        return self.indices.shape[0]


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

    def unbind(self) -> list[Self]:
        """Split into a list of NodeInfo, each containing exactly one index."""
        return [replace(self, indices=self.indices[i : i + 1]) for i in range(self.indices.shape[0])]

    def __iter__(self) -> Iterator[Self]:
        """Iterate over any single element in the node info."""
        for i in range(len(self)):
            yield self[slice(i, i + 1)]


def compute_inv_indices(indices: torch.Tensor) -> torch.Tensor:
    inv_indices = torch.empty(
        [int(indices[:, i].max() + 1) for i in range(indices.shape[1])],
        device=indices.device,
        dtype=torch.long,
    )
    inv_indices[indices.unbind(dim=1)] = torch.arange(indices.shape[0], device=indices.device)
    return inv_indices


@dataclass
class Dimension:
    device: torch.device | str
    mapper: DiscreteMapper
    node_mappings: dict[Any, Node]
    _offset_mapping: dict[str, torch.Tensor] | None = field(default=None, repr=False)
    _nodes_to_offsets_cache: dict[int, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def empty(cls, device: torch.device | str) -> Self:
        return cls(device=device, mapper=DiscreteMapper(), node_mappings={})

    @classmethod
    def _from_node_mappings(
        cls,
        node_mappings: dict[Any, Node],
        device: torch.device | str | None = None,
        mapper: DiscreteMapper | None = None,
    ) -> Self:
        mapper = mapper if mapper is not None else DiscreteMapper()
        if device is None:
            if len(node_mappings) > 0:
                first_node = next(iter(node_mappings.values()))
                device = first_node.indices.device
            else:
                raise ValueError("Cannot determine device: both `device` and `node_mappings` are empty.")

        return cls(
            device=device,
            mapper=mapper,
            node_mappings=dict(node_mappings),
        )

    @property
    def offset_mapping(self) -> dict[str, torch.Tensor]:
        if self._offset_mapping is None:
            n_elements = len(self)
            offset_mapping = {
                "keys": torch.empty(n_elements, device=self.device, dtype=torch.long),
                "indices": torch.empty(n_elements, device=self.device, dtype=torch.long),
            }
            encoded_keys = self.mapper.encode(list(self.node_mappings.keys()))
            for key, encoded_key in zip(self.node_mappings.keys(), encoded_keys):
                node = self.node_mappings[key]
                offset_mapping["keys"][node.offsets] = encoded_key
                offset_mapping["indices"][node.offsets] = torch.arange(
                    node.indices.shape[0], device=self.device, dtype=torch.long
                )
            self._offset_mapping = offset_mapping
        return cast(dict[str, torch.Tensor], self._offset_mapping)

    @classmethod
    def from_node_infos(cls, node_infos: Sequence[NodeInfo], device: torch.device | str | None = None) -> Self:
        if len(node_infos) == 0 and device is None:
            raise ValueError("Cannot build Dimension from empty `node_infos` without an explicit device.")
        device = device if device is not None else node_infos[0].indices.device

        nodes = functools.reduce(
            lambda acc, ni: (
                acc[0]
                + [
                    Node(
                        key=ni.key,
                        indices=ni.indices,
                        offsets=torch.arange(acc[1], acc[1] + len(ni), device=device, dtype=torch.long),
                    )
                ],
                acc[1] + len(ni),
            ),
            node_infos,
            ([], 0),
        )[0]

        grouped = functools.reduce(
            lambda acc, node: acc | {node.key: acc.get(node.key, []) + [node]},
            nodes,
            {},
        )

        return cls._from_node_mappings(
            node_mappings={
                key: Node(
                    key=key,
                    indices=torch.cat([node.indices for node in group], dim=0),
                    offsets=torch.cat([node.offsets for node in group], dim=0),
                )
                for key, group in grouped.items()
            },
            device=device,
        )

    @property
    def node_infos(self) -> Sequence[NodeInfo]:
        if len(self) == 0:
            return []
        offsets = torch.arange(len(self), device=self.device, dtype=torch.long)
        keys_encoded = self.offset_mapping["keys"][offsets]
        indices = self.offset_mapping["indices"][offsets]
        unique_keys_encoded, inverse_indices = torch.unique_consecutive(keys_encoded, return_inverse=True)
        unique_keys = self.mapper.decode(unique_keys_encoded.tolist())
        return [
            NodeInfo(
                key=unique_keys[i], indices=self.node_mappings[unique_keys[i]].indices[indices[inverse_indices == i]]
            )
            for i in range(len(unique_keys))
        ]

    def __add__(self, other: "Dimension") -> Self:
        if not isinstance(other, Dimension):
            raise TypeError(f"Dimension can only be added with Dimension, got {type(other)}")

        if other.device != self.device:
            other = other.to(self.device)

        self_length = len(self)
        all_keys = list(dict.fromkeys([*self.node_mappings.keys(), *other.node_mappings.keys()]))
        node_mappings = {
            key: (
                Node(
                    key=key,
                    indices=torch.cat([self.node_mappings[key].indices, other.node_mappings[key].indices], dim=0),
                    offsets=torch.cat(
                        [self.node_mappings[key].offsets, other.node_mappings[key].offsets + self_length], dim=0
                    ),
                )
                if key in self.node_mappings and key in other.node_mappings
                else self.node_mappings[key]
                if key in self.node_mappings
                else Node(
                    key=key,
                    indices=other.node_mappings[key].indices,
                    offsets=other.node_mappings[key].offsets + self_length,
                    _inv_indices=other.node_mappings[key]._inv_indices,
                )
            )
            for key in all_keys
        }

        ret = self.__class__._from_node_mappings(node_mappings=node_mappings, device=self.device, mapper=self.mapper)
        assert len(ret) == len(self) + len(other), "Dimension length mismatch"
        return ret

    def __len__(self) -> int:
        return sum([len(node) for node in self.node_mappings.values()])

    def __hash__(self) -> int:
        return hash(tuple(self.node_mappings.values()))

    @timer.time("nodes_to_offsets")
    def nodes_to_offsets(self, dimension: "Dimension") -> torch.Tensor:
        cache_key = hash(dimension)
        cached_offsets = self._nodes_to_offsets_cache.get(cache_key)
        if cached_offsets is not None:
            return cached_offsets

        offsets = torch.empty(len(dimension), device=self.device, dtype=torch.long)
        for node in dimension.node_mappings.values():
            offsets[node.offsets] = self.node_mappings[node.key].offsets[
                self.node_mappings[node.key].inv_indices[node.indices.unbind(dim=1)]
            ]

        self._nodes_to_offsets_cache[cache_key] = offsets
        return offsets

    @timer.time("offsets_to_nodes")
    def offsets_to_nodes(self, offsets: torch.Tensor) -> "Dimension":
        keys_encoded = self.offset_mapping["keys"][offsets]
        indices = self.offset_mapping["indices"][offsets]
        if offsets.numel() == 0:
            return self.__class__.empty(device=self.device)
        unique_keys_encoded, inverse_indices = torch.unique(keys_encoded, sorted=False, return_inverse=True)
        unique_keys = self.mapper.decode(unique_keys_encoded.tolist())
        node_mappings = {
            key: Node(
                key=key,
                indices=self.node_mappings[key].indices[indices[inverse_indices == i]],
                offsets=(inverse_indices == i).nonzero(as_tuple=False).squeeze(-1),
            )
            for i, key in enumerate(unique_keys)
        }
        return self.__class__._from_node_mappings(node_mappings=node_mappings, device=self.device, mapper=self.mapper)

    def __iter__(self) -> Iterator[NodeInfo]:
        """Iterate over any single node in the dimension."""
        if len(self) == 0:
            return

        unique_keys_encoded = torch.unique(self.offset_mapping["keys"], sorted=False)
        unique_keys = self.mapper.decode(unique_keys_encoded.tolist())
        key_lookup = dict(zip(unique_keys_encoded.tolist(), unique_keys))

        for offset in range(len(self)):
            key = key_lookup[int(self.offset_mapping["keys"][offset].item())]
            idx = int(self.offset_mapping["indices"][offset].item())
            yield NodeInfo(key=key, indices=self.node_mappings[key].indices[idx : idx + 1])

    def filter(self, predicate: Callable[[NodeInfo], bool]) -> Self:
        filtered = [
            NodeInfo(key=node.key, indices=node.indices)
            for node in self.node_mappings.values()
            if predicate(NodeInfo(key=node.key, indices=node.indices))
        ]
        return self.__class__.from_node_infos(filtered, device=self.device)

    @timer.time("filter_keys")
    def filter_keys(self, predicate: Callable[[str], bool]) -> Self:
        return self.__class__.from_node_infos(
            [NodeInfo(key=key, indices=node.indices) for key, node in self.node_mappings.items() if predicate(key)],
            device=self.device,
        )

    @timer.time("unique")
    def unique(self) -> Self:
        return self.__class__.from_node_infos(
            [NodeInfo(key=key, indices=node.indices.unique(dim=0)) for key, node in self.node_mappings.items()],
            device=self.device,
        )

    def to(self, device: torch.device | str) -> Self:
        node_mappings = {
            key: Node(
                key=key,
                indices=node.indices.to(device),
                offsets=node.offsets.to(device),
                _inv_indices=node._inv_indices.to(device) if node._inv_indices is not None else None,
            )
            for key, node in self.node_mappings.items()
        }
        return self.__class__._from_node_mappings(node_mappings=node_mappings, device=device)


class NodeIndexedTensor:
    def __init__(
        self,
        n_dims: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.n_dims = n_dims
        self.data = torch.zeros([0] * self.n_dims, dtype=dtype, device=device)
        self.dimensions = tuple(Dimension.empty(device=device) for _ in range(self.n_dims))

    @classmethod
    def from_data(cls, data: torch.Tensor, dimensions: tuple[Sequence[NodeInfo] | Dimension, ...]) -> Self:
        self = cls.__new__(cls)
        self.n_dims = data.ndim
        self.data = data
        self.dimensions = tuple(
            dimension if isinstance(dimension, Dimension) else Dimension.from_node_infos(dimension, device=data.device)
            for dimension in dimensions
        )
        assert all(len(dimension) == data.shape[dim] for dim, dimension in enumerate(self.dimensions)), (
            "Data length must match dimension length"
        )
        return self

    @classmethod
    def from_dimensions(
        cls,
        dimensions: tuple[Sequence[NodeInfo] | Dimension, ...],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Self:
        shape = [
            len(dimension) if isinstance(dimension, Dimension) else sum([len(ni) for ni in dimension])
            for dimension in dimensions
        ]
        return cls.from_data(torch.zeros(shape, dtype=dtype, device=device), dimensions)

    @timer.time("extend")
    def extend(self, dimension: Dimension, dim: int, data: torch.Tensor | None = None):
        new_data_shape = tuple(self.data.shape[i] if i != dim else len(dimension) for i in range(self.n_dims))

        if data is None:
            data = torch.zeros(new_data_shape, dtype=self.data.dtype, device=self.data.device)
        else:
            assert data.shape == new_data_shape, (
                f"Data shape mismatches expected shape: {data.shape} != {new_data_shape}"
            )

        self.data = torch.cat(
            [self.data, data],
            dim=dim,
        )

        self.dimensions = tuple(
            self.dimensions[i] + dimension if i == dim else self.dimensions[i] for i in range(self.n_dims)
        )

    @timer.time("__getitem__")
    def __getitem__(self, key: tuple[Dimension | None, ...] | Dimension | None) -> Self:
        """Index the tensor with Dimension selections for each dimension.

        Each dimension accepts a ``Dimension`` to select specific node
        elements, or ``None`` to select all elements along that dimension.

        For a 1-D tensor a single ``Dimension`` (or ``None``) can be
        passed directly; for higher-rank tensors, pass a tuple with one entry
        per dimension.

        Args:
            key: Dimension selectors. A tuple of ``(Dimension | None)``
                with length equal to :attr:`n_dims`, or a bare
                ``Dimension | None`` for 1-D tensors.

        Returns:
            A new :class:`NodeIndexedTensor` (or subclass) containing the
            selected sub-tensor with updated node mappings.
        """
        if not isinstance(key, tuple):
            key = (key,)
        key = cast(tuple[Dimension | None, ...], key)
        if len(key) != self.n_dims:
            raise ValueError(f"Expected {self.n_dims} dimension selectors, got {len(key)}")

        data = self.data
        for dim in range(self.n_dims):
            dimension = key[dim]
            if dimension is None:
                continue

            offsets = self.dimensions[dim].nodes_to_offsets(dimension)
            data = data.index_select(dim, offsets)

        dimensions = tuple(
            dimension if dimension is not None else self.dimensions[dim] for dim, dimension in enumerate(key)
        )

        return self.__class__.from_data(data=data, dimensions=dimensions)

    @timer.time("__setitem__")
    def __setitem__(
        self,
        key: tuple[Dimension | None, ...] | Dimension | None,
        value: Self | Tensor,
    ):
        """Assign to a Dimension-selected sub-tensor.

        Args:
            key: Dimension selectors. A tuple of ``(Dimension | None)``
                with length equal to :attr:`n_dims`, or a bare
                ``Dimension | None`` for 1-D tensors.
            value: Tensor values to write to the selected region. When a
                :class:`NodeIndexedTensor` is provided, its underlying
                :attr:`data` tensor is assigned.

        Returns:
            The updated tensor instance.
        """
        if not isinstance(key, tuple):
            key = (key,)
        key = cast(tuple[Dimension | None, ...], key)
        if len(key) != self.n_dims:
            raise ValueError(f"Expected {self.n_dims} dimension selectors, got {len(key)}")

        value = cast(Tensor, value.data if isinstance(value, NodeIndexedTensor) else value)
        indexers: list[torch.Tensor] = []
        for dim in range(self.n_dims):
            dimension = key[dim]
            offsets = (
                self.dimensions[dim].nodes_to_offsets(dimension)
                if dimension is not None
                else torch.arange(self.data.shape[dim], device=self.data.device, dtype=torch.long)
            )
            view_shape = [1] * self.n_dims
            view_shape[dim] = offsets.shape[0]
            indexers.append(offsets.view(*view_shape))

        self.data[tuple(indexers)] = value
        return self

    def __add__(self, other: Self):
        data = self.data + other.data
        return self.__class__.from_data(data, self.dimensions)

    def map(self, function: Callable[[torch.Tensor], torch.Tensor]) -> Self:
        return self.__class__.from_data(function(self.data), self.dimensions)

    def clone(self) -> Self:
        return self.__class__.from_data(self.data.clone(), self.dimensions)

    def __and__(self, other: Self) -> Self:
        return self.__class__.from_data(self.data & other.data, self.dimensions)

    def __invert__(self) -> Self:
        return self.__class__.from_data(~self.data, self.dimensions)

    @timer.time("nonzero")
    def nonzero(self) -> tuple[torch.Tensor, tuple[Dimension, ...]]:
        indices = self.data.nonzero(as_tuple=True)
        values = self.data[indices]
        return values, tuple(self.dimensions[i].offsets_to_nodes(indices[i]) for i in range(self.n_dims))

    def to(self, device: torch.device | str) -> Self:
        return self.__class__.from_data(self.data.to(device), tuple(dim.to(device) for dim in self.dimensions))


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

    def add_nodes(self, node_infos: Dimension, data: torch.Tensor | None = None):
        self.extend(node_infos, 0, data)

    def topk(self, k: int, ignore_dimension: Dimension | None = None):
        ignore_indices = (
            self.dimensions[0].nodes_to_offsets(ignore_dimension)
            if ignore_dimension is not None and len(ignore_dimension) > 0
            else None
        )
        data = self.data
        if ignore_indices is not None:
            data = data.clone()
            data[ignore_indices] = float("-inf")
        topk_values, topk_indices = torch.topk(data, k=k, dim=0)
        return topk_values, self.dimensions[0].offsets_to_nodes(topk_indices)

    @overload
    def matmul(self, other: NodeIndexedMatrix, _check_node_matching: bool = False) -> NodeIndexedVector: ...
    @overload
    def matmul(self, other: NodeIndexedVector, _check_node_matching: bool = False) -> Number: ...

    def matmul(self, other: NodeIndexedMatrix | NodeIndexedVector, _check_node_matching: bool = False):
        # if _check_node_matching:
        #     a, b = self.node_infos[0], other.node_infos[0]
        #     if len(a) != len(b) or any(not ai == bi for ai, bi in zip(a, b)):
        #         raise ValueError(f"Node matching failed: {a} != {b}")

        data = self.data @ other.data

        if isinstance(other, NodeIndexedMatrix):
            return NodeIndexedVector.from_data(data, dimensions=(other.dimensions[1],))
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

    def add_targets(self, node_infos: Dimension, data: torch.Tensor | None = None):
        self.extend(node_infos, 0, data)

    def add_sources(self, node_infos: Dimension, data: torch.Tensor | None = None):
        self.extend(node_infos, 1, data)

    @overload
    def matmul(self, other: NodeIndexedVector, _check_node_matching: bool = False) -> NodeIndexedVector: ...
    @overload
    def matmul(self, other: NodeIndexedMatrix, _check_node_matching: bool = False) -> NodeIndexedMatrix: ...

    def matmul(self, other: NodeIndexedVector | NodeIndexedMatrix, _check_node_matching: bool = False):
        # if _check_node_matching:
        #     a, b = self.dimensions[1], other.dimensions[0]
        #     if len(a) != len(b) or any(not ai == bi for ai, bi in zip(a, b)):
        #         raise ValueError(f"Node matching failed: {a} != {b}")

        data = self.data @ other.data

        if isinstance(other, NodeIndexedVector):
            return NodeIndexedVector.from_data(data, dimensions=(self.dimensions[0],))
        elif isinstance(other, NodeIndexedMatrix):
            return NodeIndexedMatrix.from_data(data, dimensions=(self.dimensions[0], other.dimensions[1]))
        else:
            raise ValueError(f"Invalid type as right operand in NodeIndexedMatrix.matmul: {type(other)}")

    def any(self, dim: int) -> NodeIndexedVector:
        """Reduce along *dim* with logical OR, mirroring ``torch.Tensor.any(dim)``.

        ``dim=0`` reduces target (row) nodes and returns a vector over source (col)
        nodes; ``dim=1`` does the reverse.  To restrict to a subset of nodes, index
        the matrix first: ``matrix[None, subset].any(0)``.
        """
        return NodeIndexedVector.from_data(self.data.any(dim), (self.dimensions[1 - dim],))

    @timer.time("masked_fill_dim_")
    def masked_fill_dim_(self, dim: int, mask: NodeIndexedVector, value: Number) -> Self:
        offsets = self.dimensions[dim].nodes_to_offsets(mask.dimensions[0])
        filled = offsets[mask.data]
        if filled.numel() > 0:
            if dim == 0:
                self.data[filled, :] = value
            else:
                self.data[:, filled] = value
        return self

    @timer.time("masked_fill_")
    def masked_fill_(self, mask: NodeIndexedMatrix, value: Number) -> Self:
        self.data.masked_fill_(mask.data, value)
        return self

    @overload
    def __matmul__(self, other: NodeIndexedVector) -> NodeIndexedVector: ...
    @overload
    def __matmul__(self, other: NodeIndexedMatrix) -> NodeIndexedMatrix: ...

    def __matmul__(self, other: NodeIndexedVector | NodeIndexedMatrix):
        return self.matmul(other)


@timer.time("_find_influence_threshold")
def _find_influence_threshold(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Find score threshold that keeps the desired fraction of total influence."""
    if scores.numel() == 0:
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    sorted_scores = torch.sort(scores.view(-1), descending=True).values
    cumulative_score = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores).clamp(min=1e-8)
    threshold_index = torch.searchsorted(cumulative_score, threshold)
    threshold_index = min(int(threshold_index.item()), len(cumulative_score) - 1)
    return sorted_scores[threshold_index]


@timer.time("prune_attribution")
def prune_attribution(
    attribution: NodeIndexedMatrix,
    logit_weights: torch.Tensor,
    node_threshold: float = 0.6,
    edge_threshold: float = 0.8,
) -> NodeIndexedMatrix:
    """Prune an attribution NodeIndexedMatrix by removing low-influence nodes and edges.

    The attribution matrix is expected to have:
    - dim 0 (targets/rows): logit nodes (key="logits") + collected feature nodes
    - dim 1 (sources/cols): embed nodes (key="hook_embed") + error nodes (key ends with ".error")
                             + all (possibly uncollected) feature nodes

    Logit nodes and embed/error source nodes are always kept. Feature nodes are pruned based
    on their cumulative contribution to the weighted logit output.

    Args:
        attribution: NodeIndexedMatrix from the attribution computation.
        node_threshold: Retain feature nodes accounting for this fraction of total influence.
        edge_threshold: Retain edges accounting for this fraction of total edge influence.
        logit_weights: Per-logit scalar weights (shape ``[n_logits]``).

    Returns:
        Pruned NodeIndexedMatrix containing only kept nodes and edges.
    """
    if node_threshold > 1.0 or node_threshold < 0.0:
        raise ValueError("node_threshold must be between 0.0 and 1.0")
    if edge_threshold > 1.0 or edge_threshold < 0.0:
        raise ValueError("edge_threshold must be between 0.0 and 1.0")

    logits_dimension = attribution.dimensions[0].filter_keys(lambda key: key == "logits")
    intermediates_dimension = attribution.dimensions[0].filter_keys(lambda key: key != "logits")
    optional_sources_dimension = (
        attribution.dimensions[1].filter_keys(lambda key: key.endswith(".error")) + intermediates_dimension
    )

    edge_scores = compute_intermediates_attribution(
        attribution, attribution.dimensions[0], intermediates_dimension, max_iter=100
    )
    node_scores = (
        NodeIndexedVector.from_data(logit_weights, dimensions=(logits_dimension,)) @ edge_scores[logits_dimension, None]
    )

    node_mask = node_scores.map(lambda x: x >= _find_influence_threshold(x, node_threshold))
    edge_mask = edge_scores.map(lambda x: x >= _find_influence_threshold(x, edge_threshold))

    old_node_mask = node_mask.clone()
    node_mask[optional_sources_dimension] = node_mask[optional_sources_dimension] & edge_mask[
        None, optional_sources_dimension
    ].any(0)
    node_mask[intermediates_dimension] = node_mask[intermediates_dimension] & edge_mask[
        intermediates_dimension, None
    ].any(1)

    while not torch.equal(node_mask.data, old_node_mask.data):
        old_node_mask = node_mask.clone()
        edge_mask.masked_fill_dim_(1, ~node_mask[optional_sources_dimension], False)
        edge_mask.masked_fill_dim_(0, ~node_mask[intermediates_dimension], False)
        node_mask[optional_sources_dimension] = node_mask[optional_sources_dimension] & edge_mask[
            None, optional_sources_dimension
        ].any(0)
        node_mask[intermediates_dimension] = node_mask[intermediates_dimension] & edge_mask[
            intermediates_dimension, None
        ].any(1)

    attribution = attribution.clone()
    attribution.masked_fill_dim_(1, ~node_mask[optional_sources_dimension], 0)
    attribution.masked_fill_dim_(0, ~node_mask[intermediates_dimension], 0)
    attribution.masked_fill_(~edge_mask, 0)
    return attribution


@dataclass
class NodeInfoRef(NodeInfo):
    """NodeInfo with reference to node (tensor) in computation graph."""

    ref: torch.Tensor


class NodeInfoQueue[T: NodeInfo]:
    def __init__(self, node_infos: Sequence[T] = []):
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


@dataclass
class AttributionResult:
    activations: NodeIndexedVector
    attribution: NodeIndexedMatrix
    logits: torch.Tensor
    probs: torch.Tensor
    prompt_token_ids: list[int] = field(default_factory=list)
    prompt_tokens: list[str] = field(default_factory=list)
    logit_token_ids: list[int] = field(default_factory=list)
    logit_tokens: list[str] = field(default_factory=list)


def get_normalized_matrix(matrix: NodeIndexedMatrix) -> NodeIndexedMatrix:
    return NodeIndexedMatrix.from_data(
        data=torch.abs(matrix.data) / torch.abs(matrix.data).sum(dim=1, keepdim=True).clamp(min=1e-8),
        dimensions=matrix.dimensions,
    )


@timer.time("compute_intermediates_attribution")
def compute_intermediates_attribution(
    attribution: NodeIndexedMatrix,
    targets: Dimension,
    intermediates: Dimension,
    max_iter: int,
) -> NodeIndexedMatrix:
    attribution = get_normalized_matrix(attribution)
    influence = attribution[targets, None]
    if len(intermediates) == 0:
        return influence
    t2i: NodeIndexedMatrix = attribution[targets, intermediates]
    i2all: NodeIndexedMatrix = attribution[intermediates, None]
    i2i: NodeIndexedMatrix = attribution[intermediates, intermediates]
    for _ in range(max_iter):
        cur_influence = t2i @ i2all
        if not torch.any(cur_influence.data):
            break
        influence += cur_influence
        t2i = t2i @ i2i
    return influence


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


def retrieval_from_intermediates(
    node_infos: Sequence[NodeInfo] | Dimension, intermediates: Sequence[tuple[NodeInfoRef, NodeInfoRef]]
):
    return [
        NodeInfoRef(key=node_info.key, indices=node_info.indices, ref=intermediate[0].ref)
        for node_info in node_infos
        for intermediate in intermediates
        if node_info.key == intermediate[0].key
    ]


@timer.profile("greedily_collect_attribution")
def greedily_collect_attribution(
    targets: Sequence[NodeInfoRef],
    sources: Sequence[NodeInfoRef],
    intermediates: Sequence[tuple[NodeInfoRef, NodeInfoRef]],  # [up as target, down as source]
    max_intermediates: int,
    reduction_weight: torch.Tensor,
    max_iter: int = 100,
) -> tuple[NodeIndexedMatrix, Dimension]:
    """
    Greedily collect attribution from targets to sources through intermediates.
    """

    all_sources = list(sources) + [intermediate[1] for intermediate in intermediates]

    targets_dimension = Dimension.from_node_infos(targets)
    all_sources_dimension = Dimension.from_node_infos(all_sources)
    source_intermediates_dimension = Dimension.from_node_infos([intermediate[1] for intermediate in intermediates])
    attribution = NodeIndexedMatrix.from_dimensions(
        dimensions=(targets_dimension, all_sources_dimension),
        device=targets[0].ref.device,
        dtype=targets[0].ref.dtype,
    )

    batch_size = targets[0].ref.shape[0]

    queue = NodeInfoQueue(targets)

    for target_batch in queue.iter(batch_size):
        clear_grads(all_sources)
        root = torch.diag(torch.cat(values(target_batch), dim=1))

        with timer.time("backward"):
            root.sum().backward(retain_graph=True)

        attribution[Dimension.from_node_infos(target_batch), None] = torch.cat(
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

    collected_intermediates_dimension = Dimension.empty(device=targets[0].ref.device)
    reduction_weight: NodeIndexedVector = NodeIndexedVector.from_data(reduction_weight, dimensions=(targets_dimension,))
    for i in tqdm(range(0, max_intermediates, batch_size)):
        cur_batch_size = min(batch_size, max_intermediates - i)
        intermediates_attribution = compute_intermediates_attribution(
            attribution, targets_dimension, collected_intermediates_dimension, max_iter
        )

        influence = reduction_weight @ intermediates_attribution[None, source_intermediates_dimension]

        _, selected_nodes = influence.topk(k=cur_batch_size, ignore_dimension=collected_intermediates_dimension)

        collected_intermediates_dimension = collected_intermediates_dimension + selected_nodes

        clear_grads(all_sources)
        node_refs = retrieval_from_intermediates(selected_nodes, intermediates)
        root = torch.diag(torch.cat(values(node_refs), dim=1))

        with timer.time("backward"):
            root.sum().backward(retain_graph=True)

        attribution.add_targets(
            selected_nodes,
            torch.cat(
                [
                    einops.einsum(
                        value[: root.shape[0]],
                        grad[: root.shape[0]],
                        "batch n_elements ..., batch n_elements ... -> batch n_elements",
                    )
                    for value, grad in zip(values(all_sources), grads(all_sources))
                ],
                dim=1,
            ),
        )

    return attribution, collected_intermediates_dimension


def ln_detach_hooks(models: TransformerLensLanguageModel) -> list[str]:
    assert models.model is not None, "model must be initialized"
    detach_hooks = []
    for i, block in enumerate(models.model.blocks):
        for module_name in ["ln1", "ln2", "ln1_post", "ln2_post"]:
            if hasattr(block, module_name) and isinstance(getattr(block, module_name), torch.nn.Module):
                detach_hooks.append(f"blocks.{i}.{module_name}.hook_scale")

    detach_hooks.append("ln_final.hook_scale")
    return detach_hooks


def attn_detach_hooks(models: TransformerLensLanguageModel) -> list[str]:
    assert models.model is not None, "model must be initialized"
    detach_hooks = []
    for i, block in enumerate(models.model.blocks):
        if hasattr(block, "attn") and isinstance(block.attn, torch.nn.Module):
            detach_hooks.append(f"blocks.{i}.attn.hook_pattern")
            if models.model.cfg.use_qk_norm:
                detach_hooks.append(f"blocks.{i}.attn.q_norm.hook_scale")
                detach_hooks.append(f"blocks.{i}.attn.k_norm.hook_scale")
    return detach_hooks


@dataclass
class QKTraceRequest:
    lorsa: LowRankSparseAttention
    head_idx: int
    q_pos: int
    k_pos: int

    @property
    def dedup_key(self) -> tuple[str, int, int, int]:
        return (self.lorsa.cfg.hook_point_out, self.head_idx, self.q_pos, self.k_pos)

    @classmethod
    def from_lorsa_feature(cls, node_info: NodeInfo, lorsa: LowRankSparseAttention, attn_scores: torch.Tensor):
        head_idx = node_info.indices[0][1] // lorsa.cfg.ov_group_size
        attn_score = attn_scores[head_idx, 0, :, :]  # (q_pos, k_pos)
        q_pos, k_pos = torch.unravel_index(attn_score.argmax(), attn_score.shape)
        return cls(lorsa, int(head_idx), int(q_pos.item()), int(k_pos.item()))


@dataclass
class QKTraceResult:
    nodes: tuple[NodeInfo, NodeInfo]
    attribution: float


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

    @timer.profile("collect_cache")
    def _collect_cache(
        self,
        inputs: torch.Tensor | str,
        replacement_modules: list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform],
        with_bias_leaves: bool = False,
    ):
        from lm_saes.models.lorsa import LowRankSparseAttention
        from lm_saes.models.sparse_dictionary import SparseDictionary

        assert self.model is not None, "model must be initialized"
        tokens = ensure_tokenized(inputs, self.tokenizer, device=self.device)

        from contextlib import nullcontext

        bias_ctx = (
            replace_biases_with_leaves(
                self.model,
                cast(list[SparseDictionary], replacement_modules),
                batch_size=tokens.shape[0],
                seq_len=tokens.shape[1],
            )
            if with_bias_leaves
            else nullcontext({})
        )

        with self.apply_saes(cast(list[SparseDictionary], replacement_modules)):
            with bias_ctx as bias_leaves:
                with self.detach_at(
                    ["hook_embed"]
                    + [replacement_module.cfg.hook_point_out + ".error" for replacement_module in replacement_modules]
                    + [
                        replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts"
                        for replacement_module in replacement_modules
                    ]
                    + [
                        replacement_module.cfg.hook_point_out + ".sae.hook_attn_pattern"
                        for replacement_module in replacement_modules
                        if isinstance(replacement_module, LowRankSparseAttention)
                    ]
                    + [
                        replacement_module.cfg.hook_point_out + item
                        for replacement_module in replacement_modules
                        if isinstance(replacement_module, LowRankSparseAttention)
                        and replacement_module.cfg.use_post_qk_ln
                        for item in (".sae.ln_q.hook_scale", ".sae.ln_k.hook_scale")
                    ]
                    + ln_detach_hooks(self)
                    + attn_detach_hooks(self)
                ):
                    logits, cache = self.run_with_ref_cache(
                        tokens,
                        names_filter=["hook_embed.post"]
                        + [
                            replacement_module.cfg.hook_point_out + ".error.post"
                            for replacement_module in replacement_modules
                        ]
                        + [
                            replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts.pre"
                            for replacement_module in replacement_modules
                        ]
                        + [
                            replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts.post"
                            for replacement_module in replacement_modules
                        ]
                        + [
                            replacement_module.cfg.hook_point_out + ".sae.hook_attn_score"
                            for replacement_module in replacement_modules
                            if isinstance(replacement_module, LowRankSparseAttention)
                        ]
                        + ln_detach_hooks(self),
                    )

        cache.update(bias_leaves)
        return logits, cache

    @timer.profile("attribute")
    def attribute(
        self,
        inputs: torch.Tensor | str,
        replacement_modules: list[SparseDictionary],
        max_n_logits: int = 10,
        desired_logit_prob: float = 0.95,
        batch_size: int = 512,
        max_features: int | None = None,
    ):
        from lm_saes.models.lorsa import LowRankSparseAttention
        from lm_saes.models.molt import MixtureOfLinearTransform
        from lm_saes.models.sae import SparseAutoEncoder

        tokens = ensure_tokenized(inputs, self.tokenizer, device=self.device)
        replacement_modules: list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform] = cast(
            list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform], replacement_modules
        )
        batch_logits, cache = self._collect_cache(einops.repeat(tokens, "n -> b n", b=batch_size), replacement_modules)

        with torch.no_grad():
            probs = torch.softmax(batch_logits[0, -1], dim=-1)
            top_p, top_idx = torch.topk(probs, max_n_logits)
            cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
            top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

        seq_len = cache["hook_embed.post"].shape[1]

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
                ref=cache["hook_embed.post"],
                indices=torch.arange(seq_len, device=self.device).unsqueeze(-1),
            )
        ] + [
            NodeInfoRef(
                key=replacement_module.cfg.hook_point_out + ".error",
                ref=cache[replacement_module.cfg.hook_point_out + ".error.post"],
                indices=torch.arange(seq_len, device=self.device).unsqueeze(-1),
            )
            for (replacement_module) in replacement_modules
        ]

        intermediates: list[tuple[NodeInfoRef, NodeInfoRef]] = [
            (
                NodeInfoRef(
                    key=replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts",
                    ref=cache[replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts.pre"],
                    indices=cache[replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts.pre"][0].nonzero(),
                ),
                NodeInfoRef(
                    key=replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts",
                    ref=cache[replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts.post"],
                    indices=cache[replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts.post"][0].nonzero(),
                ),
            )
            for replacement_module in replacement_modules
        ]

        max_intermediates = max_features if max_features is not None else len(intermediates)
        max_iter = len(replacement_modules) + 10

        attribution, collected_intermediates = greedily_collect_attribution(
            targets=targets,
            sources=sources,
            intermediates=intermediates,
            max_intermediates=max_intermediates,
            reduction_weight=top_p,
            max_iter=max_iter,
        )

        sources_dimension = Dimension.from_node_infos(sources)
        attribution = attribution[None, sources_dimension + collected_intermediates]

        intermediate_ref_map = {node_info.key: node_info.ref for node_info, _ in intermediates}
        activations = torch.cat(
            [node_info.ref[0, *node_info.indices.unbind(dim=1)] for node_info in targets]
            + [
                intermediate_ref_map[node_info.key][0, *node_info.indices.unbind(dim=1)]
                for node_info in collected_intermediates
            ]
            + [torch.ones_like(node_info.indices[:, 0], dtype=node_info.ref.dtype) for node_info in sources],
            dim=0,
        )

        activations = NodeIndexedVector.from_data(
            data=activations,
            dimensions=(
                Dimension.from_node_infos(targets) + collected_intermediates + Dimension.from_node_infos(sources),
            ),
        )

        prompt_token_ids = tokens.detach().cpu().tolist()
        logit_token_ids = top_idx.detach().cpu().tolist()

        return AttributionResult(
            activations=activations,
            attribution=attribution,
            logits=batch_logits[:, -1, top_idx],
            probs=top_p,
            prompt_token_ids=prompt_token_ids,
            prompt_tokens=[self.tokenizer.decode([token_id]) for token_id in prompt_token_ids],
            logit_token_ids=logit_token_ids,
            logit_tokens=[self.tokenizer.decode([token_id]) for token_id in logit_token_ids],
        )

    def qk_trace(
        self,
        inputs: torch.Tensor | str,
        replacement_modules: list[SparseDictionary],
        lorsa_features: list[NodeInfo],
        topk: int = 10,
        batch_size: int = 1,
    ):
        from lm_saes.models.lorsa import LowRankSparseAttention
        from lm_saes.models.molt import MixtureOfLinearTransform
        from lm_saes.models.sae import SparseAutoEncoder

        tokens = ensure_tokenized(inputs, self.tokenizer, device=self.device)
        replacement_modules: list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform] = cast(
            list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform], replacement_modules
        )
        _, cache = self._collect_cache(
            einops.repeat(tokens, "n -> b n", b=batch_size * topk), replacement_modules, with_bias_leaves=True
        )
        # print(cache["blocks.24.hook_attn_out.sae.hook_feature_acts.post"][0][2].nonzero())
        rm_mapping = {
            replacement_module.cfg.hook_point_out: replacement_module for replacement_module in replacement_modules
        }
        requests = [
            QKTraceRequest.from_lorsa_feature(
                lorsa_feature,
                cast(LowRankSparseAttention, rm_mapping[lorsa_feature.key]),
                cache[lorsa_feature.key + ".sae.hook_attn_score"],
            )
            for lorsa_feature in lorsa_features
        ]

        unique_map: dict[tuple[str, int, int, int], int] = {}
        unique_requests: list[QKTraceRequest] = []
        request_indices: list[int] = []
        for req in requests:
            key = req.dedup_key
            if key not in unique_map:
                unique_map[key] = len(unique_requests)
                unique_requests.append(req)
            request_indices.append(unique_map[key])

        unique_results = self._qk_trace_from_request(unique_requests, cache, replacement_modules, topk)
        return [unique_results[idx] for idx in request_indices]

    def _qk_trace_from_request(
        self,
        requests: list[QKTraceRequest],
        cache: dict[str, torch.Tensor],
        replacement_modules: list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform],
        topk: int,
    ) -> list[list[QKTraceResult]]:  # pyright: ignore[reportReturnType]
        seq_len = cache["hook_embed.post"].shape[1]
        fwd_batch_size = cache["hook_embed.post"].shape[0]
        bwd_batch_size = fwd_batch_size // topk
        pos_indices = torch.arange(seq_len, device=self.device).unsqueeze(-1)
        sources: list[NodeInfoRef] = (
            [
                NodeInfoRef(
                    key="hook_embed",
                    ref=cache["hook_embed.post"],
                    indices=pos_indices,
                )
            ]
            + [
                NodeInfoRef(
                    key=replacement_module.cfg.hook_point_out + ".error",
                    ref=cache[replacement_module.cfg.hook_point_out + ".error.post"],
                    indices=pos_indices,
                )
                for replacement_module in replacement_modules
            ]
            + [
                NodeInfoRef(
                    key=replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts",
                    ref=cache[replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts.post"],
                    indices=cache[replacement_module.cfg.hook_point_out + ".sae.hook_feature_acts.post"][0].nonzero(),
                )
                for replacement_module in replacement_modules
            ]
        )
        assert self.model is not None
        for i in range(len(self.model.blocks)):
            for key in (f"blocks.{i}.attn.b_O", f"blocks.{i}.mlp.b_out"):
                if key in cache:
                    sources.append(NodeInfoRef(key=key, ref=cache[key], indices=pos_indices))
        for replacement_module in replacement_modules:
            hp = replacement_module.cfg.hook_point_out
            for suffix in (".sae.b_Q", ".sae.b_K", ".sae.b_D"):
                key = hp + suffix
                if key in cache:
                    sources.append(NodeInfoRef(key=key, ref=cache[key], indices=pos_indices))

        sources_dimension = Dimension.from_node_infos(sources)
        results = []
        for batch_start in range(0, len(requests), bwd_batch_size):
            request_batch = requests[batch_start : batch_start + bwd_batch_size]
            clear_grads(sources)
            bwd_scores = torch.stack(
                [
                    cache[request.lorsa.cfg.hook_point_out + ".sae.hook_attn_score"][
                        request.head_idx, batch_idx * topk : (batch_idx + 1) * topk, request.q_pos, request.k_pos
                    ]
                    for batch_idx, request in enumerate(request_batch)
                ],
                dim=0,
            )
            bwd_scores.sum().backward(create_graph=True, retain_graph=True)
            first_order_gradients = torch.cat(
                [
                    einops.einsum(
                        value.detach(),
                        grad,
                        "batch n_elements ..., batch n_elements ... -> batch n_elements",
                    )
                    for value, grad in zip(values(sources), grads(sources))
                ],
                dim=1,
            )  # fwd_batch_size, n_sources
            second_sources = []
            all_topk_indices = []
            for batch_idx in range(len(request_batch)):
                _, topk_indices = first_order_gradients[batch_idx * topk].topk(
                    topk
                )  # get the first row of each request
                second_sources.extend(sources_dimension.offsets_to_nodes(topk_indices))
                all_topk_indices.append(topk_indices)

            all_topk_indices = torch.stack(all_topk_indices, dim=0)  # (len(request_batch), topk)
            batch_row_indices = torch.arange(len(request_batch), device=self.device).unsqueeze(1) * topk + torch.arange(
                topk, device=self.device
            ).unsqueeze(0)  # (len(request_batch), topk)
            second_bwd_values = first_order_gradients[batch_row_indices, all_topk_indices].reshape(-1)
            clear_grads(sources)
            second_bwd_values.sum().backward(retain_graph=True)

            second_order_gradients = torch.cat(
                [
                    einops.einsum(
                        value.detach(),
                        grad,
                        "batch n_elements ..., batch n_elements ... -> batch n_elements",
                    )
                    for value, grad in zip(values(sources), grads(sources))
                ],
                dim=1,
            )  # fwd_batch_size, n_sources

            for batch_idx in range(len(request_batch)):
                batch_results = []
                for k_idx in range(topk):
                    idx = batch_idx * topk + k_idx
                    topk_values, topk_indices = second_order_gradients[idx].topk(topk)
                    topk_nodes = list(sources_dimension.offsets_to_nodes(topk_indices))
                    batch_results.append(
                        QKTraceResult(
                            nodes=(second_sources[idx], topk_nodes[k_idx]),
                            attribution=topk_values[k_idx].item(),
                        )
                    )

                batch_results.sort(key=lambda x: x.attribution, reverse=True)
                results.append(batch_results[:topk])

        return results

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

    @contextmanager
    def apply_saes(self, saes: list[SparseDictionary]):
        assert self.model is not None, "model must be initialized"
        with apply_saes(self.model, saes):
            yield self

    @contextmanager
    def detach_at(self, hook_points: list[str]):
        assert self.model is not None, "model must be initialized"
        with detach_at(self.model, hook_points):
            yield self

    def run_with_ref_cache(self, *args, **kwargs) -> Any:
        assert self.model is not None, "model must be initialized"
        if self.device_mesh is None:
            return run_with_ref_cache(self.model, *args, **kwargs)

        args = pytree.tree_map(self._to_tensor, args)
        kwargs = pytree.tree_map(self._to_tensor, kwargs)
        return pytree.tree_map(self._to_dtensor, run_with_ref_cache(self.model, *args, **kwargs))

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
