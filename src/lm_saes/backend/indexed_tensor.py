from __future__ import annotations

import functools
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Callable,
    Iterator,
    Self,
    Sequence,
    cast,
    overload,
)

import torch
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate
from torch.types import Number

from lm_saes.utils.discrete import DiscreteMapper
from lm_saes.utils.distributed import DimMap, full_tensor
from lm_saes.utils.misc import tensor_id
from lm_saes.utils.timer import timer


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

    def to(self, device: torch.device | str | None = None, *, device_mesh: DeviceMesh | None = None) -> Self:
        if device is not None:
            return replace(
                self,
                indices=self.indices.to(device),
                offsets=self.offsets.to(device),
                _inv_indices=self._inv_indices.to(device) if self._inv_indices is not None else None,
            )
        elif device_mesh is not None:
            return replace(
                self,
                indices=DimMap({}).distribute(self.indices, device_mesh),
                offsets=DimMap({}).distribute(self.offsets, device_mesh),
                _inv_indices=DimMap({}).distribute(self._inv_indices, device_mesh)
                if self._inv_indices is not None
                else None,
            )
        else:
            return self

    def state_dict(self) -> dict:
        return {
            "key": self.key,
            "indices": self.indices,
            "offsets": self.offsets,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "Node":
        return cls(key=state["key"], indices=state["indices"], offsets=state["offsets"])


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

    def to(self, device: torch.device | str) -> Self:
        return replace(self, indices=self.indices.to(device))

    def full_tensor(self) -> Self:
        return replace(self, indices=full_tensor(self.indices))

    def state_dict(self) -> dict:
        return {"key": self.key, "indices": self.indices}

    @classmethod
    def from_state_dict(cls, state: dict) -> "NodeInfo":
        return cls(key=state["key"], indices=state["indices"])


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
    device_mesh: DeviceMesh | None = field(default=None, repr=False)
    _offset_mapping: dict[str, torch.Tensor] | None = field(default=None, repr=False)
    _nodes_to_offsets_cache: dict[int, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def empty(cls, device: torch.device | str, device_mesh: DeviceMesh | None = None) -> Self:
        return cls(device=device, mapper=DiscreteMapper(), node_mappings={}, device_mesh=device_mesh)

    @classmethod
    def _from_node_mappings(
        cls,
        node_mappings: dict[Any, Node],
        device: torch.device | str | None = None,
        mapper: DiscreteMapper | None = None,
        device_mesh: DeviceMesh | None = None,
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
            device_mesh=device_mesh,
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
                assert not isinstance(node.offsets, DTensor)
                offset_mapping["keys"][node.offsets] = encoded_key
                offset_mapping["indices"][node.offsets] = torch.arange(
                    node.indices.shape[0], device=self.device, dtype=torch.long
                )
            self._offset_mapping = offset_mapping
        return cast(dict[str, torch.Tensor], self._offset_mapping)

    @classmethod
    def from_node_infos(
        cls,
        node_infos: Sequence[NodeInfo],
        device: torch.device | str | None = None,
        device_mesh: DeviceMesh | None = None,
    ) -> Self:
        if len(node_infos) == 0 and device is None:
            raise ValueError("Cannot build Dimension from empty `node_infos` without an explicit device.")
        device = device if device is not None else node_infos[0].indices.device
        device_mesh = (
            device_mesh
            if device_mesh is not None
            else node_infos[0].indices.device_mesh
            if isinstance(node_infos[0].indices, DTensor)
            else None
        )

        nodes = functools.reduce(
            lambda acc, ni: (
                acc[0]
                + [
                    Node(
                        key=ni.key,
                        indices=full_tensor(ni.indices),
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
            device_mesh=device_mesh,
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
                key=unique_keys[i],
                indices=self.node_mappings[unique_keys[i]].indices[indices[inverse_indices == i]]
                if self.device_mesh is None
                else DimMap({}).from_local(
                    self.node_mappings[unique_keys[i]].indices[indices[inverse_indices == i]], self.device_mesh
                ),
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

        ret = self.__class__._from_node_mappings(
            node_mappings=node_mappings, device=self.device, mapper=self.mapper, device_mesh=self.device_mesh
        )
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

        # NotImplementedError: Operator aten.index_put_.default does not have a sharding strategy registered.
        # Set it locally as a workaround.
        for node in dimension.node_mappings.values():
            offsets[node.offsets] = self.node_mappings[node.key].offsets[
                self.node_mappings[node.key].inv_indices[node.indices.unbind(dim=1)]
            ]

        if self.device_mesh is not None:
            offsets = DimMap({}).from_local(offsets, self.device_mesh)

        self._nodes_to_offsets_cache[cache_key] = offsets
        return offsets

    @timer.time("offsets_to_nodes")
    def offsets_to_nodes(self, offsets: torch.Tensor) -> "Dimension":
        if self.device_mesh is not None:
            assert isinstance(offsets, DTensor) and all(
                isinstance(placement, Replicate) for placement in offsets.placements
            ), "Offsets must be a replicated DTensor when Dimension is distributed"
            offsets = offsets.to_local()

        keys_encoded = self.offset_mapping["keys"][offsets]
        indices = self.offset_mapping["indices"][offsets]
        if offsets.numel() == 0:
            return self.__class__.empty(device=self.device, device_mesh=self.device_mesh)
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
        return self.__class__._from_node_mappings(
            node_mappings=node_mappings, device=self.device, device_mesh=self.device_mesh, mapper=self.mapper
        )

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
            yield NodeInfo(
                key=key,
                indices=self.node_mappings[key].indices[idx : idx + 1]
                if self.device_mesh is None
                else DimMap({}).from_local(self.node_mappings[key].indices[idx : idx + 1], self.device_mesh),
            )

    def filter(self, predicate: Callable[[NodeInfo], bool]) -> Self:
        filtered = [
            NodeInfo(
                key=node.key,
                indices=node.indices
                if self.device_mesh is None
                else DimMap({}).from_local(node.indices, self.device_mesh),
            )
            for node in self.node_mappings.values()
            if predicate(
                NodeInfo(
                    key=node.key,
                    indices=node.indices
                    if self.device_mesh is None
                    else DimMap({}).from_local(node.indices, self.device_mesh),
                )
            )
        ]
        return self.__class__.from_node_infos(filtered, device=self.device, device_mesh=self.device_mesh)

    @timer.time("filter_keys")
    def filter_keys(self, predicate: Callable[[str], bool]) -> Self:
        return self.__class__.from_node_infos(
            [
                NodeInfo(
                    key=key,
                    indices=node.indices
                    if self.device_mesh is None
                    else DimMap({}).from_local(node.indices, self.device_mesh),
                )
                for key, node in self.node_mappings.items()
                if predicate(key)
            ],
            device=self.device,
            device_mesh=self.device_mesh,
        )

    @timer.time("unique")
    def unique(self) -> Self:
        return self.__class__.from_node_infos(
            [
                NodeInfo(
                    key=key,
                    indices=node.indices.unique(dim=0)
                    if self.device_mesh is None
                    else DimMap({}).from_local(node.indices.unique(dim=0), self.device_mesh),
                )
                for key, node in self.node_mappings.items()
            ],
            device=self.device,
            device_mesh=self.device_mesh,
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

    def full_tensor(self) -> Self:
        return replace(self, device_mesh=None)

    def state_dict(self) -> dict:
        """Serialize to a minimal dict — only keys, indices, and offsets per node."""
        return {
            "nodes": [node.state_dict() for node in self.node_mappings.values()],
        }

    @classmethod
    def from_state_dict(cls, state: dict, device: torch.device | str = "cpu") -> "Dimension":
        """Reconstruct from state_dict.  mapper, caches, and device_mesh are rebuilt."""
        node_mappings = {node_state["key"]: Node.from_state_dict(node_state) for node_state in state["nodes"]}
        return cls._from_node_mappings(node_mappings=node_mappings, device=device)


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
        device_mesh: DeviceMesh | None = None,
    ) -> Self:
        shape = [
            len(dimension) if isinstance(dimension, Dimension) else sum([len(ni) for ni in dimension])
            for dimension in dimensions
        ]
        return cls.from_data(
            torch.zeros(shape, dtype=dtype, device=device)
            if device_mesh is None
            else DimMap({}).distribute(torch.zeros(shape, dtype=dtype, device=device), device_mesh),
            dimensions,
        )

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
    @torch.no_grad()
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
    @torch.no_grad()
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
                if not isinstance(self.data, DTensor)
                else DimMap({}).distribute(
                    torch.arange(self.data.shape[dim], device=self.data.device, dtype=torch.long),
                    self.data.device_mesh,
                )
            )
            view_shape = [1] * self.n_dims
            view_shape[dim] = offsets.shape[0]
            indexers.append(offsets.view(*view_shape))

        if not isinstance(self.data, DTensor):
            self.data[tuple(indexers)] = value
        else:
            assert all(isinstance(placement, Replicate) for placement in self.data.placements), (
                "Only support replicate placements for now"
            )
            assert all(isinstance(placement, Replicate) for placement in cast(DTensor, value).placements), (
                "Only support replicate placements for now"
            )
            assert all(
                isinstance(placement, Replicate)
                for indexer in indexers
                for placement in cast(DTensor, indexer).placements
            ), "Only support replicate placements for now"
            self.data = DimMap({}).from_local(
                self.data.to_local().index_put(
                    tuple(cast(DTensor, indexer).to_local() for indexer in indexers), cast(DTensor, value).to_local()
                ),
                device_mesh=self.data.device_mesh,
            )
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

    def full_tensor(self) -> Self:
        return self.__class__.from_data(
            full_tensor(self.data),
            tuple(dim.full_tensor() for dim in self.dimensions),
        )

    def state_dict(self) -> dict:
        return {
            "data": self.data,
            "dimensions": [dim.state_dict() for dim in self.dimensions],
        }

    @classmethod
    def from_state_dict(cls, state: dict, device: torch.device | str = "cpu") -> Self:
        dimensions = tuple(Dimension.from_state_dict(dim_state, device=device) for dim_state in state["dimensions"])
        data = state["data"].to(device)
        return cls.from_data(data=data, dimensions=dimensions)


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
            if not isinstance(data, DTensor):
                data[ignore_indices] = float("-inf")
            else:
                assert all(isinstance(placement, Replicate) for placement in data.placements), (
                    "Only support replicate placements for now"
                )
                assert all(
                    isinstance(placement, Replicate) for placement in cast(DTensor, ignore_indices).placements
                ), "Only support replicate placements for now"
                data = DimMap({}).from_local(
                    data.to_local().index_put(
                        (cast(DTensor, ignore_indices).to_local(),),
                        torch.tensor(float("-inf"), device=data.device, dtype=data.dtype),
                    ),
                    device_mesh=data.device_mesh,
                )
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
