import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from lm_saes.activation.processors.core import BaseActivationProcessor
from lm_saes.utils.misc import all_gather_dict, is_master
from lm_saes.utils.tensor_dict import move_dict_of_tensor_to_device


@dataclass
class ChunkInfo:
    """Information about a cached activation chunk file.

    Args:
        path: Path to the chunk file
        shard_id: Shard ID extracted from filename (0 if no shard)
        chunk_id: Chunk ID extracted from filename
    """

    path: Path
    shard_id: int
    chunk_id: int

    @classmethod
    def from_path(cls, path: Path) -> "ChunkInfo":
        """Create ChunkInfo from a file path.

        Supports formats:
        - shard-{shard_id}-chunk-{chunk_id}.(pt|safetensors)
        - chunk-{chunk_id}.(pt|safetensors)

        Args:
            path: Path to chunk file

        Returns:
            ChunkInfo object with parsed shard and chunk IDs

        Raises:
            ValueError: If filename doesn't match either expected pattern
        """
        # Try shard-chunk format first
        match = re.match(r"shard-(\d+)-chunk-(\d+)\.(pt|safetensors)", path.name)
        if match:
            return cls(path=path, shard_id=int(match.group(1)), chunk_id=int(match.group(2)))

        # Try chunk-only format
        match = re.match(r"chunk-(\d+)\.(pt|safetensors)", path.name)
        if match:
            return cls(
                path=path,
                shard_id=0,  # Default shard ID for non-sharded files
                chunk_id=int(match.group(1)),
            )

        raise ValueError(
            f"Invalid chunk filename format: {path.name}. "
            "Expected 'shard-{N}-chunk-{N}.(pt|safetensors)' or 'chunk-{N}.(pt|safetensors)'"
        )


def first_data_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return batch[0]


class CachedActivationLoader(BaseActivationProcessor[None, Iterable[dict[str, Any]]]):
    """Base class for cached activation loaders.

    Args:
        cache_dir: Root directory containing cached activations
        hook_points: List of hook point names to load
        device: Device to load tensors to
        num_workers: Number of worker processes for data loading. Default is 4
        prefetch_factor: Number of samples loaded in advance by each worker. Default is 8
        distributed: Whether to use distributed batch loading and broadcasting
    """

    def __init__(
        self,
        cache_dirs: Mapping[str, str | Path],
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        device_mesh: Optional[DeviceMesh] = None,
    ):
        self.cache_dirs = {k: Path(v) for k, v in cache_dirs.items()}
        self.device = device
        self.dtype = dtype
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.device_mesh = device_mesh
        # Distributed setup
        if self.device_mesh is not None and not dist.is_initialized():
            raise RuntimeError("Distributed training must be initialized before using distributed loading")

    def load_single_hook_chunk(
        self, chunk_idx: int, hook_point: str, hook_chunks: dict[str, list[ChunkInfo]]
    ) -> dict[str, Any]:
        """Load chunk data for a single hook point at given index.

        Args:
            chunk_idx: Index of the chunk to load
            hook_point: Name of the hook point to load
            hook_chunks: Dictionary mapping hook points to their chunk info lists

        Returns:
            dict[str, Any]: Chunk data for the specific hook point
        """
        chunk = hook_chunks[hook_point][chunk_idx]
        data: dict[str, Any] = self._load_chunk(chunk.path)

        # Validate data format
        assert isinstance(data, dict), f"Loading cached activation {chunk.path} error: returned {type(data)}"
        assert "activation" in data, f"Loading cached activation {chunk.path} error: missing 'activation' field"
        assert "tokens" in data, f"Loading cached activation {chunk.path} error: missing 'tokens' field"

        return {k: v for k, v in data.items()} | {
            "hook_point": hook_point,
            "chunk_idx": chunk_idx,
            "meta": data.get("meta"),
        }

    def _get_sorted_chunks(self, hook_point: str) -> list[ChunkInfo]:
        """Get sorted list of chunk files for a hook point.

        Args:
            hook_point: Name of the hook point

        Returns:
            List of ChunkInfo objects sorted by shard ID then chunk ID

        Raises:
            FileNotFoundError: If hook point directory doesn't exist
        """
        hook_dir = self.cache_dirs[hook_point]
        if not hook_dir.exists():
            raise FileNotFoundError(f"Hook point directory not found: {hook_dir}")

        # Get both shard-chunk and chunk-only files, supporting both .pt and .safetensors
        chunks = [
            ChunkInfo.from_path(p)
            for pattern in ["shard-*-chunk-*.pt", "shard-*-chunk-*.safetensors", "chunk-*.pt", "chunk-*.safetensors"]
            for p in hook_dir.glob(pattern)
        ]
        return sorted(chunks, key=lambda x: (x.shard_id, x.chunk_id))

    def _load_chunk(self, chunk_path: Path) -> dict[str, Any]:
        """Load a chunk file, supporting both PyTorch and safetensors formats.

        Args:
            chunk_path: Path to the chunk file

        Returns:
            dict[str, Any]: Loaded data containing activations, tokens, and meta
        """
        if chunk_path.suffix == ".safetensors":
            chunk_data: dict[str, Any] = load_file(chunk_path, device="cpu")
            meta_path = chunk_path.with_suffix(".meta.json")
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                chunk_data = chunk_data | {"meta": meta}
            return chunk_data
        elif chunk_path.suffix == ".pt":
            return torch.load(chunk_path, map_location="cpu", weights_only=True)
        else:
            raise ValueError(f"Invalid chunk file format: {chunk_path}. Expected .safetensors or .pt")

    def _process_chunks(self, hook_chunks: dict[str, list[ChunkInfo]], num_chunks: int) -> Iterator[dict[str, Any]]:
        """Process chunks using the appropriate method based on distributed setting.

        Args:
            hook_chunks: Dictionary mapping hook points to their chunk info lists
            num_chunks: Total number of chunks

        Returns:
            Iterator over chunk data
        """
        cached_activation_dataset = CachedActivationDataset(
            self,
            hook_chunks,
            num_chunks,
        )
        dataloader = DataLoader(
            cached_activation_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            collate_fn=first_data_collate_fn,
            sampler=DistributedSampler(
                cached_activation_dataset,
                num_replicas=self.device_mesh.get_group().size(),
                rank=self.device_mesh.get_group().rank(),
                shuffle=False,
            )
            if self.device_mesh is not None  # abundant condition to make mypy happy
            else None,
        )

        if not self.device_mesh:
            yield from tqdm(
                dataloader,
                total=len(cached_activation_dataset),
                desc="Processing activation chunks",
                disable=not is_master(),
            )
        else:
            for data in tqdm(
                dataloader,
                total=len(cached_activation_dataset),
                desc="Processing activation chunks",
                disable=not is_master(),
            ):
                assert self.device_mesh is not None
                # Use all_gather_dict to gather chunk dicts from model parallel group
                gathered = all_gather_dict(data, group=self.device_mesh.get_group(mesh_dim="model"))
                yield from gathered

    def process(self, data: None = None, **kwargs) -> Iterable[dict[str, Any]]:
        """Load cached activations in a streaming fashion.

        Yields dictionaries containing:
            - Activations for each hook point
            - Original tokens
            - Original info field

        The activations is of the shape (n_context, d_model).
        Files are loaded and yielded in order by shard ID then chunk ID.

        Args:
            data: Not used
            **kwargs: Additional keyword arguments (not used)

        Yields:
            dict[str, Any]: Dictionary containing activations, tokens, and info

        Raises:
            ValueError: If hook points have different numbers of chunks
            AssertionError: If loaded data doesn't match expected format
        """
        # Get sorted chunks for each hook point
        hook_chunks = {hook: self._get_sorted_chunks(hook) for hook in self.cache_dirs.keys()}

        # Verify all hook points have same number of chunks
        chunk_counts = {hook: len(chunks) for hook, chunks in hook_chunks.items()}
        if len(set(chunk_counts.values())) != 1:
            raise ValueError(
                f"Hook points have different numbers of chunks: {chunk_counts}. "
                "All hook points must have the same number of chunks."
            )

        num_chunks = len(hook_chunks[list(self.cache_dirs.keys())[0]]) if hook_chunks else 0

        # Group data by chunk_idx to maintain the same output structure
        chunk_buffer: dict[int, dict[str, Any]] = {}
        stream = self._process_chunks(hook_chunks, num_chunks)

        for single_hook_data in stream:
            chunk_idx = single_hook_data["chunk_idx"]
            hook_point = single_hook_data["hook_point"]

            # Initialize chunk buffer entry if not exists
            if chunk_idx not in chunk_buffer:
                chunk_buffer[chunk_idx] = {
                    k: v for k, v in single_hook_data.items() if k not in ["hook_point", "chunk_idx", "activation"]
                }

            # Add activation for this hook point
            chunk_buffer[chunk_idx][hook_point] = single_hook_data["activation"]

            # Verify tokens consistency across hook points for the same chunk
            if not torch.allclose(single_hook_data["tokens"], chunk_buffer[chunk_idx]["tokens"]):
                raise AssertionError(f"Loading cached activation error: tokens mismatch for chunk {chunk_idx}")

            # Check if we have all hook points for this chunk
            if len(chunk_buffer[chunk_idx]) - 2 == len(self.cache_dirs):  # -2 for tokens and meta
                # Yield the complete chunk data
                activations: dict[str, Any] = move_dict_of_tensor_to_device(
                    chunk_buffer[chunk_idx],
                    device=self.device,
                )
                if self.dtype is not None:
                    for k, v in activations.items():
                        if k in self.cache_dirs.keys():
                            activations[k] = v.to(self.dtype)

                # Flatten tokens if needed (maintain existing logic)
                while activations["tokens"].ndim >= 3:

                    def flatten(x: torch.Tensor | list[list[Any]]) -> torch.Tensor | list[Any]:
                        if isinstance(x, torch.Tensor):
                            return x.flatten(start_dim=0, end_dim=1)
                        else:
                            return [a for b in x for a in b]

                    activations = {k: flatten(v) for k, v in activations.items()}

                yield activations
                # Remove from buffer to free memory
                del chunk_buffer[chunk_idx]


class CachedActivationDataset(Dataset):
    """Wrap the data loading process with torch dataset and loader for multiprocessing.

    This dataset uses (chunk_idx, hook_point) as the index to enable parallel loading
    of individual hook points instead of loading all hook points sequentially.
    """

    def __init__(
        self,
        activation_loader: CachedActivationLoader,
        hook_chunks: dict[str, list[ChunkInfo]],
        num_chunks: int,
    ):
        self.activation_loader = activation_loader
        self.hook_chunks = hook_chunks
        self.num_chunks = num_chunks
        self.hook_points = list(hook_chunks.keys())

        # Create index mapping: (chunk_idx, hook_point) -> dataset_idx
        self.index_mapping: list[Tuple[int, str]] = []
        for chunk_idx in range(num_chunks):
            for hook_point in self.hook_points:
                self.index_mapping.append((chunk_idx, hook_point))

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, dataset_idx):
        chunk_idx, hook_point = self.index_mapping[dataset_idx]
        return self.activation_loader.load_single_hook_chunk(
            chunk_idx,
            hook_point,
            self.hook_chunks,
        )
