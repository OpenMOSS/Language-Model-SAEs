import json
import re
from abc import abstractmethod
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import torch
from safetensors.torch import load_file
from tqdm import tqdm

from lm_saes.activation.processors.core import BaseActivationProcessor


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


class BaseCachedActivationLoader(BaseActivationProcessor[None, Iterable[dict[str, Any]]]):
    """Base class for cached activation loaders.

    Args:
        cache_dir: Root directory containing cached activations
        hook_points: List of hook point names to load
        device: Device to load tensors to
    """

    def __init__(self, cache_dir: str | Path, hook_points: list[str], device: str = "cpu"):
        self.cache_dir = Path(cache_dir)
        self.hook_points = hook_points
        self.device = device

    def _load_chunk_for_hooks(self, chunk_idx: int, hook_chunks: dict[str, list[ChunkInfo]]) -> dict[str, Any]:
        """Load chunk data for all hook points at given index.

        Args:
            chunk_idx: Index of the chunk to load
            hook_chunks: Dictionary mapping hook points to their chunk info lists

        Returns:
            dict[str, Any]: Combined chunk data for all hooks
        """
        chunk_data = {}

        for hook in self.hook_points:
            chunk = hook_chunks[hook][chunk_idx]
            data: dict[str, Any] = self._load_chunk(chunk.path)

            # Validate data format
            assert isinstance(data, dict), f"Loading cached activation {chunk.path} error: returned {type(data)}"
            assert "activation" in data, f"Loading cached activation {chunk.path} error: missing 'activation' field"
            assert "tokens" in data, f"Loading cached activation {chunk.path} error: missing 'tokens' field"

            chunk_data[hook] = data["activation"]

            # Store tokens and info from first hook point only
            if hook == self.hook_points[0]:
                chunk_data["tokens"] = data["tokens"]
                if "meta" in data:
                    chunk_data["meta"] = data["meta"]
            else:
                assert torch.allclose(
                    data["tokens"], chunk_data["tokens"]
                ), f"Loading cached activation {chunk.path} error: tokens mismatch"
                if "meta" in data:
                    assert (
                        data["meta"] == chunk_data["meta"]
                    ), f"Loading cached activation {chunk.path} error: info mismatch"

        return chunk_data

    def _get_sorted_chunks(self, hook_point: str) -> list[ChunkInfo]:
        """Get sorted list of chunk files for a hook point.

        Args:
            hook_point: Name of the hook point

        Returns:
            List of ChunkInfo objects sorted by shard ID then chunk ID

        Raises:
            FileNotFoundError: If hook point directory doesn't exist
        """
        hook_dir = self.cache_dir / hook_point
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

    @abstractmethod
    def _process_chunks(self, hook_chunks: dict[str, list[ChunkInfo]], total_chunks: int) -> Iterator[dict[str, Any]]:
        """Process chunks and yield samples.

        Args:
            hook_chunks: Dictionary mapping hook points to their chunk info lists
            total_chunks: Total number of chunks to process

        Yields:
            dict[str, Any]: Sample data containing activations, tokens, and meta
        """
        pass

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
        hook_chunks = {hook: self._get_sorted_chunks(hook) for hook in self.hook_points}

        # Verify all hook points have same number of chunks
        chunk_counts = {hook: len(chunks) for hook, chunks in hook_chunks.items()}
        if len(set(chunk_counts.values())) != 1:
            raise ValueError(
                f"Hook points have different numbers of chunks: {chunk_counts}. "
                "All hook points must have the same number of chunks."
            )

        yield from self._process_chunks(hook_chunks, len(hook_chunks[self.hook_points[0]]))


class SequentialCachedActivationLoader(BaseCachedActivationLoader):
    """Sequential implementation of cached activation loader."""

    def _process_chunks(self, hook_chunks: dict[str, list[ChunkInfo]], total_chunks: int) -> Iterator[dict[str, Any]]:
        for chunk_idx in range(total_chunks):
            chunk_data = self._load_chunk_for_hooks(chunk_idx, hook_chunks)
            chunk_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in chunk_data.items()}
            yield chunk_data


class ParallelCachedActivationLoader(BaseCachedActivationLoader):
    """Parallel implementation of cached activation loader using ThreadPoolExecutor.

    Args:
        cache_dir: Root directory containing cached activations
        hook_points: List of hook point names to load
        device: Device to load tensors to
        executor: ThreadPoolExecutor for parallel loading
        max_active_chunks: Maximum number of chunks to load in parallel. Defaults to 2.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        hook_points: list[str],
        device: str,
        executor: Optional[ThreadPoolExecutor] = None,
        max_active_chunks: int = 2,
    ):
        super().__init__(cache_dir, hook_points, device)
        self.executor = executor or ThreadPoolExecutor(max_workers=max_active_chunks)
        self._owned_executor = executor is None
        self.max_active_chunks = max_active_chunks

    def _process_chunks(self, hook_chunks: dict[str, list[ChunkInfo]], total_chunks: int) -> Iterator[dict[str, Any]]:
        futures = set()
        next_chunk = 0

        pbar = tqdm(total=total_chunks, desc="Loading chunks", smoothing=0.001, miniters=1)

        while next_chunk < total_chunks or futures:
            # Submit new tasks up to max_active_chunks
            while len(futures) < self.max_active_chunks and next_chunk < total_chunks:
                future = self.executor.submit(self._load_chunk_for_hooks, next_chunk, hook_chunks)
                futures.add(future)
                pbar.set_postfix({"Active chunks": len(futures)})
                next_chunk += 1

            if futures:
                # Wait for any task to complete
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                pbar.set_postfix({"Active chunks": len(futures)})
                # Process completed chunks in order
                for future in tqdm(done, desc="Processing chunks", smoothing=0.001, leave=False):
                    chunk_data = future.result()
                    chunk_data = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in chunk_data.items()
                    }
                    yield chunk_data
                    pbar.update(1)

        pbar.close()

    def __del__(self) -> None:
        if self._owned_executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
