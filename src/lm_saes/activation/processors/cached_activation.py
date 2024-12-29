import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

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

        Supports two filename formats:
        - shard-{shard_id}-chunk-{chunk_id}.pt
        - chunk-{chunk_id}.pt

        Args:
            path: Path to chunk file

        Returns:
            ChunkInfo object with parsed shard and chunk IDs

        Raises:
            ValueError: If filename doesn't match either expected pattern
        """
        # Try shard-chunk format first
        match = re.match(r"shard-(\d+)-chunk-(\d+)\.pt", path.name)
        if match:
            return cls(path=path, shard_id=int(match.group(1)), chunk_id=int(match.group(2)))

        # Try chunk-only format
        match = re.match(r"chunk-(\d+)\.pt", path.name)
        if match:
            return cls(
                path=path,
                shard_id=0,  # Default shard ID for non-sharded files
                chunk_id=int(match.group(1)),
            )

        raise ValueError(
            f"Invalid chunk filename format: {path.name}. " "Expected 'shard-{N}-chunk-{N}.pt' or 'chunk-{N}.pt'"
        )


class CachedActivationLoader(BaseActivationProcessor[None, Iterable[dict[str, Any]]]):
    """Loads cached model activations from disk in a streaming fashion.

    This processor loads pre-computed activations that were cached to disk, maintaining
    the same data format as ActivationGenerator output. Files are loaded in order by
    shard ID then chunk ID to preserve data ordering.

    Args:
        cache_dir: Root directory containing cached activations
        hook_points: List of hook point names to load
    """

    def __init__(self, cache_dir: str | Path, hook_points: list[str]):
        self.cache_dir = Path(cache_dir)
        self.hook_points = hook_points

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

        # Get both shard-chunk and chunk-only files
        chunks = [
            ChunkInfo.from_path(p) for pattern in ["shard-*-chunk-*.pt", "chunk-*.pt"] for p in hook_dir.glob(pattern)
        ]
        return sorted(chunks, key=lambda x: (x.shard_id, x.chunk_id))

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

        # Load chunks in order
        for chunk_idx in range(len(hook_chunks[self.hook_points[0]])):
            chunk_data = {}

            # Load data from each hook point
            for hook in self.hook_points:
                chunk = hook_chunks[hook][chunk_idx]
                data: dict[str, Any] = torch.load(chunk.path, map_location="cpu", weights_only=True)

                # Validate data format
                assert isinstance(data, dict), f"Loading cached activation {chunk.path} error: returned {type(data)}"
                assert "activation" in data, f"Loading cached activation {chunk.path} error: missing 'activation' field"
                assert "tokens" in data, f"Loading cached activation {chunk.path} error: missing 'tokens' field"
                assert "info" in data, f"Loading cached activation {chunk.path} error: missing 'info' field"

                chunk_data[hook] = data["activation"]

                # Store tokens and info from first hook point only
                if hook == self.hook_points[0]:
                    chunk_data["tokens"] = data["tokens"]
                    chunk_data["info"] = data["info"]
                else:
                    assert torch.allclose(
                        data["tokens"], chunk_data["tokens"]
                    ), f"Loading cached activation {chunk.path} error: tokens mismatch"
                    assert (
                        data["info"] == chunk_data["info"]
                    ), f"Loading cached activation {chunk.path} error: info mismatch"

            # Yield chunk data in sample-wise format
            for i in range(chunk_data[self.hook_points[0]].shape[0]):
                yield {k: v[i] for k, v in chunk_data.items()}
