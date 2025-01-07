import itertools
import json
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from safetensors.torch import save_file
from torch.distributed.device_mesh import DeviceMesh
from tqdm import tqdm

from lm_saes.config import ActivationWriterConfig


class ActivationWriter:
    """Writes activations to disk in a format compatible with CachedActivationLoader.

    Args:
        cfg: Configuration for writing activations
        executor: Optional ThreadPoolExecutor for parallel writing. If None, a new executor
                 will be created with max_workers=2
        max_active_chunks: Maximum number of chunks to write in parallel. Defaults to 2
    """

    def __init__(
        self,
        cfg: ActivationWriterConfig,
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        self.cache_dir = Path(cfg.cache_dir)
        self.cfg = cfg
        if cfg.num_workers is None:
            self.executor = None
        else:
            self.executor = executor or ThreadPoolExecutor(max_workers=cfg.num_workers)
        self._owned_executor = cfg.num_workers is not None and executor is None

        # Create directories for each hook point
        for hook_point in self.cfg.hook_points:
            hook_dir = self.cache_dir / hook_point
            hook_dir.mkdir(parents=True, exist_ok=True)

    def _write_chunk(
        self,
        hook_point: str,
        chunk_data: dict[str, Any],
        chunk_name: str,
        meta: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Write a single chunk to disk.

        Args:
            hook_point: Name of the hook point
            chunk_data: Dictionary containing activation and token data
            chunk_name: Name for the chunk file
            meta: Optional metadata to save
        """
        chunk_path = self.cache_dir / hook_point / f"{chunk_name}.{self.cfg.format}"

        if self.cfg.format == "pt":
            torch.save(chunk_data | ({"meta": meta} if meta is not None else {}), chunk_path)
        elif self.cfg.format == "safetensors":
            save_file(chunk_data, chunk_path)
            if meta is not None:
                meta_path = chunk_path.with_suffix(".meta.json")
                with open(meta_path, "w") as f:
                    json.dump(meta, f)
        else:
            raise ValueError(f"Invalid format: {self.cfg.format}")

    def process(
        self,
        data: Iterable[dict[str, Any]],
        *,
        device_mesh: Optional[DeviceMesh] = None,
        start_shard: int = 0,
    ) -> None:
        """Write activation data to disk in chunks.

        Processes a stream of activation dictionaries, accumulating samples until reaching
        the configured chunk size, then writes each chunk to disk. Files are organized by
        hook point with names following the pattern 'chunk-{N}.pt'.

        Args:
            data: Stream of activation dictionaries containing:
                - Activations for each hook point
                - Original tokens
                - Meta information
            device_mesh: The device mesh to use for distributed writing. If None, will write to disk on the current rank.
            start_shard: The shard to start writing from.
        """
        total = (
            self.cfg.total_generating_tokens // device_mesh.get_group("data").size()
            if device_mesh is not None and self.cfg.total_generating_tokens is not None
            else self.cfg.total_generating_tokens
        )
        pbar = tqdm(desc="Writing activations to disk", total=total)
        n_tokens_written = 0

        futures = set() if self.cfg.num_workers is not None else None

        for chunk_id, chunk in enumerate(itertools.batched(data, self.cfg.n_samples_per_chunk)):
            assert all(
                [hook_point in d for d in chunk for hook_point in self.cfg.hook_points]
            ), f"All samples must have all hook points: {self.cfg.hook_points}"

            tokens = torch.stack([d["tokens"] for d in chunk])
            meta = [d["meta"] for d in chunk] if "meta" in chunk[0] else None

            chunk_name = (
                f"chunk-{chunk_id:08d}"
                if device_mesh is None
                else f"shard-{device_mesh.get_group('data').rank() + start_shard}-chunk-{chunk_id:08d}"
            )

            # Submit writing tasks for each hook point
            for hook_point in self.cfg.hook_points:
                chunk_data = {"activation": torch.stack([d[hook_point] for d in chunk]), "tokens": tokens}
                if futures is None:
                    self._write_chunk(hook_point, chunk_data, chunk_name, meta)
                else:
                    assert self.executor is not None, "Executor is not initialized"
                    future = self.executor.submit(
                        self._write_chunk,
                        hook_point,
                        chunk_data,
                        chunk_name,
                        meta,
                    )
                    futures.add(future)

            if futures is not None:
                assert self.cfg.num_workers is not None, "num_workers must be set to use parallel writing"
                # Wait for some futures to complete if we have too many pending
                while len(futures) >= self.cfg.num_workers * 2:
                    done, futures = wait(futures, return_when="FIRST_COMPLETED")
                    for future in done:
                        future.result()  # Raise any exceptions that occurred

            n_tokens_written += tokens.shape[0] * tokens.shape[1]
            pbar.update(tokens.shape[0] * tokens.shape[1])

            if total is not None and n_tokens_written >= total:
                break

        if futures is not None:
            # Wait for remaining futures to complete
            for future in as_completed(futures):
                future.result()

        pbar.close()

    def __del__(self) -> None:
        """Cleanup the executor if we own it."""
        if self._owned_executor and self.executor is not None:
            self.executor.shutdown(wait=True)
