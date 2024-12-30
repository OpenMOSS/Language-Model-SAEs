import itertools
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors.torch import save_file
from tqdm import tqdm

from lm_saes.config import ActivationWriterConfig


class ActivationWriter:
    """Writes activations to disk in a format compatible with CachedActivationLoader.

    This processor takes activation data and saves it to disk in chunks, organizing files
    by hook point. The saved format matches what CachedActivationLoader expects.

    Args:
        cfg: Configuration for writing activations
    """

    def __init__(self, cfg: ActivationWriterConfig):
        self.cache_dir = Path(cfg.cache_dir)
        self.cfg = cfg

        # Create directories for each hook point
        for hook_point in self.cfg.hook_points:
            hook_dir = self.cache_dir / hook_point
            hook_dir.mkdir(parents=True, exist_ok=True)

    def process(self, data: Iterable[dict[str, Any]], **kwargs) -> None:
        """Write activation data to disk in chunks.

        Processes a stream of activation dictionaries, accumulating samples until reaching
        the configured chunk size, then writes each chunk to disk. Files are organized by
        hook point with names following the pattern 'chunk-{N}.pt'.

        Args:
            data: Stream of activation dictionaries containing:
                - Activations for each hook point
                - Original tokens
                - Meta information
            **kwargs: Additional keyword arguments (unused)
        """
        # Validate all inputs have required fields
        for d in data:
            assert all(
                k in d for k in ["tokens", "meta"] + self.cfg.hook_points
            ), f"Missing required fields in input. Found keys: {list(d.keys())}"

        pbar = tqdm(
            desc="Writing activations to disk",
            total=self.cfg.total_generating_tokens,
        )
        n_tokens_written = 0

        # Use itertools to batch the data
        for chunk_id, chunk in enumerate(itertools.batched(data, self.cfg.n_samples_per_chunk)):
            tokens = torch.stack([d["tokens"] for d in chunk])
            meta = [d["meta"] for d in chunk] if "meta" in chunk[0] else None

            # Write chunk for each hook point
            for hook_point in self.cfg.hook_points:
                chunk_data = {"activation": torch.stack([d[hook_point] for d in chunk]), "tokens": tokens}
                chunk_path = self.cache_dir / hook_point / f"chunk-{chunk_id}.{self.cfg.format}"
                if self.cfg.format == "pt":
                    torch.save(chunk_data | ({"meta": meta} if meta is not None else {}), chunk_path)
                elif self.cfg.format == "safetensors":
                    save_file(chunk_data, chunk_path)
                    if meta is not None:
                        # Save meta as a separate file
                        meta_path = chunk_path.with_suffix(".meta")
                        torch.save(meta, meta_path)
                else:
                    raise ValueError(f"Invalid format: {self.cfg.format}")

            n_tokens_written += self.cfg.n_samples_per_chunk
            pbar.update(self.cfg.n_samples_per_chunk)

            if self.cfg.total_generating_tokens is not None and n_tokens_written >= self.cfg.total_generating_tokens:
                break

        pbar.close()
