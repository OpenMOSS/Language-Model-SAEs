import random
from abc import ABC
from typing import Dict

import torch
import torch.distributed as dist
from einops import rearrange, repeat
from transformer_lens import HookedTransformer

from lm_saes.activation.activation_dataset import (
    list_activation_chunks,
    load_activation_chunk,
)
from lm_saes.activation.token_source import TokenSource
from lm_saes.config import ActivationStoreConfig


class ActivationSource(ABC):
    def next(self) -> Dict[str, torch.Tensor] | None:
        """
        Get the next batch of activations.

        Returns:
            A dictionary where the keys are the names of the activations and the values are tensors of shape (batch_size, d_model). If there are no more activations, return None.
        """
        raise NotImplementedError

    def next_tokens(self, batch_size: int) -> torch.Tensor | None:
        """
        Get the next batch of tokens.

        Returns:
            A tensor of shape (batch_size, seq_len) where seq_len is the length of the sequence.
        """
        raise NotImplementedError


class TokenActivationSource(ActivationSource):
    """
    An activation source that generates activations from a token source.
    """

    def __init__(self, model: HookedTransformer, cfg: ActivationStoreConfig):
        self.token_source = TokenSource.from_config(model=model, cfg=cfg.dataset)
        self.model = model
        assert model.tokenizer is not None, "Tokenizer is not set"
        self.tokenizer = model.tokenizer
        self.cfg = cfg

    def next(self) -> Dict[str, torch.Tensor] | None:
        tokens = self.next_tokens(self.cfg.dataset.store_batch_size)

        if tokens is None:
            return None
        with torch.no_grad():
            _, cache = self.model.run_with_cache_until(
                tokens, names_filter=self.cfg.hook_points, until=self.cfg.hook_points[-1]
            )

            filter_mask = torch.logical_and(
                tokens != self.tokenizer.eos_token_id, tokens != self.tokenizer.pad_token_id
            )
            filter_mask = torch.logical_and(filter_mask, tokens != self.tokenizer.bos_token_id)

            filter_mask = rearrange(filter_mask, "b l -> (b l)")

            ret = {
                k: rearrange(cache[k].to(dtype=self.cfg.dtype, device=self.cfg.device), "b l d -> (b l) d")[filter_mask]
                for k in self.cfg.hook_points
            }

            return ret

    def next_tokens(self, batch_size: int) -> torch.Tensor | None:
        return self.token_source.next(batch_size)


class CachedActivationSource(ActivationSource):
    def __init__(self, cfg: ActivationStoreConfig):
        self.cfg = cfg
        assert cfg.use_cached_activations and len(cfg.cached_activations_path) == 1
        assert len(cfg.hook_points) == 1, "CachedActivationSource only supports one hook point"
        self.hook_point = cfg.hook_points[0]
        self.chunk_paths = list_activation_chunks(cfg.cached_activations_path[0], self.hook_point)
        if cfg.ddp_size > 1:
            self.chunk_paths = [
                p for i, p in enumerate(self.chunk_paths) if i % dist.get_world_size() == dist.get_rank()
            ]
        random.shuffle(self.chunk_paths)

        self.token_buffer = torch.empty((0, cfg.dataset.context_size), dtype=torch.long, device=cfg.device)

    def _load_next_chunk(self):
        if len(self.chunk_paths) == 0:
            return None
        chunk_path = self.chunk_paths.pop()
        chunk = load_activation_chunk(chunk_path)
        return chunk

    def next(self) -> Dict[str, torch.Tensor] | None:
        chunk = self._load_next_chunk()
        if chunk is None:
            return None
        ret = {
            self.hook_point: rearrange(
                chunk["activation"].to(dtype=self.cfg.dtype, device=self.cfg.device), "b l d -> (b l) d"
            ),
            "position": rearrange(chunk["position"].to(dtype=torch.long, device=self.cfg.device), "b l -> (b l)"),
            "context": repeat(
                chunk["context"].to(dtype=torch.long, device=self.cfg.device),
                "b l -> (b repeat) l",
                repeat=chunk["activation"].size(1),
            ),
        }
        return ret

    def next_tokens(self, batch_size: int) -> torch.Tensor | None:
        if self.token_buffer.size(0) < batch_size:
            chunk = self._load_next_chunk()
            if chunk is None:
                return None
            self.token_buffer = torch.cat([self.token_buffer, chunk["context"]], dim=0)
        ret = self.token_buffer[:batch_size]
        self.token_buffer = self.token_buffer[batch_size:]
        return ret
