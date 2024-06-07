from abc import ABC
from typing import Dict
import random

import torch

from transformer_lens import HookedTransformer

from einops import rearrange, repeat

from lm_saes.activation.token_source import TokenSource
from lm_saes.activation.activation_dataset import load_activation_chunk, list_activation_chunks
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
        self.token_source = TokenSource.from_config(model=model, cfg=cfg)
        self.model = model
        self.cfg = cfg
    
    def next(self) -> Dict[str, torch.Tensor] | None:
        tokens = self.token_source.next(self.cfg.store_batch_size)

        if tokens is None:
            return None
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens, names_filter=self.cfg.hook_points)

            filter_mask = torch.logical_and(tokens.ne(self.model.tokenizer.eos_token_id), tokens.ne(self.model.tokenizer.pad_token_id))
            filter_mask = torch.logical_and(filter_mask, tokens.ne(self.model.tokenizer.bos_token_id))

            filter_mask = rearrange(filter_mask, "b l -> (b l)")

            ret = {k: rearrange(cache[k].to(dtype=self.cfg.dtype, device=self.cfg.device), "b l d -> (b l) d")[filter_mask] for k in self.cfg.hook_points}

            return ret
    
    def next_tokens(self, batch_size: int) -> torch.Tensor | None:
        return self.token_source.next(batch_size)
    
class CachedActivationSource(ActivationSource):
    def __init__(self, cfg: ActivationStoreConfig):
        self.cfg = cfg
        assert cfg.use_cached_activations and cfg.cached_activations_path is not None
        assert len(cfg.hook_points) == 1, "CachedActivationSource only supports one hook point"
        self.hook_point = cfg.hook_points[0]
        self.chunk_paths = list_activation_chunks(cfg.cached_activations_path, self.hook_point)
        if cfg.use_ddp:
            self.chunk_paths = [p for i, p in enumerate(self.chunk_paths) if i % cfg.world_size == cfg.rank]
        random.shuffle(self.chunk_paths)

        self.token_buffer = torch.empty((0, cfg.context_size), dtype=torch.long, device=cfg.device)
    
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
            self.hook_point: rearrange(chunk["activation"].to(dtype=self.cfg.dtype, device=self.cfg.device), "b l d -> (b l) d"),
            "position": rearrange(chunk["position"].to(dtype=torch.long, device=self.cfg.device), "b l -> (b l)"),
            "context": repeat(chunk["context"].to(dtype=torch.long, device=self.cfg.device), 'b l -> (b repeat) l', repeat=chunk["activation"].size(1)),
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