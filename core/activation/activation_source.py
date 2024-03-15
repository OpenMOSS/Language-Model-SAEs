from abc import ABC
from typing import Dict
import random

import torch

from transformer_lens import HookedTransformer

from einops import rearrange, repeat

from core.activation.token_source import TokenSource
from core.activation.activation_dataset import load_activation_chunk, list_activation_chunks
from core.config import ActivationStoreConfig

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
        
        _, cache = self.model.run_with_cache(tokens, names_filter=[self.cfg.hook_point])

        batch_size = tokens.size(0)
        seq_len = tokens.size(1)

        has_bos = (tokens[:, 0] == self.model.tokenizer.bos_token_id).any()
        assert has_bos == (tokens[:, 0] == self.model.tokenizer.bos_token_id).all(), "All or none of the tokens in the batch should start with the BOS token."

        if not has_bos:
            ret = {
                "activation": rearrange(cache[self.cfg.hook_point].to(dtype=self.cfg.dtype, device=self.cfg.device), "b l d -> (b l) d"),
                "position": repeat(torch.arange(seq_len, device=self.cfg.device, dtype=torch.long), 'l -> (b l)', b=batch_size),
                "context": repeat(tokens.to(dtype=torch.long), 'b l -> (b repeat) l', repeat=seq_len),
            }
        else:
            ret = {
                "activation": rearrange(cache[self.cfg.hook_point][:, 1:].to(dtype=self.cfg.dtype, device=self.cfg.device), "b l d -> (b l) d"),
                "position": repeat(torch.arange(seq_len, device=self.cfg.device, dtype=torch.long)[1:], 'l -> (b l)', b=batch_size),
                "context": repeat(tokens.to(dtype=torch.long), 'b l -> (b repeat) l', repeat=seq_len - 1),
            }

        return ret
    
    def next_tokens(self, batch_size: int) -> torch.Tensor | None:
        return self.token_source.next(batch_size)
    
class CachedActivationSource(ActivationSource):
    def __init__(self, cfg: ActivationStoreConfig):
        self.cfg = cfg
        assert cfg.use_cached_activations and cfg.cached_activations_path is not None
        self.chunk_paths = list_activation_chunks(cfg.cached_activations_path, cfg.hook_point)
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
            "activation": rearrange(chunk["activation"].to(dtype=self.cfg.dtype, device=self.cfg.device), "b l d -> (b l) d"),
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