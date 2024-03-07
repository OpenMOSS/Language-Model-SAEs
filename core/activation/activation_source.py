from abc import ABC
from typing import Iterator, Dict

import torch
from torch.utils.data import DataLoader

from transformer_lens import HookedTransformer

from einops import rearrange, repeat

from core.activation.token_source import TokenSource

class ActivationSource(ABC):
    def next(self) -> Dict[str, torch.Tensor] | None:
        """
        Get the next batch of activations.

        Returns:
            A dictionary where the keys are the names of the activations and the values are tensors of shape (batch_size, d_model). If there are no more activations, return None.
        """
        raise NotImplementedError    
class TokenActivationSource(ActivationSource):
    """
    An activation source that generates activations from a token source.
    """
    def __init__(self, token_source: TokenSource, model: HookedTransformer, token_batch_size: int, act_name: str, d_model: int, seq_len: int, device: str, dtype: torch.dtype):
        self.token_source = token_source
        self.token_batch_size = token_batch_size
        self.model = model
        self.act_name = act_name
        self.device = device
        self.dtype = dtype
    
    def next(self) -> Dict[str, torch.Tensor] | None:
        tokens = self.token_source.next(self.token_batch_size)

        if tokens is None:
            return None
        
        _, cache = self.model.run_with_cache(tokens, names_filter=[self.act_name])

        batch_size = tokens.size(0)
        seq_len = tokens.size(1)

        ret = {
            "activation": rearrange(cache[self.act_name].to(dtype=self.dtype, device=self.device), "b l d -> (b l) d"),
            "position": repeat(torch.arange(seq_len, device=self.device, dtype=torch.long), 'l -> (b l)', b=batch_size),
            "context": repeat(tokens.to(dtype=torch.long), 'b l -> (b repeat) l', repeat=seq_len),
        }

        return ret
    
    def next_tokens(self, batch_size: int) -> torch.Tensor | None:
        return self.token_source.next(batch_size)