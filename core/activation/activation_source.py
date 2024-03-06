from abc import ABC
from typing import Iterator, Dict

import torch
from torch.utils.data import DataLoader

from transformer_lens import HookedTransformer

from einops import rearrange, repeat


class ActivationSource(ABC):
    def next(self, batch_size: int) -> Dict[str, torch.Tensor] | None:
        """
        Get the next batch of activations.

        Args:
            batch_size: The size of the batch to return.

        Returns:
            A dictionary where the keys are the names of the activations and the values are tensors of shape (batch_size, d_model). If there are no more activations, return None.
        """
        raise NotImplementedError
    
class TokenSource:
    def __init__(self, dataloader: DataLoader, model: HookedTransformer, is_dataset_tokenized: bool, seq_len: int, device: str):
        self.dataloader = dataloader
        self.model = model
        self.is_dataset_tokenized = is_dataset_tokenized
        self.seq_len = seq_len
        self.device = device

        self.data_iter = iter(self.dataloader)
        self.token_buffer = torch.empty((0, seq_len), dtype=torch.int64, device=self.device)
        self.bos_token_id_tensor = torch.tensor([self.model.tokenizer.bos_token_id], dtype=torch.int64, device=self.device)
        self.resid = self.bos_token_id_tensor.clone()
    
    def next(self, batch_size: int) -> torch.Tensor | None:
        while self.token_buffer.size(0) < batch_size:
            try:
                batch = next(self.data_iter)
            except StopIteration:
                return None
            if self.is_dataset_tokenized:
                tokens: torch.Tensor = batch["tokens"].to(self.device)
            else:
                tokens = self.model.to_tokens(batch["text"]).to(self.device)
            while tokens.size(0) > 0:
                cur_tokens = tokens[0]
                cur_tokens = cur_tokens[torch.logical_and(cur_tokens != self.model.tokenizer.pad_token_id, cur_tokens != self.model.tokenizer.bos_token_id)]
                self.resid = torch.cat([self.resid, self.bos_token_id_tensor.clone(), cur_tokens], dim=0)
                while self.resid.size(0) >= self.seq_len:
                    self.token_buffer = torch.cat([self.token_buffer, self.resid[:self.seq_len].unsqueeze(0)], dim=0)
                    self.resid = self.resid[self.seq_len:]
                    self.resid = torch.cat([self.bos_token_id_tensor.clone(), self.resid], dim=0)
                tokens = tokens[1:]

        ret = self.token_buffer[:batch_size]
        self.token_buffer = self.token_buffer[batch_size:]

        return ret
    
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
        self.act_buffer = {
            "activation": torch.empty((0, d_model), dtype=dtype, device=device),
            "position": torch.empty((0,), dtype=torch.long, device=device),
            "context": torch.empty((0, seq_len), dtype=torch.long, device=device),
        }
    
    def next(self, batch_size: int) -> Dict[str, torch.Tensor] | None:
        while self.act_buffer["activation"].size(0) < batch_size:
            tokens = self.token_source.next(self.token_batch_size)

            if tokens is None:
                return None
            
            _, cache = self.model.run_with_cache(tokens, names_filter=[self.act_name])

            seq_len = tokens.size(1)

            self.act_buffer["activation"] = torch.cat([
                self.act_buffer["activation"], 
                rearrange(cache[self.act_name].to(dtype=self.dtype, device=self.device), "b l d -> (b l) d")
            ], dim=0)
            self.act_buffer["position"] = torch.cat([
                self.act_buffer["position"],
                repeat(torch.arange(seq_len, device=self.device, dtype=torch.long), 'l -> (b l)', b=batch_size)
            ], dim=0)
            self.act_buffer["context"] = torch.cat([
                self.act_buffer["context"],
                repeat(tokens.to(dtype=torch.long), 'b l -> (b repeat) l', repeat=seq_len)
            ], dim=0)

        ret = {
            "activation": self.act_buffer["activation"][:batch_size],
            "position": self.act_buffer["position"][:batch_size],
            "context": self.act_buffer["context"][:batch_size],
        }

        self.act_buffer["activation"] = self.act_buffer["activation"][batch_size:]
        self.act_buffer["position"] = self.act_buffer["position"][batch_size:]
        self.act_buffer["context"] = self.act_buffer["context"][batch_size:]

        return ret
    
    def next_tokens(self, batch_size: int) -> torch.Tensor | None:
        return self.token_source.next(batch_size)