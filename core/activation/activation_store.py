from typing import Callable, Dict, Generator, Iterable
import torch
import torch.distributed as dist
import torch.utils.data

from transformer_lens import HookedTransformer

from datasets import load_dataset, load_from_disk

from core.config import ActivationStoreConfig
from core.activation.activation_source import ActivationSource, TokenActivationSource
from core.activation.token_source import TokenSource

class ActivationStore:
    def __init__(self, 
        act_source: ActivationSource,
        d_model: int,
        n_tokens_in_buffer=500000,
        device="cuda",
        use_ddp=False
    ):
        self.act_source = act_source
        self.d_model = d_model
        self.buffer_size = n_tokens_in_buffer
        self.device = device
        self.use_ddp = use_ddp
        self._store: Dict[str, torch.Tensor] = {}
        
    def initialize(self):
        self.refill()
        self.shuffle()

    def shuffle(self):
        for k in self._store:
            self._store[k] = self._store[k][torch.randperm(len(self._store[k]))]

    def refill(self):
        while self.__len__() < self.buffer_size:
            new_act = self.act_source.next()
            if new_act is None:
                break
            for k, v in new_act.items():
                v = v.to(self.device)
                if k not in self._store:
                    self._store[k] = v
                else:
                    self._store[k] = torch.cat([self._store[k], v], dim=0)
            # Check if all activations have the same size
            assert len(set(v.size(0) for v in self._store.values())) == 1

    def __len__(self):
        if len(self._store) == 0:
            return 0
        return next(iter(self._store.values())).size(0)

    def next(self, batch_size) -> Dict[str, torch.Tensor] | None:
        # Check if the activation store needs to be refilled.
        need_refill = torch.tensor([self.__len__() < self.buffer_size // 2], device=self.device, dtype=torch.int)
        if self.use_ddp: # When using DDP, we do refills in a synchronized manner to save time
            dist.all_reduce(need_refill, op=dist.ReduceOp.MAX)
        if need_refill.item() > 0:
            self.refill()
            self.shuffle()
        if self.use_ddp: # Wait for all processes to refill the store
            dist.barrier()

        ret = {k: v[:batch_size] for k, v in self._store.items()}
        for k in self._store:
            self._store[k] = self._store[k][batch_size:]
        return ret if len(ret) > 0 else None
        
    def next_tokens(self, batch_size: int) -> torch.Tensor | None:
        if isinstance(self.act_source, TokenActivationSource):
            return self.act_source.next_tokens(batch_size)
        raise NotImplementedError("This method is only implemented for TokenActivationSource")
    
    @staticmethod
    def from_config(model: HookedTransformer, cfg: ActivationStoreConfig):
        if cfg.use_cached_activations:
            # TODO: Implement this
            raise NotImplementedError
        else:
            token_source = TokenSource.from_config(model=model, cfg=cfg)
            
            return ActivationStore(
                act_source=TokenActivationSource(
                    token_source=token_source,
                    model=model,
                    token_batch_size=cfg.store_batch_size,
                    act_name=cfg.hook_point,
                    d_model=cfg.d_model,
                    seq_len=cfg.context_size,
                    device=cfg.device,
                    dtype=cfg.dtype,
                ),
                d_model=cfg.d_model,
                n_tokens_in_buffer=cfg.n_tokens_in_buffer,
                device=cfg.device,
                use_ddp=cfg.use_ddp,
            )