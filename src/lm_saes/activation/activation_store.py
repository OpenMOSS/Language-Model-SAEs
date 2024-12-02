import math
from typing import Dict

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.utils.data
from torch.distributed.device_mesh import init_device_mesh
from transformer_lens import HookedTransformer

from ..config import ActivationStoreConfig
from .activation_source import (
    ActivationSource,
    CachedActivationSource,
    TokenActivationSource,
)


class ActivationStore:
    def __init__(self, act_source: ActivationSource, cfg: ActivationStoreConfig):
        self.act_source = act_source
        self.shuffle_activations = cfg.shuffle_activations
        self.buffer_size = cfg.n_tokens_in_buffer
        self.device = cfg.device
        self.ddp_size = cfg.ddp_size
        self.tp_size = cfg.tp_size
        self._store: Dict[str, torch.Tensor] = {}
        self._all_gather_buffer: Dict[str, torch.Tensor] = {}
        self.device_mesh = init_device_mesh("cuda", (self.ddp_size, self.tp_size), mesh_dim_names=("ddp", "tp"))

    def initialize(self):
        self.refill()
        if self.shuffle_activations:
            self.shuffle()

    def shuffle(self):
        if len(self._store) == 0:
            return
        perm = torch.randperm(len(self._store[next(iter(self._store))]))
        for k in self._store:
            self._store[k] = self._store[k][perm]

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
        need_refill = torch.tensor(
            [self.__len__() < self.buffer_size // 2],
            device=self.device,
            dtype=torch.int,
        )
        if dist.is_initialized():  # When using DDP, we do refills in a synchronized manner to save time
            dist.all_reduce(need_refill, op=dist.ReduceOp.MAX)
        if need_refill.item() > 0:
            self.refill()
            if self.shuffle_activations:
                self.shuffle()
        if dist.is_initialized():  # Wait for all processes to refill the store
            dist.barrier()
        if self.tp_size > 1:
            for k, v in self._store.items():
                if k not in self._all_gather_buffer:
                    self._all_gather_buffer[k] = torch.empty(size=(0,), dtype=v.dtype, device=self.device)

                gather_len = math.ceil((batch_size - self._all_gather_buffer[k].size(0)) / self.tp_size)

                assert gather_len <= v.size(0), "Not enough activations in the store"
                gather_tensor = funcol.all_gather_tensor(v[:gather_len], gather_dim=0, group=self.device_mesh["tp"])
                self._store[k] = v[gather_len:]

                self._all_gather_buffer[k] = torch.cat([self._all_gather_buffer[k], gather_tensor], dim=0)

            ret = {k: self._all_gather_buffer[k][:batch_size] for k in self._store}
            for k in self._store:
                self._all_gather_buffer[k] = self._all_gather_buffer[k][batch_size:]
            return ret if len(ret) > 0 else None

        else:
            ret = {k: v[:batch_size] for k, v in self._store.items()}
            for k in self._store:
                self._store[k] = self._store[k][batch_size:]
            return ret if len(ret) > 0 else None

    def next_tokens(self, batch_size: int) -> torch.Tensor | None:
        if self.tp_size > 1:
            # TODO: only get next token from the root process
            next_tokens = self.act_source.next_tokens(batch_size)
            # funcol.broadcast does not work and we dont know why
            dist.broadcast(next_tokens, src=0)
            return next_tokens
        else:
            return self.act_source.next_tokens(batch_size)

    @staticmethod
    def from_config(model: HookedTransformer, cfg: ActivationStoreConfig):
        act_source: ActivationSource
        if cfg.use_cached_activations:
            act_source = CachedActivationSource(cfg=cfg)
        else:
            act_source = TokenActivationSource(
                model=model,
                cfg=cfg,
            )
        return ActivationStore(
            act_source=act_source,
            cfg=cfg,
        )
