import os

import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from ..config import (
    ActivationGenerationConfig,
)
from ..utils.misc import is_master, print_once
from .activation_store import ActivationStore
from .token_source import TokenSource


class SingletonActStore:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SingletonActStore, cls).__new__(cls)
        return cls._instance

    def __init__(self, model: HookedTransformer, cfg: ActivationGenerationConfig):
        if not hasattr(self, "_initialized"):
            self.act_store = ActivationStore.from_config(model=model, cfg=cfg.act_store)
            self.act_store.initialize()
            self._initialized = True

    def next(self, *args, **kwargs):
        return self.act_store.next(*args, **kwargs)


class SingletonTokenSource:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SingletonTokenSource, cls).__new__(cls)
        return cls._instance

    def __init__(self, model: HookedTransformer, cfg: ActivationGenerationConfig):
        if not hasattr(self, "_initialized"):
            self.token_source = TokenSource.from_config(model=model, cfg=cfg.dataset)
            self._initialized = True

    def next(self, *args, **kwargs):
        return self.token_source.next(*args, **kwargs)


def generate_unshuffled_activation(model: HookedTransformer, cfg: ActivationGenerationConfig):
    token_source = SingletonTokenSource(model, cfg)
    tokens = token_source.next(cfg.dataset.store_batch_size)
    assert tokens is not None, "Out of tokens"
    num_generated_tokens = tokens.size(0) * tokens.size(1)

    _, cache = model.run_with_cache_until(tokens, names_filter=cfg.hook_points, until=cfg.hook_points[-1])

    return cache, tokens, num_generated_tokens


def generate_shuffled_activation(model: HookedTransformer, cfg: ActivationGenerationConfig):
    act_store = SingletonActStore(model, cfg)
    activations = act_store.next(batch_size=cfg.generate_batch_size)
    assert activations is not None, "Out of activations"

    return activations, cfg.generate_batch_size


@torch.no_grad()
def make_activation_dataset(model: HookedTransformer, cfg: ActivationGenerationConfig):
    element_size = torch.finfo(cfg.lm.dtype).bits / 8
    token_act_size = element_size * cfg.lm.d_model
    max_tokens_per_chunk = cfg.chunk_size // token_act_size
    print(f"Each token takes {token_act_size} bytes.")
    print_once(f"Making activation dataset with approximately {max_tokens_per_chunk} tokens per chunk")

    if is_master():
        for hook_point in cfg.hook_points:
            os.makedirs(os.path.join(cfg.activation_save_path, hook_point), exist_ok=False)

    if cfg.ddp_size > 1:
        dist.barrier()
        total_generating_tokens = cfg.total_generating_tokens // dist.get_world_size()
    else:
        total_generating_tokens = cfg.total_generating_tokens

    n_tokens = 0
    chunk_idx = 0
    pbar = tqdm(
        total=total_generating_tokens,
        desc=f"Activation dataset Rank {dist.get_rank()}" if dist.is_initialized() else "Activation dataset",
    )

    while n_tokens < total_generating_tokens:
        act_shape = (cfg.dataset.context_size, cfg.lm.d_model) if cfg.generate_with_context else (cfg.lm.d_model,)

        act_dict = {
            hook_point: torch.empty((0,) + act_shape, dtype=cfg.lm.dtype, device=cfg.lm.device)
            for hook_point in cfg.hook_points
        }

        if cfg.generate_with_context:
            context = torch.empty((0, cfg.dataset.context_size), dtype=torch.long, device=cfg.lm.device)
        else:
            context = None

        n_tokens_in_chunk = 0

        while n_tokens_in_chunk < max_tokens_per_chunk:
            if cfg.generate_with_context:
                assert context is not None, "Context is not initialized"
                activations, tokens, num_generated_tokens = generate_unshuffled_activation(model, cfg)
                context = torch.cat([context, tokens], dim=0)
            else:
                activations, num_generated_tokens = generate_shuffled_activation(model, cfg)

            for hook_point in cfg.hook_points:
                act_dict[hook_point] = torch.cat([act_dict[hook_point], activations[hook_point]], dim=0)

            n_tokens += num_generated_tokens
            n_tokens_in_chunk += num_generated_tokens

            pbar.update(num_generated_tokens)

        if cfg.generate_with_context:
            assert context is not None, "Context is not initialized"
            position = (
                torch.arange(cfg.dataset.context_size, device=cfg.lm.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(context.size(0), -1)
            )
        else:
            position = None

        if cfg.zero_center_activations:
            non_activation_dims = [0, 1] if cfg.generate_with_context else 0
            for hook_point in cfg.hook_points:
                act_dict[hook_point] -= act_dict[hook_point].mean(dim=non_activation_dims)

        for hook_point in cfg.hook_points:
            result = {"activation": act_dict[hook_point]}
            if cfg.generate_with_context:
                assert context is not None, "Context is not initialized"
                assert position is not None, "Position is not initialized"
                result["context"] = context
                result["position"] = position
            torch.save(
                result,
                os.path.join(
                    cfg.activation_save_path,
                    hook_point,
                    f"chunk-{str(chunk_idx).zfill(5)}.pt"
                    if not dist.is_initialized()
                    else f"shard-{dist.get_rank()}-chunk-{str(chunk_idx).zfill(5)}.pt",
                ),
            )
        chunk_idx += 1
        torch.cuda.empty_cache()

    pbar.close()
