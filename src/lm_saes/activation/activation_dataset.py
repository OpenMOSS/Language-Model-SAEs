from typing import Dict
from transformer_lens import HookedTransformer
import torch
from tqdm.auto import tqdm
import os
import torch.distributed as dist

from lm_saes.utils.misc import print_once
from lm_saes.config import ActivationGenerationConfig
from lm_saes.activation.token_source import TokenSource


@torch.no_grad()
def make_activation_dataset(
    model: HookedTransformer,
    cfg: ActivationGenerationConfig
):
    element_size = torch.finfo(cfg.dtype).bits / 8
    token_act_size = element_size * cfg.d_model
    max_tokens_per_chunk = cfg.chunk_size // token_act_size
    print_once(f"Making activation dataset with approximately {max_tokens_per_chunk} tokens per chunk")

    token_source = TokenSource.from_config(model=model, cfg=cfg)

    if not cfg.use_ddp or cfg.rank == 0:
        for hook_point in cfg.hook_points:
            os.makedirs(os.path.join(cfg.activation_save_path, hook_point), exist_ok=False)

    if cfg.use_ddp:
        dist.barrier()
        total_generating_tokens = cfg.total_generating_tokens // cfg.world_size
    else:
        total_generating_tokens = cfg.total_generating_tokens

    n_tokens = 0
    chunk_idx = 0
    pbar = tqdm(total=total_generating_tokens, desc=f"Activation dataset Rank {cfg.rank}" if cfg.use_ddp else "Activation dataset")

    while n_tokens < total_generating_tokens:
        act_dict = {hook_point: torch.empty((0, cfg.context_size, cfg.d_model), dtype=cfg.dtype, device=cfg.device) for hook_point in cfg.hook_points}
        context = torch.empty((0, cfg.context_size), dtype=torch.long, device=cfg.device)

        n_tokens_in_chunk = 0

        while n_tokens_in_chunk < max_tokens_per_chunk:
            tokens = token_source.next(cfg.store_batch_size)
            _, cache = model.run_with_cache_until(tokens, names_filter=cfg.hook_points, until=cfg.hook_points[-1])
            for hook_point in cfg.hook_points:
                act = cache[hook_point]
                act_dict[hook_point] = torch.cat([act_dict[hook_point], act], dim=0)
            context = torch.cat([context, tokens], dim=0)
            n_tokens += tokens.size(0) * tokens.size(1)
            n_tokens_in_chunk += tokens.size(0) * tokens.size(1)

            pbar.update(tokens.size(0) * tokens.size(1))

        position = torch.arange(cfg.context_size, device=cfg.device, dtype=torch.long).unsqueeze(0).expand(context.size(0), -1)
        
        for hook_point in cfg.hook_points:
            torch.save(
                {
                    "activation": act_dict[hook_point],
                    "context": context,
                    "position": position,
                },
                os.path.join(cfg.activation_save_path, hook_point, f"chunk-{str(chunk_idx).zfill(5)}.pt" if not cfg.use_ddp else f"shard-{cfg.rank}-chunk-{str(chunk_idx).zfill(5)}.pt")
            )
        chunk_idx += 1

    pbar.close()

@torch.no_grad()
def list_activation_chunks(activation_path: str, hook_point: str) -> list[str]:
    return sorted([os.path.join(activation_path, hook_point, f) for f in os.listdir(os.path.join(activation_path, hook_point)) if f.endswith(".pt")])

@torch.no_grad()
def load_activation_chunk(chunk_path: str) -> Dict[str, torch.Tensor]:
    return torch.load(chunk_path)