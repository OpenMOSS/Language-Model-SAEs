from tqdm import tqdm

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from einops import repeat

from core.sae import SparseAutoEncoder
from core.config import LanguageModelSAEAnalysisConfig
from core.activation.activation_store import ActivationStore
from core.utils import print_once

def sample_feature_activations(
    sae: SparseAutoEncoder,
    activation_store: ActivationStore,
    cfg: LanguageModelSAEAnalysisConfig,
):
    total_analyzing_tokens = cfg.total_analyzing_tokens
    total_analyzing_steps = total_analyzing_tokens // cfg.effective_batch_size

    print_once(f"Total Analyzing Tokens: {total_analyzing_tokens}")
    print_once(f"Total Analyzing Steps: {total_analyzing_steps}")

    n_training_steps = 0
    n_training_tokens = 0

    sae_module = sae
    if cfg.use_ddp:
        sae = DDP(sae, device_ids=[cfg.rank], output_device=cfg.device)
        sae_module: SparseAutoEncoder = sae.module

    sae.eval()

    if not cfg.use_ddp or cfg.rank == 0:
        pbar = tqdm(total=total_analyzing_tokens, desc="Sampling activations", smoothing=0.01)

    elt = torch.empty((0, cfg.d_sae), dtype=cfg.dtype, device=cfg.device)
    feature_acts = torch.empty((0, cfg.d_sae), dtype=cfg.dtype, device=cfg.device)
    contexts = torch.empty((0, cfg.d_sae, cfg.context_size), dtype=torch.long, device=cfg.device)
    positions = torch.empty((0, cfg.d_sae), dtype=torch.long, device=cfg.device)

    with torch.no_grad():
        while n_training_tokens < total_analyzing_tokens:
            batch = activation_store.next(batch_size=cfg.analysis_batch_size)

            (
                _,
                (_, aux_data),
            ) = sae_module.forward(batch["activation"])

            weights = aux_data["feature_acts"].clamp(min=0.0).pow(2)
            elt_cur = torch.randn(batch["activation"].size(0), cfg.d_sae, device=cfg.device, dtype=cfg.dtype).sqrt() / weights
            elt_cur[weights == 0.0] = -torch.inf
            elt = torch.cat([elt, elt_cur], dim=0)
            feature_acts = torch.cat([feature_acts, aux_data["feature_acts"]], dim=0)
            contexts = torch.cat([contexts, repeat(batch["context"], 'b c -> b d c', d=cfg.d_sae)], dim=0)
            positions = torch.cat([positions, repeat(batch["position"], 'b -> b d', d=cfg.d_sae)], dim=0)

            # Sort elt, and extract the top n_samples
            elt, idx = torch.sort(elt, dim=0, descending=True)
            elt = elt[:cfg.n_samples]
            idx = idx[:cfg.n_samples]
            feature_acts = feature_acts.gather(1, idx)
            contexts = contexts.gather(1, idx.unsqueeze(-1).expand(-1, -1, contexts.size(-1)))
            positions = positions.gather(1, idx)

            n_tokens_current = torch.tensor(batch["activation"].size(0), device=cfg.device, dtype=torch.int)
            if cfg.use_ddp:
                dist.reduce(n_tokens_current, dst=0)
            n_training_tokens += n_tokens_current.item()

            n_training_steps += 1

            if not cfg.use_ddp or cfg.rank == 0:
                pbar.update(n_tokens_current.item())

    if not cfg.use_ddp or cfg.rank == 0:
        pbar.close()

    if cfg.use_ddp:
        print_once("Gathering top feature activations")

        if cfg.rank == 0:
            all_elt = [torch.empty_like(elt) for _ in range(cfg.world_size)]
            all_feature_acts = [torch.empty_like(feature_acts) for _ in range(cfg.world_size)]
            all_contexts = [torch.empty_like(contexts) for _ in range(cfg.world_size)]
            all_positions = [torch.empty_like(positions) for _ in range(cfg.world_size)]

        dist.gather(elt, all_elt if cfg.rank == 0 else None, dst=0)
        dist.gather(feature_acts, all_feature_acts if cfg.rank == 0 else None, dst=0)
        dist.gather(contexts, all_contexts if cfg.rank == 0 else None, dst=0)
        dist.gather(positions, all_positions if cfg.rank == 0 else None, dst=0)

        if cfg.rank == 0:
            elt = torch.cat(all_elt, dim=0)
            feature_acts = torch.cat(all_feature_acts, dim=0)
            contexts = torch.cat(all_contexts, dim=0)
            positions = torch.cat(all_positions, dim=0)

            elt, idx = torch.sort(elt, dim=0, descending=True)
            elt = elt[:cfg.n_samples]
            idx = idx[:cfg.n_samples]
            feature_acts = feature_acts.gather(1, idx)
            contexts = contexts.gather(1, idx.unsqueeze(-1).expand(-1, -1, contexts.size(-1)))
            positions = positions.gather(1, idx)

    if not cfg.use_ddp or cfg.rank == 0:
        torch.save(
            {
                "elt": elt,
                "feature_acts": feature_acts,
                "contexts": contexts,
                "positions": positions,
            },
            cfg.analysis_save_path,
        )

