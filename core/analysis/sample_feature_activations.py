from tqdm import tqdm

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from einops import repeat, rearrange

from datasets import Dataset

from transformer_lens import HookedTransformer

from core.sae import SparseAutoEncoder
from core.config import LanguageModelSAEAnalysisConfig
from core.activation.activation_store import ActivationStore
from core.utils.misc import print_once
from core.utils.tensor_dict import concat_dict_of_tensor, sort_dict_of_tensor

@torch.no_grad()
def sample_feature_activations(
    sae: SparseAutoEncoder,
    model: HookedTransformer,
    activation_store: ActivationStore,
    cfg: LanguageModelSAEAnalysisConfig,
):
    total_analyzing_tokens = cfg.total_analyzing_tokens
    total_analyzing_steps = total_analyzing_tokens // cfg.store_batch_size // cfg.context_size

    print_once(f"Total Analyzing Tokens: {total_analyzing_tokens}")
    print_once(f"Total Analyzing Steps: {total_analyzing_steps}")

    n_training_steps = 0
    n_training_tokens = 0

    if cfg.use_ddp:
        sae = DDP(sae, device_ids=[cfg.rank], output_device=cfg.device)

    sae.eval()

    if not cfg.use_ddp or cfg.rank == 0:
        pbar = tqdm(total=total_analyzing_tokens, desc="Sampling activations", smoothing=0.01)

    sample_result = {
        "weights": torch.empty((0, cfg.d_sae), dtype=cfg.dtype, device=cfg.device),
        "elt": torch.empty((0, cfg.d_sae), dtype=cfg.dtype, device=cfg.device),
        "feature_acts": torch.empty((0, cfg.d_sae, cfg.context_size), dtype=cfg.dtype, device=cfg.device),
        "contexts": torch.empty((0, cfg.d_sae, cfg.context_size), dtype=torch.long, device=cfg.device),
    }
    act_times = torch.zeros((cfg.d_sae,), dtype=torch.long, device=cfg.device)
    feature_act_bins = torch.arange(0, (cfg.n_bins + 1) * cfg.bin_width, cfg.bin_width, device=cfg.device, dtype=cfg.dtype)
    feature_act_hist = torch.zeros((cfg.d_sae, cfg.n_bins), dtype=torch.long, device=cfg.device)

    sort_key = "elt" if cfg.enable_sampling else "weights"

    while n_training_tokens < total_analyzing_tokens:
        batch = activation_store.next_tokens(cfg.store_batch_size)

        if batch is None:
            raise ValueError("Not enough tokens to sample")
        
        _, cache = model.run_with_cache(batch, names_filter=[cfg.hook_point])
        activations = cache[cfg.hook_point].to(dtype=cfg.dtype, device=cfg.device)

        (
            _,
            (_, aux_data),
        ) = sae.forward(activations)

        act_times += aux_data["feature_acts"].gt(0.0).sum(dim=[0, 1])

        weights = aux_data["feature_acts"].clamp(min=0.0).pow(cfg.sample_weight_exponent).max(dim=1).values
        elt = torch.rand(batch.size(0), cfg.d_sae, device=cfg.device, dtype=cfg.dtype).log() / weights
        elt[weights == 0.0] = -torch.inf
        sample_result = concat_dict_of_tensor(
            sample_result,
            {
                "weights": weights,
                "elt": elt,
                "feature_acts": rearrange(aux_data["feature_acts"], 'batch_size context_size d_sae -> batch_size d_sae context_size'),
                "contexts": repeat(batch, 'batch_size context_size -> batch_size d_sae context_size', d_sae=cfg.d_sae),
            },
            dim=0,
        )

        # Sort elt, and extract the top n_samples
        sample_result = sort_dict_of_tensor(sample_result, sort_dim=0, sort_key=sort_key, descending=True)
        sample_result = {k: v[:cfg.n_samples] for k, v in sample_result.items()}

        # Update feature activation histogram
        feature_act_hist += torch.histc(
            aux_data["feature_acts"].clamp(min=0.0).flatten(),
            bins=cfg.n_bins,
            min=0.0,
            max=(cfg.n_bins + 1) * cfg.bin_width,
        ).view(cfg.d_sae, cfg.n_bins)

        n_tokens_current = torch.tensor(batch.size(0) * batch.size(1), device=cfg.device, dtype=torch.int)
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
            all_result = {
                k: [torch.empty_like(v) for _ in range(cfg.world_size)] for k, v in sample_result.items()
            }

        for k in sample_result:
            dist.gather(sample_result[k], all_result[k], dst=0)

        dist.reduce(act_times, dst=0)
        dist.reduce(feature_act_hist, dst=0)

        if cfg.rank == 0:
            sample_result = {
                k: torch.cat(v, dim=0) for k, v in all_result.items()
            }
            sample_result = sort_dict_of_tensor(sample_result, sort_dim=0, sort_key=sort_key, descending=True)

    if not cfg.use_ddp or cfg.rank == 0:
        sample_result = {
            k: rearrange(v, 'n_samples d_sae ... -> d_sae n_samples ...') for k, v in sample_result.items()
        }

        result = {
            "act_times": act_times,
            "feature_act_bins": feature_act_bins,
            "feature_act_hist": feature_act_hist,
            **sample_result,
        }

        Dataset.from_dict(result).save_to_disk(cfg.analysis_save_path, num_shards=1024)