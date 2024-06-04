import os

from tqdm import tqdm

import torch

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
    sae_chunk_id: int = 0,
    n_sae_chunks: int = 1,  # By default, we do not chunk the SAE. When the model & SAE is large, we can chunk the SAE to save memory.
):
    if cfg.use_ddp:
        raise ValueError("Sampling feature activations does not support DDP yet")

    total_analyzing_tokens = cfg.total_analyzing_tokens
    total_analyzing_steps = total_analyzing_tokens // cfg.store_batch_size // cfg.context_size

    print_once(f"Total Analyzing Tokens: {total_analyzing_tokens}")
    print_once(f"Total Analyzing Steps: {total_analyzing_steps}")

    n_training_steps = 0
    n_training_tokens = 0

    sae.eval()

    pbar = tqdm(total=total_analyzing_tokens, desc=f"Sampling activations of chunk {sae_chunk_id} of {n_sae_chunks}", smoothing=0.01)

    d_sae = cfg.d_sae // n_sae_chunks
    start_index = sae_chunk_id * d_sae
    end_index = (sae_chunk_id + 1) * d_sae

    hook_point_out = cfg.hook_point_out
    stop_at_layer = int(hook_point_out.split(".")[1]) + 1 # fuck this hard code

    sample_result = {k: {
        "elt": torch.empty((0, d_sae), dtype=cfg.dtype, device=cfg.device),
        "feature_acts": torch.empty((0, d_sae, cfg.context_size), dtype=cfg.dtype, device=cfg.device),
        "contexts": torch.empty((0, d_sae, cfg.context_size), dtype=torch.int32, device=cfg.device),
    } for k in cfg.subsample.keys()}
    act_times = torch.zeros((d_sae,), dtype=torch.long, device=cfg.device)
    feature_acts_all = [torch.empty((0,), dtype=cfg.dtype, device=cfg.device) for _ in range(d_sae)]
    max_feature_acts = torch.zeros((d_sae,), dtype=cfg.dtype, device=cfg.device)

    while n_training_tokens < total_analyzing_tokens:
        batch = activation_store.next_tokens(cfg.store_batch_size)

        if batch is None:
            raise ValueError("Not enough tokens to sample")

        _, cache = model.run_with_cache(batch, names_filter=[cfg.hook_point_in, cfg.hook_point_out], stop_at_layer=stop_at_layer)
        activation_in, activation_out = cache[cfg.hook_point_in], cache[cfg.hook_point_out]

        (
            _,
            (_, aux_data),
        ) = sae.forward(activation_in, label=activation_out)
        feature_acts = aux_data["feature_acts"][..., start_index: end_index]

        act_times += feature_acts.gt(0.0).sum(dim=[0, 1])

        for name in cfg.subsample.keys():

            if cfg.enable_sampling:
                weights = feature_acts.clamp(min=0.0).pow(cfg.sample_weight_exponent).max(dim=1).values
                elt = torch.rand(batch.size(0), d_sae, device=cfg.device, dtype=cfg.dtype).log() / weights
                elt[weights == 0.0] = -torch.inf
            else:
                elt = feature_acts.clamp(min=0.0).max(dim=1).values

            elt[feature_acts.max(dim=1).values > max_feature_acts.unsqueeze(0) * cfg.subsample[name]["proportion"]] = -torch.inf

            if sample_result[name]["elt"].size(0) > 0 and (elt.max(dim=0).values <= sample_result[name]["elt"][-1]).all():
                continue

            sample_result[name] = concat_dict_of_tensor(
                sample_result[name],
                {
                    "elt": elt,
                    "feature_acts": rearrange(feature_acts, 'batch_size context_size d_sae -> batch_size d_sae context_size'),
                    "contexts": repeat(batch.to(torch.int32), 'batch_size context_size -> batch_size d_sae context_size', d_sae=d_sae),
                },
                dim=0,
            )

            sample_result[name] = sort_dict_of_tensor(sample_result[name], sort_dim=0, sort_key="elt", descending=True)
            sample_result[name] = {
                k: v[:cfg.subsample[name]["n_samples"]] for k, v in sample_result[name].items()
            }


        # Update feature activation histogram every 10 steps
        if n_training_steps % 50 == 49:
            feature_acts_cur = rearrange(feature_acts, 'batch_size context_size d_sae -> d_sae (batch_size context_size)')
            for i in range(d_sae):
                feature_acts_all[i] = torch.cat([feature_acts_all[i], feature_acts_cur[i][feature_acts_cur[i] > 0.0]], dim=0)

        max_feature_acts = torch.max(max_feature_acts, feature_acts.max(dim=0).values.max(dim=0).values)

        n_tokens_current = torch.tensor(batch.size(0) * batch.size(1), device=cfg.device, dtype=torch.int)
        n_training_tokens += n_tokens_current.item()
        n_training_steps += 1

        pbar.update(n_tokens_current.item())

    pbar.close()

    sample_result = {k1: {
        k2: rearrange(v2, 'n_samples d_sae ... -> d_sae n_samples ...') for k2, v2 in v1.items()
    } for k1, v1 in sample_result.items()}

    result = {
        "index": torch.arange(start_index, end_index, device=cfg.device, dtype=torch.int32),
        "act_times": act_times,
        "feature_acts_all": feature_acts_all,
        "max_feature_acts": max_feature_acts,
        "analysis": [
            {
                "name": k,
                "feature_acts": v["feature_acts"],
                "contexts": v["contexts"],
            } for k, v in sample_result.items()
        ],
    }

    return result