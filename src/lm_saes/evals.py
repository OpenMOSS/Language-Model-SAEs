from typing import Any, cast

import pandas as pd
import torch
import torch.distributed as dist
import wandb
from tqdm import tqdm
from transformer_lens import HookedTransformer

from lm_saes.activation.activation_store import ActivationStore

# from lm_saes.activation_store_theirs import ActivationStoreTheirs
from lm_saes.config import LanguageModelSAERunnerConfig
from lm_saes.sae import SparseAutoEncoder
from lm_saes.utils.misc import is_master


@torch.no_grad()
def run_evals(
    model: HookedTransformer,
    sae: SparseAutoEncoder,
    activation_store: ActivationStore,
    cfg: LanguageModelSAERunnerConfig,
    n_training_steps: int,
):
    # Get Reconstruction Score
    losses_df = recons_loss_batched(
        model,
        sae,
        activation_store,
        cfg,
        n_batches=50,
    )

    ntp_loss = losses_df["loss"].mean()
    recons_loss = losses_df["recons_loss"].mean()

    # Standard Evals
    batch = activation_store.next(batch_size=4096)

    assert batch is not None, "Activation store is empty"

    activation_in, activation_out = (
        batch[cfg.sae.hook_point_in],
        batch[cfg.sae.hook_point_out],
    )

    feature_acts = sae.encode(activation_in)
    reconstructed = sae.decode(feature_acts)

    per_token_l2_loss = (reconstructed - activation_out).pow(2).sum(dim=-1)
    total_variance = (activation_out - activation_out.mean(0)).pow(2).sum(dim=-1)

    if "cuda" in str(model.cfg.device):
        torch.cuda.empty_cache()

    l2_norm_in = torch.norm(activation_out, dim=-1)
    l2_norm_out = torch.norm(reconstructed, dim=-1)
    if cfg.sae.ddp_size > 1:
        dist.reduce(l2_norm_in, dst=0, op=dist.ReduceOp.AVG)
        dist.reduce(l2_norm_out, dst=0, op=dist.ReduceOp.AVG)
    l2_norm_ratio = l2_norm_out / l2_norm_in

    l0 = (feature_acts > 0).float().sum(-1)
    explained_variance = 1 - per_token_l2_loss / total_variance

    metrics = {
        # l2 norms
        "metrics/l2_norm_reconstructed": l2_norm_out.mean().item(),
        "metrics/l2_norm_original": l2_norm_in.mean().item(),
        "metrics/l2_ratio": l2_norm_ratio.mean().item(),
        # variance explained & L0
        "metrics/explained_variance": explained_variance.mean().item(),
        "metrics/l0": l0.mean().item(),
        # CE Loss
        "metrics/delta_ce_loss": recons_loss - ntp_loss,
        "metrics/ce_loss_without_sae": ntp_loss,
        "metrics/ce_loss_with_sae": recons_loss,
    }

    if cfg.wandb.log_to_wandb and is_master():
        wandb.log(
            metrics,
            step=n_training_steps + 1,
        )

    return metrics


def recons_loss_batched(
    model: HookedTransformer,
    sae: SparseAutoEncoder,
    activation_store: ActivationStore,
    cfg: LanguageModelSAERunnerConfig,
    n_batches: int = 100,
):
    losses = []
    if is_master():
        pbar = tqdm(total=n_batches, desc="Evaluation", smoothing=0.01)
    for _ in range(n_batches):
        batch_tokens = activation_store.next_tokens(cfg.act_store.dataset.store_batch_size)
        assert batch_tokens is not None, "Not enough tokens in the store"
        loss, recons_loss = get_recons_loss(model, sae, cfg, batch_tokens)
        if cfg.sae.ddp_size > 1:
            dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)
            dist.reduce(recons_loss, dst=0, op=dist.ReduceOp.AVG)
        losses.append(
            (
                loss.mean().item(),
                recons_loss.mean().item(),
            )
        )
        if is_master():
            pbar.update(1)

    if is_master():
        pbar.close()

    losses = pd.DataFrame(losses, columns=cast(Any, ["loss", "recons_loss"]))

    return losses


@torch.no_grad()
def get_recons_loss(
    model: HookedTransformer,
    sae: SparseAutoEncoder,
    cfg: LanguageModelSAERunnerConfig,
    batch_tokens: torch.Tensor,
):
    batch_tokens = batch_tokens.to(torch.int64)

    logits_mask = torch.logical_and(
        batch_tokens.ne(model.tokenizer.eos_token_id), batch_tokens.ne(model.tokenizer.pad_token_id)
    )
    logits_mask = torch.logical_and(logits_mask, batch_tokens.ne(model.tokenizer.bos_token_id))

    loss, cache = model.run_with_cache(
        batch_tokens,
        return_type="loss",
        loss_per_token=True,
        names_filter=[cfg.sae.hook_point_in, cfg.sae.hook_point_out],
    )

    # _, cache = model.run_with_cache_until(
    #     batch_tokens,
    #     names_filter=[cfg.sae.hook_point_in, cfg.sae.hook_point_out],
    #     until=cfg.sae.hook_point_out,
    # )
    activations_in, activations_out = (
        cache[cfg.sae.hook_point_in],
        cache[cfg.sae.hook_point_out],
    )
    replacements = sae.forward(activations_in).to(activations_out.dtype)

    def replacement_hook(activations: torch.Tensor, hook: Any):
        return replacements.where(logits_mask[..., None], activations)

    recons_loss: torch.Tensor = model.run_with_hooks(
        batch_tokens, return_type="loss", fwd_hooks=[(cfg.sae.hook_point_out, replacement_hook)], loss_per_token=True
    )

    def get_useful_token_loss(per_token_loss):
        per_token_loss = per_token_loss.where(logits_mask[:, 1:], 0)
        return per_token_loss.sum() / per_token_loss.ne(0).sum()

    loss, recons_loss = get_useful_token_loss(loss), get_useful_token_loss(recons_loss)

    return loss, recons_loss


def zero_ablate_hook(activations: torch.Tensor, hook: Any):
    activations = torch.zeros_like(activations)
    return activations
