import math
import os
from typing import cast

import torch
import torch.distributed as dist
import wandb
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from tqdm import tqdm
from transformer_lens import HookedTransformer

from .activation.activation_store import ActivationStore
from .config import LanguageModelSAEPruningConfig, LanguageModelSAETrainingConfig
from .evals import run_evals
from .optim import get_scheduler
from .sae import SparseAutoEncoder
from .utils.misc import is_master, print_once


def train_sae(
    sae: SparseAutoEncoder,
    activation_store: ActivationStore,
    cfg: LanguageModelSAETrainingConfig,
    model: HookedTransformer | None = None,
):
    total_training_tokens = cfg.total_training_tokens
    total_training_steps = total_training_tokens // cfg.effective_batch_size

    print_once(f"Total Training Tokens: {total_training_tokens}")
    print_once(f"Total Training Steps: {total_training_steps}")

    n_training_steps = 0
    n_training_tokens = 0
    log_feature_sparsity = None

    activation_store.initialize()
    if is_master():
        print("Activation Store Initialized.")
    # Initialize the SAE decoder bias if necessary
    # if cfg.use_decoder_bias and (not cfg.use_ddp or cfg.rank == 0):
    #     sae.initialize_decoder_bias(activation_store._store[cfg.hook_point_in])

    act_freq_scores = torch.zeros(cfg.sae.d_sae, device=cfg.sae.device, dtype=cfg.sae.dtype)
    n_forward_passes_since_fired = torch.zeros(cfg.sae.d_sae, device=cfg.sae.device, dtype=cfg.sae.dtype)
    n_frac_active_tokens = torch.tensor([0], device=cfg.sae.device, dtype=torch.int)

    if cfg.sae.tp_size > 1:
        plan = {
            "encoder": ColwiseParallel(output_layouts=Replicate()),
            "decoder": RowwiseParallel(input_layouts=Replicate()),
        }
        if cfg.sae.use_glu_encoder:
            plan["encoder_glu"] = ColwiseParallel(output_layouts=Replicate())
        sae = parallelize_module(sae, device_mesh=sae.device_mesh["tp"], parallelize_plan=plan)  # type: ignore
        sae.tensor_paralleled = True

    elif cfg.sae.ddp_size > 1:
        # parallelize_module does not work with DDP
        _ = DDP(sae, device_mesh=sae.device_mesh["ddp"])

    optimizer = Adam(sae.parameters(), lr=cfg.lr, betas=cfg.betas)

    scheduler = get_scheduler(
        cfg.lr_scheduler_name,
        optimizer=optimizer,
        warm_up_steps=cfg.lr_warm_up_steps,
        cool_down_steps=cfg.lr_cool_down_steps,
        training_steps=total_training_steps,
        lr_end_ratio=cfg.lr_end_ratio,
    )

    scheduler.step()

    pbar = tqdm(total=total_training_tokens, desc="Training SAE", smoothing=0.01) if is_master() else None
    while n_training_tokens < total_training_tokens:
        sae.train()
        if not sae.cfg.act_fn == "topk":
            sae.update_l1_coefficient(n_training_steps)
        else:
            sae.update_k(n_training_steps)
        # Get the next batch of activations

        batch = activation_store.next(batch_size=cfg.train_batch_size)
        assert batch is not None, "Activation store is empty"
        activation_in, activation_out = (
            batch[cfg.sae.hook_point_in],
            batch[cfg.sae.hook_point_out],
        )

        scheduler.step()
        optimizer.zero_grad()

        ghost_grad_neuron_mask = (n_forward_passes_since_fired > cfg.dead_feature_window).bool()
        # Forward pass
        (
            loss,
            (
                loss_data,
                aux_data,
            ),
        ) = sae.compute_loss(
            activation_in,
            dead_feature_mask=ghost_grad_neuron_mask,
            label=activation_out,
        )

        did_fire = (aux_data["feature_acts"] > 0).float().sum(0) > 0
        n_forward_passes_since_fired += 1
        n_forward_passes_since_fired[did_fire] = 0
        if cfg.sae.ddp_size > 1:
            dist.all_reduce(
                n_forward_passes_since_fired,
                op=dist.ReduceOp.MIN,
            )

        if cfg.finetuning:
            loss = loss_data["l_rec"].mean()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            sae.parameters(),
            max_norm=cfg.clip_grad_norm if cfg.clip_grad_norm > 0 else math.inf,
        )

        if cfg.remove_gradient_parallel_to_decoder_directions:
            sae.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()
        if not cfg.sae.sparsity_include_decoder_norm:
            sae.set_decoder_norm_to_fixed_norm(1)
        with torch.no_grad():
            act_freq_scores += (aux_data["feature_acts"].abs() > 0).float().sum(0)
            n_frac_active_tokens += activation_in.size(0)

            n_tokens_current = torch.tensor(activation_in.size(0), device=cfg.sae.device, dtype=torch.int)
            if cfg.sae.ddp_size > 1:
                dist.reduce(n_tokens_current, dst=0)
            n_training_tokens += cast(int, n_tokens_current.item())

            # log and then reset the feature sparsity every feature_sampling_window steps
            if (n_training_steps + 1) % cfg.feature_sampling_window == 0:
                if cfg.sae.ddp_size > 1:
                    dist.reduce(act_freq_scores, dst=0)
                    dist.reduce(n_frac_active_tokens, dst=0)
                if cfg.wandb.log_to_wandb and (cfg.wandb.log_on_every_rank or is_master()):
                    feature_sparsity = act_freq_scores / n_frac_active_tokens
                    log_feature_sparsity = torch.log10(feature_sparsity + 1e-10)

                    log_dict = {
                        "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
                        # "plots/feature_density_line_chart": wandb_histogram,
                        "sparsity/below_1e-5": (feature_sparsity < 1e-5).sum().item(),
                        "sparsity/below_1e-6": (feature_sparsity < 1e-6).sum().item(),
                    }

                    if cfg.wandb.log_on_every_rank:
                        dist.all_reduce(feature_sparsity, op=dist.ReduceOp.MAX)
                        log_dict.update(
                            {
                                "sparsity/overall_below_1e-5": (feature_sparsity < 1e-5).sum().item(),
                                "sparsity/overall_below_1e-6": (feature_sparsity < 1e-6).sum().item(),
                            }
                        )

                    wandb.log(log_dict, step=n_training_steps + 1)

                act_freq_scores = torch.zeros(cfg.sae.d_sae, device=cfg.sae.device)
                n_frac_active_tokens = torch.tensor([0], device=cfg.sae.device, dtype=torch.int)

            if (n_training_steps + 1) % cfg.log_frequency == 0:
                # metrics for currents acts
                l0 = (aux_data["feature_acts"] > 0).float().sum(-1).mean()
                l_rec = loss_data["l_rec"].mean()
                l_l1 = (
                    loss_data["l_l1"].mean() if not cfg.sae.act_fn == "topk" else torch.tensor(0, device=cfg.sae.device)
                )
                l_ghost_resid = (
                    loss_data["l_ghost_resid"].mean()
                    if cfg.sae.use_ghost_grads
                    else torch.tensor(0, device=cfg.sae.device)
                )

                if cfg.sae.ddp_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                    dist.all_reduce(l0, op=dist.ReduceOp.AVG)
                    dist.all_reduce(l_rec, op=dist.ReduceOp.AVG)
                    if not cfg.sae.act_fn == "topk":
                        dist.all_reduce(l_l1, op=dist.ReduceOp.AVG)
                    if cfg.sae.use_ghost_grads:
                        dist.all_reduce(l_ghost_resid, op=dist.ReduceOp.AVG)

                per_token_l2_loss = (aux_data["reconstructed"] - activation_out).pow(2).sum(dim=-1)
                total_variance = (activation_out - activation_out.mean(0)).pow(2).sum(dim=-1)

                l2_norm_error = per_token_l2_loss.sqrt().mean()
                l2_norm_error_ratio = l2_norm_error / activation_out.norm(p=2, dim=-1).mean()

                if cfg.sae.ddp_size > 1:
                    dist.all_reduce(l2_norm_error, op=dist.ReduceOp.AVG)
                    dist.all_reduce(l2_norm_error_ratio, op=dist.ReduceOp.AVG)

                    # Replace gather with all_gather
                    per_token_l2_loss_list = [torch.zeros_like(per_token_l2_loss) for _ in range(dist.get_world_size())]
                    total_variance_list = [torch.zeros_like(total_variance) for _ in range(dist.get_world_size())]

                    dist.all_gather(per_token_l2_loss_list, per_token_l2_loss)
                    dist.all_gather(total_variance_list, total_variance)

                    per_token_l2_loss = torch.cat(per_token_l2_loss_list, dim=0)
                    total_variance = torch.cat(total_variance_list, dim=0)

                explained_variance = 1 - per_token_l2_loss / total_variance

                # mean_thomson_potential = sae_module.compute_thomson_potential()

                current_learning_rate = optimizer.param_groups[0]["lr"]

                if cfg.wandb.log_to_wandb:
                    decoder_norm = sae.decoder_norm().mean()
                    encoder_norm = sae.encoder_norm().mean()
                    logs = {
                        # losses
                        "losses/mse_loss": l_rec.item(),
                        "losses/overall_loss": loss.item(),
                        # variance explained
                        "metrics/explained_variance": explained_variance.mean().item(),
                        "metrics/explained_variance_std": explained_variance.std().item(),
                        "metrics/l0": l0.item(),
                        # "metrics/mean_thomson_potential": mean_thomson_potential.item(),
                        "metrics/l2_norm_error": l2_norm_error.item(),
                        "metrics/l2_norm_error_ratio": l2_norm_error_ratio.item(),
                        # norm
                        "metrics/decoder_norm": decoder_norm.item(),
                        "metrics/encoder_norm": encoder_norm.item(),
                        "metrics/decoder_bias_norm": (
                            sae.decoder.bias.norm().item() if sae.cfg.use_decoder_bias else 0
                        ),
                        "metrics/encoder_bias_norm": sae.encoder.bias.norm().item(),
                        "metrics/gradients_norm": grad_norm.item(),
                        # sparsity
                        "sparsity/mean_passes_since_fired": n_forward_passes_since_fired.mean().item(),
                        "sparsity/dead_features": ghost_grad_neuron_mask.sum().item(),
                        "details/current_learning_rate": current_learning_rate,
                        "details/n_training_tokens": n_training_tokens,
                    }
                    if cfg.sae.use_ghost_grads:
                        logs["losses/ghost_resid_loss"] = l_ghost_resid.item()
                    if not cfg.sae.act_fn == "topk":
                        assert l_l1 is not None, "L1 loss is None"
                        logs["losses/l1_loss"] = l_l1.item()
                        logs["sparsity/l1_coefficient"] = sae.current_l1_coefficient
                    else:
                        logs["sparsity/k"] = sae.current_k

                    if cfg.wandb.log_on_every_rank or is_master():
                        wandb.log(
                            logs,
                            step=n_training_steps + 1,
                        )

            # record loss frequently, but not all the time.
            if (n_training_steps + 1) % (cfg.eval_frequency) == 0:
                if model is None:
                    raise NotImplementedError("Evaluation for model-free training is not implemented yet.")
                sae.eval()
                run_evals(
                    sae=sae,
                    activation_store=activation_store,
                    model=model,
                    cfg=cfg,
                    n_training_steps=n_training_steps,
                )
                sae.train()

            # Checkpoint if at checkpoint frequency
            if len(cfg.checkpoint_thresholds) > 0 and n_training_tokens >= cfg.checkpoint_thresholds[0]:
                # Save the model and optimizer state
                path = os.path.join(
                    cfg.exp_result_path,
                    "checkpoints",
                    f"{n_training_steps}.safetensors",
                )
                if not cfg.sae.sparsity_include_decoder_norm:
                    sae.set_decoder_norm_to_fixed_norm(1)
                sae.save_pretrained(path)
                cfg.checkpoint_thresholds.pop(0)

            n_training_steps += 1

            if pbar is not None:
                l_rec = loss_data["l_rec"].mean()
                desc = f"{n_training_steps}| MSE Loss {l_rec.item():.3f}"
                if not cfg.sae.act_fn == "topk":
                    l_l1 = loss_data["l_l1"].mean()
                    desc += f" | L1 Loss {l_l1.item():.3f}"

                pbar.set_description(desc)
                pbar.update(n_tokens_current.item())

    if pbar is not None:
        pbar.close()

    # Save the final model
    if not cfg.sae.sparsity_include_decoder_norm:
        sae.set_decoder_norm_to_fixed_norm(1)
    path = os.path.join(cfg.exp_result_path, "checkpoints", "final.safetensors")
    sae.save_pretrained(path)


@torch.no_grad()
def prune_sae(
    sae: SparseAutoEncoder,
    activation_store: ActivationStore,
    cfg: LanguageModelSAEPruningConfig,
):
    sae.eval()
    n_training_tokens = 0
    act_times = torch.zeros(cfg.sae.d_sae, device=cfg.sae.device, dtype=torch.int)
    max_acts = torch.zeros(cfg.sae.d_sae, device=cfg.sae.device, dtype=cfg.sae.dtype)
    activation_store.initialize()

    if cfg.sae.ddp_size > 1:
        _ = DDP(sae, device_mesh=sae.device_mesh["ddp"])

    pbar = tqdm(total=cfg.total_training_tokens, desc="Pruning SAE", smoothing=0.01) if is_master() else None
    while n_training_tokens < cfg.total_training_tokens:
        # Get the next batch of activations
        batch = activation_store.next(batch_size=cfg.train_batch_size)
        assert batch is not None, "Activation store is empty"
        activation_in, _ = (
            batch[cfg.sae.hook_point_in],
            batch[cfg.sae.hook_point_out],
        )

        feature_acts = sae.encode(activation_in)

        act_times += (feature_acts > 0).int().sum(0)
        max_acts = torch.max(max_acts, feature_acts.max(0).values)

        n_tokens_current = activation_in.size(0)
        if cfg.sae.ddp_size > 1:
            dist.reduce(n_tokens_current, dst=0)
        n_training_tokens += n_tokens_current

        if pbar is not None:
            pbar.update(n_tokens_current)

    if pbar is not None:
        pbar.close()

    if cfg.sae.ddp_size > 1:
        dist.all_reduce(act_times, op=dist.ReduceOp.SUM)
        dist.all_reduce(max_acts, op=dist.ReduceOp.MAX)

    decoder_norm = sae.decoder_norm()
    if is_master():
        sae.feature_act_mask.data = (
            (act_times > cfg.dead_feature_threshold * cfg.total_training_tokens)
            & (max_acts > cfg.dead_feature_max_act_threshold)
            & (decoder_norm >= cfg.decoder_norm_threshold)
        ).to(cfg.sae.dtype)
        sae.feature_act_mask.requires_grad_(False)

        if cfg.wandb.log_to_wandb:
            wandb.log(
                {
                    "sparsity/dead_features": (act_times < cfg.dead_feature_threshold * cfg.total_training_tokens)
                    .sum()
                    .item(),
                    "sparsity/max_acts_below_threshold": (max_acts < cfg.dead_feature_max_act_threshold).sum().item(),
                    "sparsity/decoder_norm_below_threshold": (decoder_norm < cfg.decoder_norm_threshold).sum().item(),
                    "sparsity/total_pruned_features": (sae.feature_act_mask == 0).sum().item(),
                },
            )

        print(
            "Dead features:",
            (act_times < cfg.dead_feature_threshold * cfg.total_training_tokens).sum().item(),
        )
        print(
            "Max acts below threshold:",
            (max_acts < cfg.dead_feature_max_act_threshold).sum().item(),
        )
        print(
            "Decoder norm below threshold:",
            (decoder_norm < cfg.decoder_norm_threshold).sum().item(),
        )
        print("Total pruned features:", (sae.feature_act_mask == 0).sum().item())

        path = os.path.join(cfg.exp_result_path, "checkpoints", "pruned.safetensors")
        sae.save_pretrained(path)

    return sae
