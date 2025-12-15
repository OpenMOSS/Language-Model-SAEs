import argparse
import math
import os

import torch

from lm_saes import (
    ActivationFactoryActivationsSource,
    ActivationFactoryConfig,
    ActivationFactoryTarget,
    InitializerConfig,
    SAEConfig,
    TrainerConfig,
    TrainSAESettings,
    WandbConfig,
    train_sae,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="160m")
    parser.add_argument("--step", type=str, default="main")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--l1_coefficient", type=float, default=0.005)
    parser.add_argument("--expansion_factor", type=int, default=8)
    parser.add_argument("--init_encoder_factor", type=float, default=1)
    parser.add_argument("--sparsity_loss_type", type=str, default="tanh-quad")
    parser.add_argument("--jumprelu_lr_factor", type=float, default=1.0)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--name-suffix", type=str, default="")
    return parser.parse_args()


d_model_map = {
    "70m": 512,
    "160m": 768,
    "410m": 1024,
    "1b": 2048,
    "1.4b": 2048,
    "2.8b": 2048,
    "6.9b": 4096,
    "12b": 5120,
}

n_layers_map = {
    "70m": 6,
    "160m": 12,
    "410m": 24,
    "1b": 16,
    "1.4b": 24,
    "2.8b": 32,
    "6.9b": 32,
    "12b": 36,
}

if __name__ == "__main__":
    args = parse_args()

    name = f"L{args.layer}R-lr{args.lr}-l1c{args.l1_coefficient}-{args.expansion_factor}x-{args.step}"
    if args.init_encoder_factor != 1:
        name += f"-ie{args.init_encoder_factor}"
    if args.sparsity_loss_type != "tanh-quad":
        name += f"-{args.sparsity_loss_type}"
    if args.jumprelu_lr_factor != 1.0:
        name += f"-jlr{args.jumprelu_lr_factor}"
    if args.name_suffix:
        name += f"-{args.name_suffix}"

    exp_result_path = os.path.expanduser(f"~/results/pythia-{args.size}/{name}")


if __name__ == "__main__":
    settings = TrainSAESettings(
        sae=SAEConfig(
            hook_point_in=f"blocks.{args.layer}.hook_resid_post",
            hook_point_out=f"blocks.{args.layer}.hook_resid_post",
            d_model=d_model_map[args.size],
            expansion_factor=args.expansion_factor,
            act_fn="jumprelu",
            jumprelu_threshold_window=4.0,
            dtype=torch.float32,
            device="cuda",
        ),
        initializer=InitializerConfig(
            grid_search_init_norm=True,
            init_log_jumprelu_threshold_value=math.log(0.1),
            init_encoder_with_decoder_transpose_factor=args.init_encoder_factor,
        ),
        trainer=TrainerConfig(
            lr=args.lr,
            l1_coefficient=args.l1_coefficient,
            total_training_tokens=800_000_000,
            log_frequency=2000,
            eval_frequency=1000000,
            n_checkpoints=0,
            sparsity_loss_type="tanh-quad",
            check_point_save_mode="linear",
            exp_result_path=exp_result_path,
            jumprelu_lr_factor=args.jumprelu_lr_factor,
        ),
        wandb=WandbConfig(
            wandb_project="pythia-crosscoder",
            exp_name=name,
        ),
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryActivationsSource(
                    path=os.path.expanduser(
                        f"~/activations/SlimPajama-3B-activations-pythia-{args.size}-1d-all-fp16/{args.step}"
                    ),
                    sample_weights=1.0,
                    name=args.step,
                    device="cuda",
                    dtype=torch.float32,
                )
            ],
            target=ActivationFactoryTarget.ACTIVATIONS_1D,
            hook_points=[f"blocks.{args.layer}.hook_resid_post"],
            batch_size=4096,
            buffer_size=None,
            ignore_token_ids=[0],
        ),
        sae_name=name,
        sae_series="pythia-crosscoder",
    )
    train_sae(settings)
