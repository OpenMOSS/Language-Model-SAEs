import argparse
import math
import os
from pathlib import Path

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

    # Hyperparameters for the SAE architecture
    parser.add_argument("--size", type=str, default="160m")
    parser.add_argument("--expansion_factor", type=int, default=32)
    parser.add_argument("--layer", type=int)

    # Hyperparameters for the SAE training
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--l1_coefficient", type=float, default=0.3)
    parser.add_argument("--init_encoder_factor", type=float, default=1)
    parser.add_argument("--sparsity_loss_type", type=str, default="tanh-quad")
    parser.add_argument("--jumprelu_lr_factor", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--total_training_tokens", type=int, default=800_000_000)

    # Parallelism parameters
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)

    # Result & Activation paths
    parser.add_argument("--result_path", type=str)
    parser.add_argument("--activation_path", type=str)
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

if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    args = parse_args()

    name = f"L{args.layer}R-lr{args.lr}-l1c{args.l1_coefficient}-{args.expansion_factor}x-{args.step}"
    if args.jumprelu_lr_factor != 1.0:
        name += f"-jlr{args.jumprelu_lr_factor}"

    exp_result_path = Path(args.result_path).expanduser()
    exp_result_path.mkdir(parents=True, exist_ok=True)

    if int(os.environ.get("WORLD_SIZE", 1)) != args.dp * args.tp:
        raise ValueError(
            f"WORLD_SIZE ({int(os.environ.get('WORLD_SIZE', 1))}) must be equal to dp * tp ({args.dp * args.tp})"
        )

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
            sparsity_loss_type=args.sparsity_loss_type,
            use_batch_norm_mse=False,
            check_point_save_mode="linear",
            exp_result_path=exp_result_path,
            jumprelu_lr_factor=args.jumprelu_lr_factor,
        ),
        wandb=WandbConfig(
            wandb_project="lm-saes",
            exp_name=name,
        ),
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryActivationsSource(
                    path=Path(args.activation_path).expanduser(),
                    name=f"pythia-{args.size}-1d",
                    device="cuda",
                    dtype=torch.float32,
                )
            ],
            target=ActivationFactoryTarget.ACTIVATIONS_1D,
            hook_points=[f"blocks.{args.layer}.hook_resid_post"],
            batch_size=args.batch_size,
            buffer_size=None,
        ),
        sae_name=name,
        sae_series="pythia-sae",
        data_parallel_size=args.dp,
        model_parallel_size=args.tp,
    )
    train_sae(settings)
