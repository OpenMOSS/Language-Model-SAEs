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

    # Parallelism parameters
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)

    # Activation path
    parser.add_argument("--activation_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    args = parse_args()

    if int(os.environ.get("WORLD_SIZE", 1)) != args.dp * args.tp:
        raise ValueError(
            f"WORLD_SIZE ({os.environ.get('WORLD_SIZE', 1)}) must be equal to dp * tp ({args.dp * args.tp})"
        )

    settings = TrainSAESettings(
        sae=SAEConfig(
            hook_point_in="blocks.6.hook_resid_post",
            hook_point_out="blocks.6.hook_resid_post",
            d_model=768,
            expansion_factor=8,
            act_fn="jumprelu",
            jumprelu_threshold_window=4.0,
            dtype=torch.float32,
            device="cuda",
        ),
        initializer=InitializerConfig(
            grid_search_init_norm=True,
            init_log_jumprelu_threshold_value=math.log(0.1),
            init_encoder_with_decoder_transpose_factor=1.0,
        ),
        trainer=TrainerConfig(
            lr=5e-5,
            l1_coefficient=0.3,
            total_training_tokens=800_000_000,
            log_frequency=2000,
            eval_frequency=1000000,
            n_checkpoints=0,
            sparsity_loss_type="tanh-quad",
            check_point_save_mode="linear",
            exp_result_path="results",
            jumprelu_lr_factor=0.1,
        ),
        wandb=WandbConfig(
            wandb_project="lm-saes",
            exp_name="pythia-160m-sae",
        ),
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryActivationsSource(
                    path=str(Path(args.activation_path).expanduser()),
                    name="pythia-160m-1d",
                    device="cuda",
                    dtype=torch.float32,
                )
            ],
            target=ActivationFactoryTarget.ACTIVATIONS_1D,
            hook_points=["blocks.6.hook_resid_post"],
            batch_size=4096,
            buffer_size=None,
        ),
        sae_name="pythia-160m-sae",
        sae_series="pythia-sae",
        data_parallel_size=args.dp,
        model_parallel_size=args.tp,
    )
    train_sae(settings)
