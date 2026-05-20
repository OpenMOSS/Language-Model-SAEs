from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from llamascopium import (
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

from llamascopium.utils.evo2_hooks import (
    default_activation_dir,
    default_result_dir,
    get_evo2_arch,
    tc_hook_points_for_layer,
    validate_layer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an Evo2 transcoder.")
    parser.add_argument("--model-name", type=str, default="evo2_7b")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--layer", type=int, default=31)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--exp-factor", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--activations-path", type=str, default=str(default_activation_dir("tc_1d")))
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--total-training-tokens", type=int, default=10_000_000)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    arch = get_evo2_arch(args.model_name)
    d_model = int(arch["hidden_size"])
    validate_layer(args.layer, int(arch["num_layers"]))
    hook_point_in, hook_point_out = tc_hook_points_for_layer(args.layer)

    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    seed = 42
    print(f"[INFO] Using seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    run_name = f"{args.model_name}_L{args.layer}_tc_{args.exp_factor}x_k{args.k}_lr{args.lr:.0e}"
    result_dir = Path(default_result_dir("tc")) / run_name

    settings = TrainSAESettings(
        sae=SAEConfig(
            hook_point_in=hook_point_in,
            hook_point_out=hook_point_out,
            d_model=d_model,
            proj_data=True,
            expansion_factor=args.exp_factor,
            act_fn="topk",
            top_k=args.k,
            norm_activation="dataset-wise",
            sparsity_include_decoder_norm=True,
            dtype=torch.float32,
            device=args.device,
            use_auxk=True,
            k_aux=512,
        ),
        initializer=InitializerConfig(
            state="training",
            init_search=True,
            bias_init_method="geometric_median",
            init_encoder_with_decoder_transpose=False,
            decoder_uniform_bound=(d_model * args.exp_factor) ** (-0.5),
            encoder_uniform_bound=d_model ** (-0.5),
        ),
        trainer=TrainerConfig(
            use_batch_norm_mse=False,
            initial_k=d_model / 2,
            k_warmup_steps=0.1,
            lr=args.lr,
            lr_scheduler_name="constantwithwarmup",
            lr_warm_up_steps=100,
            lr_cool_down_steps=0.2,
            total_training_tokens=args.total_training_tokens,
            log_frequency=10,
            feature_sampling_window=100,
            eval_frequency=1_000_000,
            n_checkpoints=0,
            check_point_save_mode="linear",
            exp_result_path=str(result_dir),
        ),
        wandb=WandbConfig(
            wandb_project="evo2_tc",
            wandb_entity="fnlp-mechinterp",
            exp_name=run_name,
        ),
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryActivationsSource(
                    path=args.activations_path,
                    sample_weights=1.0,
                    name="opengenome_sae",
                    device=args.device,
                    dtype=torch.float32,
                    prefetch=args.prefetch,
                )
            ],
            target=ActivationFactoryTarget.ACTIVATIONS_1D,
            hook_points=[hook_point_in, hook_point_out],
            batch_size=args.batch_size,
            buffer_size=None,
            ignore_token_ids=[],
        ),
        sae_name=run_name,
        sae_series=f"{args.model_name}-tc",
        model_parallel_size=args.tp,
        data_parallel_size=args.dp,
        device_type="cpu" if args.device == "cpu" else "cuda",
    )
    train_sae(settings)
