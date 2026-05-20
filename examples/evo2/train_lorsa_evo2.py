from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from llamascopium import (
    ActivationFactoryActivationsSource,
    ActivationFactoryConfig,
    ActivationFactoryTarget,
    InitializerConfig,
    LanguageModelConfig,
    LorsaConfig,
    TrainerConfig,
    TrainLorsaSettings,
    WandbConfig,
    train_lorsa,
)

from llamascopium.utils.evo2_hooks import default_activation_dir, default_result_dir, get_evo2_arch, validate_layer
from llamascopium.utils.evo2_hooks import gen_evo2_hook_points_for_layer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Evo2 LORSA on attention layers.")
    parser.add_argument("--model-name", type=str, default="evo2_7b")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--exp-factor", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-qk-heads", type=int, default=None)
    parser.add_argument("--context-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--activations-path", type=str, default=str(default_activation_dir("lorsa_2d")))
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--total-training-tokens", type=int, default=100_000)
    parser.add_argument("--init-search", action="store_true")
    parser.add_argument("--initialize-lorsa-with-mhsa", action="store_true")
    parser.add_argument("--initialize-W-D-with-active-subspace", action="store_true")
    parser.add_argument("--use-smolgen", action="store_true")
    parser.add_argument("--initialize-lorsa-smolgen-from-encoder", action="store_true")
    parser.add_argument("--initialize-lorsa-attn-scale-from-encoder", action="store_true")
    parser.add_argument("--k-aux", type=int, default=512)
    parser.add_argument("--aux-coefficient", type=float, default=1 / 32)
    parser.add_argument("--dead-threshold", type=int, default=1_000_000)
    parser.add_argument("--log-to-wandb", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    arch = get_evo2_arch(args.model_name)
    validate_layer(args.layer, int(arch["num_layers"]))
    attn_layers = arch["attn_layers"]
    hook_points, layer_kind = gen_evo2_hook_points_for_layer(args.layer, args.model_name)
    if layer_kind != "lorsa":
        raise ValueError(
            f"LORSA requires an attention layer. Got layer {args.layer}. "
            f"Available Evo2 attention layers: {sorted(attn_layers)}"
        )

    d_model = int(arch["hidden_size"])
    n_qk_heads = args.n_qk_heads or int(arch["num_attention_heads"])
    d_qk_head = int(arch["d_qk_head"])
    rotary_dim = int(arch["rotary_dim"])
    hook_point_in, hook_point_out = hook_points

    print("=" * 80)
    print("config arguments of training")
    print("=" * 80)
    print(f"[model_name]: {args.model_name}")
    print(f"[lr]: {args.lr}")
    print(f"[layer]: {args.layer}")
    print(f"[k]: {args.k}")
    print(f"[exp_factor]: {args.exp_factor}")
    print(f"[d_model]: {d_model}")
    print(f"[n_qk_heads]: {n_qk_heads}")
    print(f"[d_qk_head]: {d_qk_head}")
    print(f"[rotary_dim]: {rotary_dim}")
    print(f"[context_size]: {args.context_size}")
    print("=" * 80)

    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    seed = 42
    print(f"[INFO] Using seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    run_name = f"{args.model_name}_L{args.layer}_lorsa_{args.exp_factor}x_k{args.k}_lr{args.lr:.0e}"
    result_dir = Path(default_result_dir("lorsa")) / run_name

    settings = TrainLorsaSettings(
        sae=LorsaConfig(
            hook_point_in=hook_point_in,
            hook_point_out=hook_point_out,
            d_model=d_model,
            expansion_factor=args.exp_factor,
            n_qk_heads=n_qk_heads,
            act_fn="topk",
            d_qk_head=d_qk_head,
            rotary_dim=rotary_dim,
            n_ctx=args.context_size,
            top_k=args.k,
            dtype=torch.float32,
            device=args.device,
            use_smolgen=args.use_smolgen,
            use_learnable_attn_scale=True,
            use_auxk=True,
            k_aux=args.k_aux,
            aux_coefficient=args.aux_coefficient,
            dead_threshold=args.dead_threshold,
        ),
        model=LanguageModelConfig(
            model_name=args.model_name,
            device=args.device,
            dtype="torch.float32",
            backend="evo2",
            prepend_bos=False,
            max_length=args.context_size,
        ),
        initializer=InitializerConfig(
            bias_init_method="geometric_median",
            init_encoder_with_decoder_transpose=False,
            decoder_uniform_bound=1 / math.sqrt(d_model),
            encoder_uniform_bound=1 / math.sqrt(d_model * args.exp_factor),
            state="training",
            init_search=args.init_search,
            initialize_W_D_with_active_subspace=args.initialize_W_D_with_active_subspace,
            d_active_subspace=d_qk_head,
            model_layer=args.layer,
            initialize_lorsa_with_mhsa=args.initialize_lorsa_with_mhsa,
            initialize_lorsa_smolgen_from_encoder=args.initialize_lorsa_smolgen_from_encoder,
            initialize_lorsa_attn_scale_from_encoder=args.initialize_lorsa_attn_scale_from_encoder,
        ),
        trainer=TrainerConfig(
            lr=args.lr,
            initial_k=args.k,
            k_warmup_steps=0.8,
            k_schedule_type="linear",
            amp_dtype=torch.float32,
            use_batch_norm_mse=False,
            total_training_tokens=args.total_training_tokens,
            log_frequency=10,
            eval_frequency=1_000_000,
            n_checkpoints=0,
            check_point_save_mode="linear",
            exp_result_path=str(result_dir),
        ),
        wandb=(
            WandbConfig(
                wandb_project="evo2_lorsa",
                wandb_entity="fnlp-mechinterp",
                exp_name=run_name,
            )
            if args.log_to_wandb
            else None
        ),
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryActivationsSource(
                    name="dna",
                    path=args.activations_path,
                    device=args.device,
                    dtype=torch.float32,
                    prefetch=args.prefetch,
                )
            ],
            target=ActivationFactoryTarget.ACTIVATIONS_2D,
            hook_points=[hook_point_in, hook_point_out],
            batch_size=args.batch_size,
            buffer_size=None,
            ignore_token_ids=[],
        ),
        sae_name=run_name,
        sae_series=f"{args.model_name}-lorsa",
        model_parallel_size=args.tp,
        data_parallel_size=args.dp,
        device_type="cpu" if args.device == "cpu" else "cuda",
    )
    train_lorsa(settings)
