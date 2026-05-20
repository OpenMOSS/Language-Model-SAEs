from __future__ import annotations

import argparse

import torch

from llamascopium import (
    ActivationFactoryTarget,
    DatasetConfig,
    GenerateActivationsSettings,
    LanguageModelConfig,
    generate_activations,
)

from llamascopium.utils.evo2_hooks import (
    default_activation_dir,
    gen_evo2_hook_points_for_layer,
    get_evo2_arch,
    validate_layer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Evo2 LORSA activations (2D).")
    parser.add_argument("--model-name", type=str, default="evo2_7b")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="dna")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--total-tokens", type=int, default=1_000_000)
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument("--model-batch-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="torch.float32")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    arch = get_evo2_arch(args.model_name)
    validate_layer(args.layer, int(arch["num_layers"]))
    hook_points, layer_kind = gen_evo2_hook_points_for_layer(args.layer, args.model_name)
    print(f"[INFO] layer={args.layer} kind={layer_kind} hook_points={hook_points}")
    output_dir = args.output_dir or str(default_activation_dir(f"{layer_kind}_2d"))

    settings = GenerateActivationsSettings(
        model=LanguageModelConfig(
            model_name=args.model_name,
            device=args.device,
            dtype=args.dtype,
            backend="evo2",
            prepend_bos=False,
            max_length=args.context_size,
        ),
        model_name=args.model_name,
        dataset=DatasetConfig(
            dataset_name_or_path=args.dataset_path,
            is_dataset_on_disk=True,
        ),
        dataset_name=args.dataset_name,
        hook_points=hook_points,
        output_dir=output_dir,
        total_tokens=args.total_tokens,
        context_size=args.context_size,
        n_samples_per_chunk=None,
        model_batch_size=args.model_batch_size,
        target=ActivationFactoryTarget.ACTIVATIONS_2D,
        batch_size=args.batch_size,
        device_type="cpu" if args.device == "cpu" else "cuda",
    )
    generate_activations(settings)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
