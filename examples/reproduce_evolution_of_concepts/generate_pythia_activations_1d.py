import argparse
import os

import torch

from lm_saes import (
    ActivationFactoryTarget,
    BufferShuffleConfig,
    DatasetConfig,
    GenerateActivationsSettings,
    LanguageModelConfig,
    generate_activations,
)

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

steps = [
    0,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1000,
    7000,
    14000,
    20000,
    27000,
    34000,
    40000,
    47000,
    54000,
    60000,
    67000,
    74000,
    80000,
    87000,
    94000,
    100000,
    107000,
    114000,
    120000,
    127000,
    134000,
    143000,
    2000,
    3000,
    4000,
    5000,
    6000,
    8000,
    9000,
    10000,
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="160m")
    parser.add_argument("--dtype", type=str, default="torch.float16")
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int, default=len(steps))
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--batch-size-scale", type=float, default=1.0)
    args = parser.parse_args()

    dtype_suffix = (
        "fp32"
        if args.dtype == "torch.float32"
        else "fp16"
        if args.dtype == "torch.float16"
        else "bf16"
        if args.dtype == "torch.bfloat16"
        else None
    )
    if dtype_suffix is None:
        raise ValueError(f"Invalid dtype: {args.dtype}")

    layers = [int(x.strip()) for x in args.layer.split(",")]

    # timer.enable()
    for step in steps[args.start_step : args.end_step]:
        print(f"Generating activations for pythia-{args.size} at step {step} for layers {layers}")
        settings = GenerateActivationsSettings(
            model=LanguageModelConfig(
                model_name=f"EleutherAI/pythia-{args.size}",
                device="cuda",
                dtype=args.dtype,
                d_model=d_model_map[args.size],
                model_from_pretrained_path=os.path.expanduser(f"~/models/pythia-{args.size}-all/step{step}"),
            ),
            model_name=f"pythia-{args.size}",
            dataset=DatasetConfig(
                dataset_name_or_path=os.path.expanduser("~/data/SlimPajama-3B"),
                is_dataset_on_disk=True,
            ),
            dataset_name="SlimPajama-3B",
            hook_points=[f"blocks.{layer}.hook_resid_post" for layer in layers],
            output_dir=os.path.expanduser(
                f"~/activations/SlimPajama-3B-activations-pythia-{args.size}-1d-all-{dtype_suffix}/step{step}"
            ),
            total_tokens=800_000_000,
            context_size=2048,
            n_samples_per_chunk=None,
            model_batch_size=int(32 * args.batch_size_scale),
            num_workers=None,
            target=ActivationFactoryTarget.ACTIVATIONS_1D,
            batch_size=int(2048 * 64 * args.batch_size_scale),
            buffer_size=int(2048 * 200 * args.batch_size_scale),
            buffer_shuffle=BufferShuffleConfig(
                perm_seed=42,
                generator_device="cuda",
            ),
            device_type="cuda",
        )
        generate_activations(settings)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
