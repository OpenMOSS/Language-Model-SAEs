import argparse
import os

import torch

from lm_saes import (
    ActivationFactoryTarget,
    DatasetConfig,
    GenerateActivationsSettings,
    LanguageModelConfig,
    generate_activations,
)

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="160m")
    parser.add_argument("--dtype", type=str, default="torch.float16")
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--model_batch_size", type=int, default=32)

    parser.add_argument("--activation_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    args = parse_args()

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

    settings = GenerateActivationsSettings(
        model=LanguageModelConfig(
            model_name=f"EleutherAI/pythia-{args.size}",
            device="cuda",
            dtype=args.dtype,
            d_model=d_model_map[args.size],
        ),
        model_name=f"pythia-{args.size}",
        dataset=DatasetConfig(dataset_name_or_path="Hzfinfdu/SlimPajama-3B"),
        dataset_name="SlimPajama-3B",
        hook_points=[f"blocks.{layer}.hook_resid_post" for layer in layers],
        output_dir=os.path.expanduser(args.activation_path),
        total_tokens=100_000_000,
        context_size=2048,
        model_batch_size=args.model_batch_size,
        target=ActivationFactoryTarget.ACTIVATIONS_2D,
        device_type="cuda",
    )
    generate_activations(settings)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
