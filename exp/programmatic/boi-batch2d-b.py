import argparse
import multiprocessing

import torch

from lm_saes import (
    ActivationFactoryTarget,
    DatasetConfig,
    GenerateActivationsSettings,
    LanguageModelConfig,
    generate_activations,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Activations")
    parser.add_argument("--start-shard", type=int, required=True, help="Start shard index")
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn")

    settings = GenerateActivationsSettings(
        model=LanguageModelConfig(
            model_name="meta-llama/Llama-3.1-8B",
            model_from_pretrained_path="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs/exp/diff_models/Llama-3.1-8B",
            device="cuda",
            dtype="torch.bfloat16",
        ),
        model_name="meta-llama/Llama-3.1-8B",
        dataset=DatasetConfig(
            dataset_name_or_path="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/data/SlimPajama-3B",
            is_dataset_on_disk=True,
        ),
        dataset_name="SlimPajama-3B",
        hook_points=["blocks.15.hook_resid_post", "blocks.7.hook_resid_post", "blocks.23.hook_resid_post"],
        output_dir="/inspire/hdd/global_user/hezhengfu-240208120186/jiaxing_activations/pretraindata-b-2d-801M-ctx8192",
        total_tokens=801_000_000,
        n_shards=16,
        start_shard=args.start_shard,
        context_size=8192,
        n_samples_per_chunk=None,
        # model_batch_size=4,
        num_workers=36,
        target=ActivationFactoryTarget.ACTIVATIONS_2D,
        # batch_size=2048 * 16,
        buffer_size=0,
    )
    generate_activations(settings)  # runner 都不会污染全局环境，所以可以安全地在一个脚本里运行多个 runner。
    torch.distributed.destroy_process_group()  # 若没有这行无法正确退出 NCCL。
