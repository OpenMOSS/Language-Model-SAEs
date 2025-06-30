import torch

from lm_saes import (
    ActivationFactoryTarget,
    BufferShuffleConfig,
    DatasetConfig,
    GenerateActivationsSettings,
    LanguageModelConfig,
    generate_activations,
)

if __name__ == "__main__":
    settings = GenerateActivationsSettings(
        model=LanguageModelConfig(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            model_from_pretrained_path="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/models/Llama-3.1-8B-Instruct",
            device="cuda",
            dtype="torch.bfloat16",
        ),
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        dataset=DatasetConfig(
            dataset_name_or_path="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/data/SlimPajama-3B",
            is_dataset_on_disk=True,
        ),
        dataset_name="SlimPajama-3B",
        hook_points=[f"blocks.{l}.hook_resid_post" for l in [7, 15, 23]],
        output_dir="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs/exp/gen_act/i-1d-l15",
        total_tokens=860_000_000,
        context_size=1024,
        n_samples_per_chunk=None,
        model_batch_size=4,
        num_workers=36,
        target=ActivationFactoryTarget.BATCHED_ACTIVATIONS_1D,
        batch_size=2048 * 16,
        buffer_size=2048 * 32,
        buffer_shuffle=BufferShuffleConfig(perm_seed=42, generator_device="cuda"),
    )
    generate_activations(settings)  # runner 都不会污染全局环境，所以可以安全地在一个脚本里运行多个 runner。
    torch.distributed.destroy_process_group()  # 若没有这行无法正确退出 NCCL。
