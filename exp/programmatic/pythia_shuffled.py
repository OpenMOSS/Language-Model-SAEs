import torch

from lm_saes import (
    ActivationFactoryTarget,
    DatasetConfig,
    GenerateActivationsSettings,
    LanguageModelConfig,
    generate_activations,
)

if __name__ == "__main__":
    settings = GenerateActivationsSettings(
        model=LanguageModelConfig(
            model_name="EleutherAI/pythia-160m",
            device="cuda",
            dtype="torch.float32",
        ),
        model_name="pythia-160m",
        dataset=DatasetConfig(
            dataset_name_or_path="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/data/SlimPajama-3B",
            is_dataset_on_disk=True,
        ),
        dataset_name="SlimPajama-3B",
        hook_points=[f"blocks.{layer}.ln1.hook_normalized" for layer in range(12)],
        output_dir="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/data/SlimPajama-3B-activations-pythia",
        total_tokens=800_000_000,
        context_size=1024,
        n_samples_per_chunk=1,
        model_batch_size=32,
        target=ActivationFactoryTarget.BATCHED_ACTIVATIONS_1D,
        batch_size=2048 * 16,
        buffer_size=2048 * 200,
    )
    generate_activations(settings)  # runner 都不会污染全局环境，所以可以安全地在一个脚本里运行多个 runner。
    torch.distributed.destroy_process_group()  # 若没有这行无法正确退出 NCCL。
