import torch

from lm_saes import (
    ActivationFactoryTarget,
    DatasetConfig,
    GenerateActivationsSettings,
    LanguageModelConfig,
    MongoDBConfig,
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
            dataset_name_or_path="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs/exp/diff_datas/analyze_datas/OpenR1-Math-220k",
            is_dataset_on_disk=True,
        ),
        dataset_name="OpenR1-Math-220k-filter",
        mongo=MongoDBConfig(),
        hook_points=["blocks.3.ln1.hook_normalized"],
        output_dir="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs/exp/data/SlimPajama-3B-activations-pythia-2d",
        total_tokens=1_000,
        context_size=1024,
        n_samples_per_chunk=None,
        model_batch_size=32,
        target=ActivationFactoryTarget.ACTIVATIONS_2D,
    )
    generate_activations(settings)  # runner 都不会污染全局环境，所以可以安全地在一个脚本里运行多个 runner。
    torch.distributed.destroy_process_group()  # 若没有这行无法正确退出 NCCL。
