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
            dataset_name_or_path="Skylion007/openwebtext",
            is_dataset_on_disk=True,
        ),
        dataset_name="openwebtext",
        hook_points=[f"blocks.{layer}.ln1.hook_normalized" for layer in range(12)],
        output_dir="activations",
        total_tokens=800_000_000,
        context_size=1024,
        n_samples_per_chunk=1,
        model_batch_size=32,
        target=ActivationFactoryTarget.BATCHED_ACTIVATIONS_1D,
        batch_size=2048 * 16,
        buffer_size=2048 * 200,
    )
    generate_activations(settings)
    torch.distributed.destroy_process_group()
