import torch

from lm_saes import (
    ActivationFactoryTarget,
    DatasetConfig,
    GenerateActivationsSettings,
    LanguageModelConfig,
    generate_activations,
)

if __name__ == "__main__":
    layers = [0, 5]
    
    settings = GenerateActivationsSettings(
        model=LanguageModelConfig(
            model_name="gpt2",
            device="cuda",
            model_from_pretrained_path="path/to/gpt2",  # download https://huggingface.co/openai-community/gpt2
            dtype="torch.float32",
        ),
        model_name="pythia-160m",
        dataset=DatasetConfig(
            dataset_name_or_path="path/to/SlimPajama-3B",  # download https://huggingface.co/datasets/Hzfinfdu/SlimPajama-3B to your local disk.
            is_dataset_on_disk=True,
        ),
        dataset_name="SlimPajama-3B",
        hook_points=[f"blocks.{layer}.hook_resid_post" for layer in layers],
        output_dir="activations",
        total_tokens=801_000_000,
        context_size=1024,
        n_samples_per_chunk=1,
        model_batch_size=32,
        target=ActivationFactoryTarget.BATCHED_ACTIVATIONS_1D,
        batch_size=2048 * 16,
        buffer_size=2048 * 200,
    )
    generate_activations(settings)
    torch.distributed.destroy_process_group()
