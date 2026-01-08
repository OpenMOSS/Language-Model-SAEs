# Language Model SAEs

Welcome to the documentation for **Language Model SAEs** - a library for training and analyzing Sparse Autoencoders (SAEs) on language models.

## Overview

Sparse Autoencoders (SAEs) are neural network models used to extract interpretable features from language models. They help address the superposition problem in neural networks by learning sparse, interpretable representations of activations.

This library provides:

- **Scalability**: Our framework is fully distributed with arbitrary combinations of data, model, and head parallelism for both training and analysis. Enjoy training SAEs with millions of features!
- **Flexibility**: We support a wide range of SAE variants, including vanilla SAEs, Lorsa (Low-rank Sparse Attention), CLT (Cross-layer Transcoder), MoLT (Mixture of Linear Transforms), CrossCoder, and more. Each variant can be combined with different activation functions (e.g., ReLU, JumpReLU, TopK, BatchTopK) and sparsity penalties (e.g., L1, Tanh).
- **Easy to Use**: We provide high-level `runners` APIs to quickly launch experiments with simple configurations. Check our [examples](https://github.com/OpenMOSS/Language-Model-SAEs/tree/main/examples) for verified hyperparameters.
- **Visualization**: We provide a unified web interface to visualize learned SAE variants and their features.

## Quick Start

### Installation

=== "Astral uv"

    We strongly recommend users to use [uv](https://docs.astral.sh/uv/) for dependency management. uv is a modern drop-in replacement of poetry or pdm, with a lightning fast dependency resolution and package installation. See their [instructions](https://docs.astral.sh/uv/getting-started/) on how to initialize a Python project with uv.

    To add our library as a project dependency, run:

    ```bash
    uv add lm-saes
    ```

    We also support [Ascend NPU](https://github.com/Ascend/pytorch) as an accelerator backend. To add our library as a project dependency with NPU dependency constraints, run:

    ```bash
    uv add lm-saes[npu]
    ```

=== "Pip"

    Of course, you can also directly use [pip](https://pypi.org/project/pip/) to install our library. To install our library with pip, run:

    ```bash
    pip install lm-saes
    ```

    Note that since we use a forked version of [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), so it'll be better to install the package in a seperate environment created by [conda](https://github.com/conda-forge/miniforge) or [virtualenv](https://virtualenv.pypa.io/en/latest/) to avoid conflicts.

    We also support [Ascend NPU](https://github.com/Ascend/pytorch) as an accelerator backend. To install our library with NPU dependency constraints, run:

    ```bash
    pip install lm-saes[npu]
    ```

### Load a trained Sparse Autoencoder from HuggingFace

WIP

### Training a Sparse Autoencoder

To train a simple Sparse Autoencoder on `blocks.5.hook_resid_post` of a Pythia-160M model with $768*8$ features, you can use the following:

```python
settings = TrainSAESettings(
    sae=SAEConfig(
        hook_point_in="blocks.6.hook_resid_post",
        hook_point_out="blocks.6.hook_resid_post",
        d_model=768,
        expansion_factor=8,
        act_fn="topk",
        top_k=50,
        dtype=torch.float32,
        device="cuda",
    ),
    initializer=InitializerConfig(
        grid_search_init_norm=True,
    ),
    trainer=TrainerConfig(
        amp_dtype=torch.float32,
        lr=1e-4,
        initial_k=50,
        k_warmup_steps=0.1,
        k_schedule_type="linear",
        total_training_tokens=800_000_000,
        log_frequency=1000,
        eval_frequency=1000000,
        n_checkpoints=0,
        check_point_save_mode="linear",
        exp_result_path="results",
    ),
    model=LanguageModelConfig(
        model_name="EleutherAI/pythia-160m",
        device="cuda",
        dtype="torch.float16",
    ),
    model_name="pythia-160m",
    datasets={
        "SlimPajama-3B": DatasetConfig(
            dataset_name_or_path="Hzfinfdu/SlimPajama-3B",
        )
    },
    wandb=WandbConfig(
        wandb_project="lm-saes",
        exp_name="pythia-160m-sae",
    ),
    activation_factory=ActivationFactoryConfig(
        sources=[
            ActivationFactoryDatasetSource(
                name="SlimPajama-3B",
            )
        ],
        target=ActivationFactoryTarget.ACTIVATIONS_1D,
        hook_points=["blocks.6.hook_resid_post"],
        batch_size=4096,
        buffer_size=4096 * 4,
        buffer_shuffle=BufferShuffleConfig(
            perm_seed=42,
            generator_device="cuda",
        ),
    ),
    sae_name="pythia-160m-sae",
    sae_series="pythia-sae",
)
train_sae(settings)
```

### Analyze a trained Sparse Autoencoder

WIP

### Convert trained Sparse Autoencoder to SAELens format

WIP

## Citation

If you find this library useful in your research, please cite:

```
@misc{Ge2024OpenMossSAEs,
    title  = {OpenMoss Language Model Sparse Autoencoders},
    author = {Xuyang Ge and Wentao Shu and Junxuan Wang and Guancheng Zhou and Jiaxing Wu and Fukang Zhu and Lingjie Chen and Zhengfu He},
    url    = {https://github.com/OpenMOSS/Language-Model-SAEs},
    year   = {2024}
}
```
