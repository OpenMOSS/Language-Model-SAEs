# Walkthrough

`Language-Model-SAEs` provides a general way to train, analyze and visualize Sparse Autoencoders and their variants. To help you get started quickly, we've included [example scripts]() that guide you through each stage of working with SAEs. This guide begins with a foundational example and progressively introduces the core features and capabilities of the library.

## Training Basic Sparse Autoencoders

A [Sparse Autoencoder]() is trained to reconstruct model activations at specific position. We depend on [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) to take activations out of model forward pass, specified by hook points. `Language-Model-SAEs` provides complete abstraction on the necessary components to train Sparse Autoencoders at ease.

### Load Model & Dataset

Generation of our training data, the model activations, requires the presence of both the language model and the dataset. First, we can load a pretrained language model by:

```python
from lm_saes import LanguageModelConfig, TransformerLensLanguageModel

model = TransformerLensLanguageModel(
    LanguageModelConfig(
        model_name="EleutherAI/pythia-160m",
        device="cuda",
        dtype="torch.float16",
    )
)
```

where `TransformerLensLanguageModel` is a simple wrapper around the TransformerLens `HookedTransformer`, enhanced with:

1. **Unified interface** for extracting activations (compatible with native HuggingFace transformers) and tracing token positions from original texts.
2. **Distributed training support** with simple data parallelism integrated.

See the [section]() below to find how to use HuggingFace transformers directly for generating activation.

!!! tip "Use Half Precision"
    Activation generation constitutes the majority of training time. We strongly recommend using half precision (`float16` or `bfloat16`) to accelerate the forward pass and reduce GPU memory usage. Here we use FP16 since Pythia models are trained in FP16.

Next, we load some text corpus from HuggingFace. Different pretraining text corpus often does not have so much effect on SAE training. Here we load [`Hzfinfdu/SlimPajama-3B`](https://huggingface.co/datasets/Hzfinfdu/SlimPajama-3B), a 3B-token subset of the [627B SlimPajama dataset](https://huggingface.co/datasets/cerebras/SlimPajama-627B), which is typically sufficient for basic SAE training.

```python
import datasets

dataset = datasets.load_dataset(
    "Hzfinfdu/SlimPajama-3B", 
    split="train",
)
```

### Generate Activations

Model activations often require some further transformation to ensure correct and efficient SAE training. We provide `ActivationFactory` as the core abstraction for producing activation streams. It provides a comprehensive interface to generate activations from model forward passes, filter unnecessary tokens, and reshape, re-batch, and shuffle activations.

We can create an `ActivationFactory` as follow:

```python
from lm_saes import (
    ActivationFactory,
    ActivationFactoryConfig,
    ActivationFactoryDatasetSource,
    ActivationFactoryTarget,
    BufferShuffleConfig,
)

activation_factory = ActivationFactory(
    ActivationFactoryConfig(
        sources=[ActivationFactoryDatasetSource(name="SlimPajama-3B")],
        target=ActivationFactoryTarget.ACTIVATIONS_1D,
        hook_points=["blocks.6.hook_resid_post"],
        batch_size=4096,
        buffer_size=4096 * 4, # Set to enable the online activation shuffling
        buffer_shuffle=BufferShuffleConfig(
            perm_seed=42,
            generator_device="cuda",
        ),
    )
)
```

Then, we call the `.process` method, passing in our loaded model and dataset, to start the stream processing of the activations.

```python
activations_stream = activation_factory.process(
    model=model,
    model_name="pythia-160m",
    datasets={"SlimPajama-3B": dataset},
)
```

The `ActivationFactory.process()` method returns a streaming iterator. Each item is a dictionary containing:

| Key | Description |
|-----|-------------|
| `"blocks.6.hook_resid_post"` | Activation tensor with shape `(batch_size, d_model)` — in this config, `(4096, 768)` |
| `"tokens"` | The corresponding token IDs |

With `ActivationFactoryTarget.ACTIVATIONS_1D`, activations have no sequence dimension. They are shuffled across both samples and context positions to ensure the SAE trains on randomly sampled activations from any position in any sample.



### Step 5: Configure the SAE

Define the SAE architecture using `SAEConfig`:

```python
import torch
from lm_saes import SAEConfig

sae_cfg = SAEConfig(
    hook_point_in="blocks.6.hook_resid_post",
    hook_point_out="blocks.6.hook_resid_post",
    d_model=768,
    expansion_factor=8,
    act_fn="relu",
    dtype=torch.float32,
    device="cuda",
)
```

| Parameter | Description |
|-----------|-------------|
| `hook_point_in` / `hook_point_out` | When identical, this defines an SAE; when different, it becomes a transcoder |
| `d_model` | Must match the model's hidden size (768 for Pythia-160m) |
| `expansion_factor` | Multiplier for the latent dimension. Here, `d_sae = 768 × 8 = 6144` |
| `act_fn` | Activation function. Modern SAEs often use `"jumprelu"` or `"batchtopk"`, but we use `"relu"` for simplicity |

### Step 6: Initialize the SAE

Proper initialization is crucial for SAE performance. The `Initializer` class handles parameter initialization with the `grid_search_init_norm` option (recommended), which searches for the optimal encoder/decoder parameter scale to minimize initial MSE loss on the activation distribution.

```python
from lm_saes import Initializer, InitializerConfig

initializer = Initializer(InitializerConfig(grid_search_init_norm=True))
sae = initializer.initialize_sae_from_config(
    sae_cfg,
    activation_stream=activations_stream
)
```

### Step 7: Configure the Trainer

The `Trainer` manages optimizer and scheduler states. Key configuration options:

```python
from lm_saes import Trainer, TrainerConfig

trainer = Trainer(
    TrainerConfig(
        amp_dtype=torch.float32,
        lr=1e-4,
        total_training_tokens=800_000_000,
        log_frequency=1000,
        exp_result_path="results",
    )
)
```

| Parameter | Description |
|-----------|-------------|
| `amp_dtype` | Mixed precision dtype. Also handles precision mismatches between SAE parameters and activations |
| `lr` | Learning rate |
| `total_training_tokens` | Total tokens for training. Training steps = total tokens / batch size |
| `log_frequency` | Logging interval (in steps) for console and W&B |
| `exp_result_path` | Directory for saving results and checkpoints |

### Step 8: Train and Save

Before training, save the hyperparameters. The path should match `exp_result_path` to ensure the trained SAE can be loaded correctly later.

```python
sae.cfg.save_hyperparameters("results")

trainer.fit(
    sae=sae, 
    activation_stream=activations_stream,
)

sae.save_pretrained(save_path="results")
```

---

## Using the High-Level API

For a more streamlined experience, `Language-Model-SAEs` also provides a high-level `train_sae` function that bundles all configuration into a single `TrainSAESettings` object:

```python
import torch
from lm_saes import (
    TrainSAESettings,
    train_sae,
    SAEConfig,
    InitializerConfig,
    TrainerConfig,
    LanguageModelConfig,
    DatasetConfig,
    ActivationFactoryConfig,
    ActivationFactoryDatasetSource,
    ActivationFactoryTarget,
    BufferShuffleConfig,
)

settings = TrainSAESettings(
    sae=SAEConfig(
        hook_point_in="blocks.6.hook_resid_post",
        hook_point_out="blocks.6.hook_resid_post",
        d_model=768,
        expansion_factor=8,
        act_fn="relu",
        dtype=torch.float32,
        device="cuda",
    ),
    initializer=InitializerConfig(
        grid_search_init_norm=True,
    ),
    trainer=TrainerConfig(
        amp_dtype=torch.float32,
        lr=1e-4,
        total_training_tokens=800_000_000,
        log_frequency=1000,
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
