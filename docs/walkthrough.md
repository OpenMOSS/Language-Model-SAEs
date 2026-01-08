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

It returns a streaming iterator. Each item is a dictionary mainly containing:

| Key | Description |
|-----|-------------|
| `"blocks.6.hook_resid_post"` | Activation tensor with shape `(batch_size, d_model)` — in this config, `(4096, 768)` |
| `"tokens"` | The corresponding token IDs |

With target set to `ActivationFactoryTarget.ACTIVATIONS_1D`, the produced activations will have no sequence dimension. They are shuffled across both samples and context positions to ensure the SAE trains on randomly sampled activations from any position in any sample.

### Create and Initialize SAE

We've successfully prepared the data we need. It's time to turn to the SAE itself! But before training, we should first define the SAE architecture and initialize it. Create an `SAEConfig` to define the SAE architecture:

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

Here're some brief explanations of the config we set:

| Parameter | Description |
|-----------|-------------|
| `hook_point_in` / `hook_point_out` | When identical, this defines an SAE; when different, it becomes a transcoder |
| `d_model` | Must match the model's hidden size (768 for Pythia-160m) |
| `expansion_factor` | Multiplier for the latent dimension. Here, `d_sae = 768 × 8 = 6144` |
| `act_fn` | Activation function. Modern SAEs often use `"jumprelu"` or `"batchtopk"`, but we use `"relu"` for simplicity |

More options of `SAEConfig` are introduced in the [reference]().

With only SAEConfig defined, the created SAE will have nothing but empty tensors as parameters. We need to fill the empty parameters with proper initialization, which is often proved crucial for final SAE performance. The `Initializer` class handles parameter initialization. The `grid_search_init_norm` option (recommended) searches for the optimal encoder/decoder parameter scale to minimize initial MSE loss on the activation distribution.

```python
from lm_saes import Initializer, InitializerConfig

initializer = Initializer(InitializerConfig(grid_search_init_norm=True))
sae = initializer.initialize_sae_from_config(
    sae_cfg,
    activation_stream=activations_stream
)
```

### Train SAE

Finally, we can start training! A `Trainer` instance is responsible for holding optimizer & scheduler states.

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

More options on the optimizer/scheduler and other hyperparameters are available. See the [reference]() for more detail.

Just run `trainer.fit` and pass in the initialized SAE and the activation stream, and keep eyes on the console log to see whether the training goes well!

```python
sae.cfg.save_hyperparameters("results") # Save hyperparameter before training

trainer.fit(
    sae=sae, 
    activation_stream=activations_stream,
)

sae.save_pretrained(save_path="results") # Save the trained weight after training
```

!!! note "Consistent Save Path"
    The path in `sae.cfg.save_hyperparameters` and `sae.save_pretrained` should be the same as specified in `exp_result_path` in `TrainerConfig`. Otherwise, the trained SAE may not be able to be correctly loaded.

---

## Using the High-Level Runner API

For a more streamlined experience, `Language-Model-SAEs` also provides a high-level `train_sae` function that bundles all configuration into a single `TrainSAESettings` object. You can programmatically create the settings object and call the `train_sae`, or you can also use a configuration-file based settings and run it with our `lm-saes` CLI:

=== "Runner"

    Create the `TrainSAESettings` in Python and call `train_sae` with it. 

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

=== "CLI"

    CLI-based workflow requires a configuration file containing the settings consistent with `TrainSAESettings`. Common configuration file type like TOML, JSON and YAML are supported.

    Create a TOML configuration file (e.g., `train_config.toml`) with the following content:

    ```toml
    sae_name = "pythia-160m-sae"
    sae_series = "pythia-sae"
    model_name = "pythia-160m"
    device_type = "cuda"

    [sae]
    sae_type = "sae"
    hook_point_in = "blocks.6.hook_resid_post"
    hook_point_out = "blocks.6.hook_resid_post"
    d_model = 768
    expansion_factor = 8
    act_fn = "relu"
    dtype = "torch.float32"
    device = "cuda"

    [initializer]
    grid_search_init_norm = true

    [trainer]
    amp_dtype = "torch.float32"
    lr = 0.0001
    total_training_tokens = 800_000_000
    log_frequency = 1000
    exp_result_path = "results"

    [model]
    model_name = "EleutherAI/pythia-160m"
    device = "cuda"
    dtype = "torch.float16"

    [datasets."SlimPajama-3B"]
    dataset_name_or_path = "Hzfinfdu/SlimPajama-3B"

    [activation_factory]
    target = "activations-1d"
    hook_points = ["blocks.6.hook_resid_post"]
    batch_size = 4096
    buffer_size = 16384

    [[activation_factory.sources]]
    type = "dataset"
    name = "SlimPajama-3B"

    [activation_factory.buffer_shuffle]
    perm_seed = 42
    generator_device = "cuda"
    ```

    Then run the training with:

    ```bash
    lm-saes train train_config.toml
    ```

=== "Full Script"

    We also recommend users to directly use the low level semantics for launching training, which allows more granular control and easier customizing:

    ```python
    import datasets
    import torch
    from lm_saes import (
        ActivationFactory,
        ActivationFactoryConfig,
        ActivationFactoryDatasetSource,
        ActivationFactoryTarget,
        BufferShuffleConfig,
        Initializer,
        InitializerConfig,
        LanguageModelConfig,
        SAEConfig,
        Trainer,
        TrainerConfig,
        TransformerLensLanguageModel,
    )

    # 1. Load Model & Dataset
    model = TransformerLensLanguageModel(
        LanguageModelConfig(
            model_name="EleutherAI/pythia-160m",
            device="cuda",
            dtype="torch.float16",
        )
    )

    dataset = datasets.load_dataset(
        "Hzfinfdu/SlimPajama-3B",
        split="train",
    )

    # 2. Generate Activations
    activation_factory = ActivationFactory(
        ActivationFactoryConfig(
            sources=[ActivationFactoryDatasetSource(name="SlimPajama-3B")],
            target=ActivationFactoryTarget.ACTIVATIONS_1D,
            hook_points=["blocks.6.hook_resid_post"],
            batch_size=4096,
            buffer_size=4096 * 4,
            buffer_shuffle=BufferShuffleConfig(
                perm_seed=42,
                generator_device="cuda",
            ),
        )
    )

    activations_stream = activation_factory.process(
        model=model,
        model_name="pythia-160m",
        datasets={"SlimPajama-3B": dataset},
    )

    # 3. Create and Initialize SAE
    sae_cfg = SAEConfig(
        hook_point_in="blocks.6.hook_resid_post",
        hook_point_out="blocks.6.hook_resid_post",
        d_model=768,
        expansion_factor=8,
        act_fn="relu",
        dtype=torch.float32,
        device="cuda",
    )

    initializer = Initializer(InitializerConfig(grid_search_init_norm=True))
    sae = initializer.initialize_sae_from_config(
        sae_cfg,
        activation_stream=activations_stream,
    )

    # 4. Train SAE
    trainer = Trainer(
        TrainerConfig(
            amp_dtype=torch.float32,
            lr=1e-4,
            total_training_tokens=800_000_000,
            log_frequency=1000,
            exp_result_path="results",
        )
    )

    sae.cfg.save_hyperparameters("results")

    trainer.fit(
        sae=sae,
        activation_stream=activations_stream,
    )

    sae.save_pretrained(save_path="results")
    ```

## Logging to W&B

Aside from the console logger, we support logging to Weights & Biases for tracking loss and metric changes throughout the training. Training metrics including [explained variance](https://www.lesswrong.com/posts/E3nsbq2tiBv6GLqjB/x-explains-z-of-the-variance-in-y) and $L_0$ norm will be automatically recorded. Below is a screenshot of the W&B logging:

![Screenshot of W&B logging in training our LlamaScope 2 Beta PLTs](assets/images/wandb.png)
/// caption
Screenshot of W&B logging in training our LlamaScope 2 Beta PLTs.
///

To enable W&B logging, add the `wandb` configuration to your training setup:

=== "Runner"

    ```python
    from lm_saes import WandbConfig

    settings = TrainSAESettings(
        # ... other settings ...
        wandb=WandbConfig(
            wandb_project="my-sae-training",
            exp_name="pythia-160m-sae",
        ),
    )

    train_sae(settings)
    ```

=== "CLI"

    ```toml
    # ... other configurations ... 

    [wandb]
    wandb_project = "my-sae-training"
    exp_name = "pythia-160m-sae"
    ```

=== "Full Script"

    ```python
    import wandb

    # ... other training logics ...

    # Create a W&B instance
    wandb_logger = wandb.init(
        project="my-sae-training",
        name="pythia-160m-sae",
    )

    # Pass it to `trainer.fit`
    trainer.fit(
        sae=sae,
        activation_stream=activations_stream,
        wandb_logger=wandb_logger,
    )

    wandb_logger.finish()
    ```

## Checkpoints and Continue Training

WIP

## Activation Functions

Activation functions are the direct architectural design to enforce a sparse feature activations in SAE and its variants. 

### ReLU

ReLU is the most classical activation, proposed in initial works (*[Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600)* and *[Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features)*) using SAEs to disentangle superposition. Though its performance is found inferior to other activation functions in term of explained variance and $L_0$ norms, it might be a good starting point to understand how SAE works due to its simplicity.

To use ReLU activation function, just set `#!python act_fn = "relu"` in `SAEConfig`.

### JumpReLU

JumpReLU is a state-of-the-art activation function proposed in *[Jumping Ahead: Improving Reconstruction
Fidelity with JumpReLU Sparse Autoencoders](https://arxiv.org/abs/2407.14435)*, and adopted by both Google DeepMind [GemmaScope](https://arxiv.org/abs/2408.05147) and [GemmaScope 2](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/Gemma_Scope_2_Technical_Paper.pdf), and Anthropic [Cross Layer Transcoder](https://transformer-circuits.pub/2025/attribution-graphs/methods.html#building-architecture).

JumpReLU modifies the ReLU activation function, allowing only elements that passing the corresponding element-wise thresholds to activate. Consider an input element $x$ and a log-threshold $t$, it computes:

$$
\operatorname{JumpReLU}(x;t) = H(x-e^t)x
$$

where $H(\cdot)$ is the [Heaviside step function](https://en.wikipedia.org/wiki/Heaviside_step_function)[^1]. For comparison, ReLU can be written as $\operatorname{ReLU}(x) = H(x)x$.

[^1]: 
    The Heaviside step function $H(x)$ is defined as:

    $$
    H(x) =
    \begin{cases}
        1 & \text{if } x > 0 \\
        0 & \text{otherwise}
    \end{cases}
    $$ 

Since the Heaviside step function cannot be differentiated, JumpReLU uses a straight-through estimator of the gradient through the discontinuity of the nonlinearity. See [Anthropic Circuit Update - January 2025](https://transformer-circuits.pub/2025/january-update/index.html#DL) to learn how (log) JumpReLU thresholds are optimized.

To use the JumpReLU activation function, set `#!python act_fn = "jumprelu"` in `SAEConfig`. You may also adjust the `jumprelu_threshold_window` to control the sensitivity of how JumpReLU thresholds update.

!!! note "Dedicated Learning Rate for Log JumpReLU Thresholds"
    In our crosscoder training experiments in [Evolution of Concepts in Language Model Pre-Training](https://arxiv.org/abs/2509.17196), we find it better to apply a smaller learning rate (0.1x) to the log JumpReLU thresholds. Though this setting hardly affects the final performance on reconstruction and sparsity, it makes the training loss far more smooth. The mean feature activation becomes lower after the change.

### TopK

TopK is an activation function proposed in [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093). It keeps only the $k$ largest elements, zeroing out the rest, thus directly enforcing strict sparsity on feature activation. To this end, it removes the need for additional sparsity penalties (which are typically basic requirements for ReLU & JumpReLU activations), and enables direct control over the sparsity quantitative ($L_0$) of feature activation.

To use TopK activation function, set `#!python act_fn = "topk"` in `SAEConfig`, and set the `top_k` value to control the final sparsity of feature activation. We also provide some options in `TrainerConfig` to enable scheduling on $k$ value during training:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `initial_k` | The starting $k$ value for scheduling. Must be greater than or equal to `top_k` set in `SAEConfig`. | `None` |
| `k_warmup_steps` | Steps (int) or fraction of total steps (float) for $k$ to decay from `initial_k` to `top_k`. | `0.1` |
| `k_cold_booting_steps` | Steps (int) or fraction of total steps (float) to keep $k$ at `initial_k` before starting the decay. | `0` |
| `k_schedule_type` | Scheduling strategy: `"linear"` or `"exponential"`. | `"linear"` |
| `k_exponential_factor` | Controls the curvature of the exponential decay. | `3.0` |

!!! info "Use BatchTopK Activation"

    TopK activation enforces unnecessary fixed allocation of active latents. For strict architectural sparsity control, we recommend using [BatchTopK](#batchtopk) for better performance.

### BatchTopK

BatchTopK is a state-of-the-art activation function proposed in [BatchTopK Sparse Autoencoders](https://arxiv.org/abs/2412.06410). It follows the idea of TopK to directly enforce sparsity, but replaces the sample-level TopK operation with a batch-level BatchTopK operation. For pre-feature-activations of shape `(batch_size, d_sae)`, it selects the top `batch_size * top_k` activations across the entire batch of `batch_size` samples. This allows for more flexible allocation of active latents.

To use TopK activation function, set `#!python act_fn = "batchtopk"` in `SAEConfig`, and set the `top_k` value to control the final sparsity of feature activation. We also provide some options in `TrainerConfig` to enable scheduling on $k$ value during training:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `initial_k` | The starting $k$ value for scheduling. Must be greater than or equal to `top_k` set in `SAEConfig`. | `None` |
| `k_warmup_steps` | Steps (int) or fraction of total steps (float) for $k$ to decay from `initial_k` to `top_k`. | `0.1` |
| `k_cold_booting_steps` | Steps (int) or fraction of total steps (float) to keep $k$ at `initial_k` before starting the decay. | `0` |
| `k_schedule_type` | Scheduling strategy: `"linear"` or `"exponential"`. | `"linear"` |
| `k_exponential_factor` | Controls the curvature of the exponential decay. | `3.0` |

#### Convert BatchTopK to JumpReLU

BatchTopK introduces a dependency between the activations for the samples in a batch. To eliminate the effect, it's better to estimate a threshold $\theta$ as average minimum positive activation values, and convert the activation function to JumpReLU with this threshold.

## Sparsity Penalties

WIP