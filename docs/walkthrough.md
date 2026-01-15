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

Aside from the console logger, we support logging to Weights & Biases for tracking loss and metric changes throughout the training. Training metrics including [explained variance](https://www.lesswrong.com/posts/E3nsbq2tiBv6GLqjB/x-explains-z-of-the-variance-in-y) and $L^0$ norm will be automatically recorded. Below is a screenshot of the W&B logging:

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

ReLU is the most classical activation, proposed in initial works (*[Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600)* and *[Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features)*) using SAEs to disentangle superposition. Though its performance is found inferior to other activation functions in term of explained variance and $L^0$ norms, it might be a good starting point to understand how SAE works due to its simplicity.

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

TopK is an activation function proposed in [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093). It keeps only the $k$ largest elements, zeroing out the rest, thus directly enforcing strict sparsity on feature activation. To this end, it removes the need for additional sparsity penalties (which are typically basic requirements for ReLU & JumpReLU activations), and enables direct control over the sparsity quantitative ($L^0$) of feature activation.

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

Activation functions like ReLU and JumpReLU do not strictly enforce sparsity on feature activations. It's the responsibility of the regularization functions to provide dynamics pushing feature activations sparse. `Language-Model-SAEs` supports the following sparsity penalties on feature activations:

### $L^p$-Norm

Sparsity referes to the number of active latents in feature activation. In principle, we may want to directly add $L^0$ norm to the loss term. However, $L^0$ norm is discontinuous and cannot be differentiated,

In practice, $L^1$ norm, as the _best convex approximation_ to $L^0$[^2], is widely used in SAE training for controlling sparsity without lossing convexity of the optimization. `Language-Model-SAEs` implements a more general $L^p$ norm as regularization, which is computed as:

$$L_s = \lambda \| f(x) \cdot \| W_\text{dec} \|_2 \|_p$$

where $f(x)$ is the feature activation, $\| W_\text{dec} \|_2$ is the decoder norm, $p$ is the $L^p$ power, and $\lambda$ is the coefficient for the sparsity loss term.

[^2]: See more discussion on $L^0$ approximate functions in [Comparing Measures of Sparsity](https://arxiv.org/abs/0811.4706) and [Why $\ell_1$ Is a Good Approximation to $\ell_0$: A Geometric Explanation](https://www.cs.utep.edu/vladik/2013/tr13-18.pdf).

To use the $L^p$ norm, set `#!python sparsity_loss_type="power"` in `TrainerConfig`. Other parameters include:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `l1_coefficient` | Coefficient $\lambda$ for the sparsity loss term. | `0.00008` |
| `l1_coefficient_warmup_steps` | Steps (int) or fraction of total steps (float) to warm up the sparsity coefficient from 0. | `0.1` |
| `p` | The power $p$ for $L^p$ norm. Set to $1$ for $L^1$ norm. | `1` |

### Tanh

One challenge with $L^1$ penalty is _shrinkage_: in addition to encouraging sparsity, the penalty encourages activations to be smaller than they would be otherwise. This causes SAEs to recover a smaller fraction of the model loss than might be expected[^3].

The $\tanh$ penalty addresses shrinkage by applying a bounded function to feature activations. For marginal cases where a feature is on the edge of activating, it provides the same gradient towards zero as $L^1$, but for strongly-activating features it provides no penalty and hence no incentive to shrink the activation. The loss is computed as:

$$L_s = \lambda \sum_i \tanh(c \cdot f_i(x) \cdot \| W_{\text{dec},i} \|_2)$$

where $f_i(x)$ is the $i$-th feature activation, $\| W_{\text{dec},i} \|_2$ is the decoder norm for feature $i$, and $c$ is the stretch coefficient. Since $\tanh(x) \to 1$ as $x \to \infty$, this loss approximates counting the number of active features ($L^0$ norm).

While the $\tanh$ penalty was found to be a Pareto improvement in the $L^0$/MSE tradeoff, [Anthropic's experiments](https://transformer-circuits.pub/2024/feb-update/index.html#dict-learning-tanh) showed that features trained with tanh were much harder to interpret due to many more high-frequency features (some activating on over 10% of inputs). However, their [following up experiments](https://transformer-circuits.pub/2024/june-update/index.html#topk-gated-comparison) show that these high density features don’t seem to be pathological as previous thought.

[^3]: See [Fixing Feature Suppression in SAEs](https://www.lesswrong.com/posts/3JuSjTZyMzaSeTxKk/fixing-feature-suppression-in-saes-2) for more discussion.

To use the $\tanh$ penalty, set `#!python sparsity_loss_type="tanh"` in `TrainerConfig`. Other parameters include:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `l1_coefficient` | Coefficient $\lambda$ for the sparsity loss term. | `0.00008` |
| `l1_coefficient_warmup_steps` | Steps (int) or fraction of total steps (float) to warm up the sparsity coefficient from 0. | `0.1` |
| `tanh_stretch_coefficient` | Stretch coefficient $c$ controlling the steepness of the tanh function. | `4.0` |

### Tanh-Quadratic

A key issue with standard sparsity penalties ($L^0$, $L^1$, or $\tanh$) is that they only control the _average_ number of active features, but are indifferent to the _distribution_ of firing frequencies (See [Removing High Frequency Latents from JumpReLU SAEs](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks#Removing_High_Frequency_Latents_from_JumpReLU_SAEs) from the GDM Mech Interp Team for a detailed analysis). This allows some features to fire on a large fraction of inputs (>10%), which often leads to uninterpretable high-frequency features.

The tanh-quadratic loss addresses this by adding a quadratic term that specifically penalizes high-frequency features. First, an approximate frequency $\hat{p}_i$ is computed by averaging the tanh scores across samples:

$$\hat{p}_i = \mathbb{E}_{x}\left[\tanh(c \cdot f_i(x) \cdot \| W_{\text{dec},i} \|_2)\right]$$

Then the loss is:

$$L_s = \lambda \sum_i \hat{p}_i \left(1 + \frac{\hat{p}_i}{s}\right)$$

where $s$ is the frequency scale (controlled by `frequency_scale`). The first term $\hat{p}_i$ behaves like a standard sparsity penalty for low-frequency features ($\hat{p}_i \ll s$), while the quadratic term $\hat{p}_i^2 / s$ dominates for high-frequency features ($\hat{p}_i \gtrsim s$), making it increasingly expensive for features to activate on a large fraction of inputs.

This formulation successfully eliminates high-frequency latents with only a modest impact on reconstruction loss, while improving frequency-weighted interpretability scores compared to standard JumpReLU SAEs.

!!! note

    Our implementation of quadratic loss term uses $\tanh$ as differentiable $L^0$ proxies, which is different to the original proposal by GDM which directly use $L^0$ paired with straight-through estimators.

To use tanh-quadratic, set `#!python sparsity_loss_type="tanh-quad"` in `TrainerConfig`. Other parameters include:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `l1_coefficient` | Coefficient $\lambda$ for the sparsity loss term. | `0.00008` |
| `l1_coefficient_warmup_steps` | Steps (int) or fraction of total steps (float) to warm up the sparsity coefficient from 0. | `0.1` |
| `tanh_stretch_coefficient` | Stretch coefficient $c$ controlling the steepness of the tanh function. | `4.0` |
| `frequency_scale` | Scale factor $s$ for the quadratic penalty. Smaller values penalize high-frequency features more aggressively. Typical values are `0.1` or `0.01` to suppress features firing on >10% of tokens. | `0.01` |

## Auxiliary Losses

### JumpReLU Pre-act Loss

For JumpReLU SAEs, an additional $L_p$ penalty proposed by [Anthropic](https://transformer-circuits.pub/2025/january-update/index.html) can encourage the threshold to stay above the pre-activation values:

$$L_p = \lambda_p \sum_i \text{ReLU}(e^{\theta_i} - h_i) \| W_{\text{dec},i} \|_2$$

where $\theta_i$ is the log-threshold and $h_i$ is the pre-activation. This loss pushes the threshold higher, reducing the number of active features.

To use the JumpReLU $L^p$ penalty, set `lp_coefficient` to a positive value in `TrainerConfig`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lp_coefficient` | Coefficient $\lambda_p$ for the JumpReLU $L^p$ penalty. Set to `None` to disable. | `None` |

### Aux-K Loss

WIP @Junxuan Wang

## Legacy Stategies

Early researches on SAEs employ strategies like [Neuron Resampling](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-resampling) and [Ghost Grads](https://transformer-circuits.pub/2024/jan-update/index.html#dict-learning-resampling) to make dead neurons live again. However, modern initialization and sparsity losses have largely alleviated dead neurons. Thus, we remove the support for these strategies for simplicity of our internal code structure.

## Caching Activations

Training with cached activations is a common workflow in practice. It enables efficient hyperparameter sweeping by reusing pre-generated activations and facilitates parallelized training and analysis (DP/TP). This approach significantly accelerates training; for example, training an 8x expansion SAE on Pythia 160M with 800M tokens typically drops from ~6 hours (on-the-fly) to ~30 minutes (cached). However, caching requires substantial disk space. For 800M tokens of activations from a single Pythia 160M layer ($d_{\text{model}}=768$) stored in FP16, the storage requirement is:

$$ 800 \times 10^6 \times 768 \times 2 \text{ bytes} \approx 1.2 \text{ TB} $$

In this workflow, a separate task caches activations to disk at the output of `ActivationFactory`. When training, we re-configure the `ActivationFactory` to directly read from disk instead of generating activation from the language model on the fly.

To cache activation on disk, you can:

=== "Runner"

    Create the `GenerateActivationsSettings` in Python and call `generate_activations` with it. Configurations except `output_dir` and `total_tokens` should be consistent with on-the-fly settings [above](#using-the-high-level-runner-api). `output_dir` is where you want to place your generated activations. Ensure you have enough space at this directory. `total_tokens` should be equal or greater than the `total_training_tokens` you want to train your SAE on. 

    ```python
    from lm_saes import (
        GenerateActivationsSettings,
        generate_activations,
        LanguageModelConfig,
        DatasetConfig,
        ActivationFactoryTarget,
        BufferShuffleConfig,
    )

    settings = GenerateActivationsSettings(
        model=LanguageModelConfig(
            model_name="EleutherAI/pythia-160m",
            device="cuda",
            dtype="torch.float16",
        ),
        model_name="pythia-160m",
        dataset=DatasetConfig(dataset_name_or_path="Hzfinfdu/SlimPajama-3B"),
        dataset_name="SlimPajama-3B",
        hook_points=["blocks.6.hook_resid_post"],
        output_dir="path/to/activations",
        total_tokens=800_000_000,
        context_size=1024,
        target=ActivationFactoryTarget.ACTIVATIONS_1D,
        model_batch_size=32,
        batch_size=4096,
        buffer_size=16384,
        buffer_shuffle=BufferShuffleConfig(
            perm_seed=42,
            generator_device="cuda",
        ),
        device_type="cuda",
    )

    generate_activations(settings)
    ```

=== "CLI"

    CLI-based workflow requires a configuration file containing the settings consistent with `GenerateActivationsSettings`. Common configuration file type like TOML, JSON and YAML are supported.

    Create a TOML configuration file (e.g., `generate_config.toml`) with the following content:

    ```toml
    model_name = "pythia-160m"
    dataset_name = "SlimPajama-3B"
    hook_points = ["blocks.6.hook_resid_post"]
    output_dir = "path/to/activations"
    total_tokens = 800_000_000
    context_size = 1024
    target = "activations-1d"
    model_batch_size = 32
    batch_size = 4096
    buffer_size = 16384
    device_type = "cuda"

    [model]
    model_name = "EleutherAI/pythia-160m"
    device = "cuda"
    dtype = "torch.float16"

    [dataset]
    dataset_name_or_path = "Hzfinfdu/SlimPajama-3B"

    [buffer_shuffle]
    perm_seed = 42
    generator_device = "cuda"
    ```

    Then run the generation with:

    ```bash
    lm-saes generate generate_config.toml
    ```

=== "Full Script"

    Also, you can directly create `ActivationFactory` and `ActivationWriter` instances to generate and write activations to disk.

    ```python
    import datasets
    from lm_saes import (
        LanguageModelConfig,
        TransformerLensLanguageModel,
        ActivationFactory,
        ActivationFactoryConfig,
        ActivationFactoryDatasetSource,
        ActivationFactoryTarget,
        BufferShuffleConfig,
        ActivationWriter,
        ActivationWriterConfig,
    )

    # Use same way to generate activations
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

    factory_cfg = ActivationFactoryConfig(
        sources=[ActivationFactoryDatasetSource(name="SlimPajama-3B")],
        target=ActivationFactoryTarget.ACTIVATIONS_1D,
        hook_points=["blocks.6.hook_resid_post"],
        context_size=1024,
        model_batch_size=32,
        batch_size=4096,
        buffer_size=16384,
        buffer_shuffle=BufferShuffleConfig(
            perm_seed=42,
            generator_device="cuda",
        ),
    )
    factory = ActivationFactory(factory_cfg)

    activations = factory.process(
        model=model,
        model_name="pythia-160m",
        datasets={"SlimPajama-3B": (dataset, None)},
    )

    # Create an ActivationWriter to write the activation stream to disk
    writer_cfg = ActivationWriterConfig(
        hook_points=["blocks.6.hook_resid_post"],
        total_generating_tokens=800_000_000,
        cache_dir="path/to/activations",
    )
    writer = ActivationWriter(writer_cfg)
    
    writer.process(activations)
    ```

### Training with Cached Activations

Once you have generated and saved activations to disk, you can configure the `ActivationFactory` to read from these files instead of running the language model. This is done by replacing `ActivationFactoryDatasetSource` with `ActivationFactoryActivationsSource` in the configuration.

=== "Runner"

    ```python
    import torch
    from lm_saes import (
        TrainSAESettings,
        train_sae,
        SAEConfig,
        TrainerConfig,
        ActivationFactoryConfig,
        ActivationFactoryActivationsSource,
        ActivationFactoryTarget,
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
        trainer=TrainerConfig(
            amp_dtype=torch.float32,
            lr=1e-4,
            total_training_tokens=800_000_000,
            exp_result_path="results",
        ),
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryActivationsSource(
                    path="path/to/activations",
                    name="pythia-160m-cached",
                    device="cuda",
                )
            ],
            target=ActivationFactoryTarget.ACTIVATIONS_1D,
            hook_points=["blocks.6.hook_resid_post"],
            batch_size=4096,
        ),
        sae_name="pythia-160m-sae",
        sae_series="pythia-sae",
    )

    train_sae(settings)
    ```

=== "CLI"

    Update your training configuration to use the `activations` source type:

    ```toml
    # ... other configurations ...

    [activation_factory]
    target = "activations-1d"
    hook_points = ["blocks.6.hook_resid_post"]
    batch_size = 4096

    [[activation_factory.sources]]
    type = "activations"
    name = "pythia-160m-cached"
    path = "path/to/activations"
    device = "cuda"
    ```

=== "Full Script"

    In a full script, you can omit the language model and dataset loading, and directly use `ActivationFactory` with cached sources:

    ```python
    import torch
    from lm_saes import (
        ActivationFactory,
        ActivationFactoryConfig,
        ActivationFactoryActivationsSource,
        ActivationFactoryTarget,
        SAEConfig,
        Trainer,
        TrainerConfig,
        SparseAutoEncoder,
    )

    # 1. Configure Activation Factory with Cached Source
    factory_cfg = ActivationFactoryConfig(
        sources=[
            ActivationFactoryActivationsSource(
                path="path/to/activations",
                name="pythia-160m-cached",
                device="cuda",
            )
        ],
        target=ActivationFactoryTarget.ACTIVATIONS_1D,
        hook_points=["blocks.6.hook_resid_post"],
        batch_size=4096,
    )

    factory = ActivationFactory(factory_cfg)
    activations_stream = factory.process()

    # 2. Initialize SAE and Trainer
    sae = SparseAutoEncoder(SAEConfig(
        hook_point_in="blocks.6.hook_resid_post",
        hook_point_out="blocks.6.hook_resid_post",
        d_model=768,
        expansion_factor=8,
        act_fn="relu",
        device="cuda",
    ))

    trainer = Trainer(TrainerConfig(
        lr=1e-4,
        total_training_tokens=800_000_000,
        exp_result_path="results",
    ))

    # 3. Train
    trainer.fit(sae=sae, activation_stream=activations_stream)
    ```

## Use HuggingFace Backend