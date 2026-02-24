# Sparse Dictionary Models

This section provides comprehensive documentation for the various sparse dictionary architectures supported by `Language-Model-SAEs`. While all models share the common goal of learning interpretable sparse representations of neural network activations, they differ in their architectural designs and the computational patterns they aim to capture.

## Overview

`Language-Model-SAEs` supports multiple sparse dictionary variants:

- **[Sparse Autoencoder (SAE)](sae.md)**: The foundational architecture that learns to decompose activations from a single layer into sparse, interpretable features.

- **[Transcoder](transcoder.md)**: Also known as Per-Layer Transcoder (PLT), this variant reads from one hook point and writes to another, enabling the decomposition of specific computational units like MLP layers.

- **[Cross Layer Transcoder (CLT)](clt.md)**: An advanced architecture that captures cross-layer interactions by allowing features extracted at one layer to influence reconstructions at multiple downstream layers.

- **[Low-Rank Sparse Attention (Lorsa)](lorsa.md)**: A specialized architecture designed to decompose attention computations into interpretable sparse components.

## Common Configuration Parameters

All sparse dictionary variants inherit from `BaseSAEConfig`, which provides common configuration parameters. These parameters are available for all model types unless specifically overridden.

### Core Architecture Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `d_model` | `int` | The dimension of the input/label activation space. In common settings where activations come from a transformer, this is the dimension of the model (also be known as `hidden_size`) | Required |
| `expansion_factor` | `float` | The expansion factor of the sparse dictionary. The hidden dimension of the sparse dictionary `d_sae` is `d_model * expansion_factor` | Required |
| `use_decoder_bias` | `bool` | Whether to use a bias term in the decoder. Including a bias term may make it easier to train a better sparse dictionary, in exchange for increased architectural complexity | `True` |

### Activation Function Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `act_fn` | `str` | The activation function to use for the sparse dictionary. Options: `"relu"`, `"jumprelu"`, `"topk"`, `"batchtopk"`, `"batchlayertopk"`, `"layertopk"`. See [Activation Functions](../train-saes.md#activation-functions) for details | `"relu"` |
| `top_k` | `int` | The k value to use for the TopK family of activation functions. For vanilla TopK, the L0 norm of the feature activations is `top_k` | `50` |

**Activation function descriptions:**

- `relu`: ReLU activation function. Used in the most vanilla SAE settings.
- `jumprelu`: JumpReLU activation function, adding a trainable element-wise threshold that pre-activations must pass to be activated. Proposed in [*Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders*](https://arxiv.org/abs/2407.14435).
- `topk`: TopK activation function. Retains the top K activations per sample, zeroing out the rest. Proposed in [*Scaling and evaluating sparse autoencoders*](https://openreview.net/forum?id=tcsZt9ZNKD).
- `batchtopk`: BatchTopK activation function. Relaxes TopK to batch-level, selecting the top `k * batch_size` activations per batch. Allows more adaptive allocation of latents on each sample. Proposed in [*BatchTopK Sparse Autoencoders*](https://arxiv.org/abs/2412.06410).
- `batchlayertopk`: (For CrossLayerTranscoder only) Extension of BatchTopK to layer-and-batch-aware, retaining the top `k * batch_size * n_layers` activations per batch and layer.
- `layertopk`: (For CrossLayerTranscoder only) Extension of TopK to layer-aware, retaining the top `k * n_layers` activations per layer.

### JumpReLU-Specific Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `jumprelu_threshold_window` | `float` | The window size for the JumpReLU threshold. When pre-activations are element-wise in the window-neighborhood of the threshold, the threshold will begin to receive gradient. See [Anthropic's Circuits Update - January 2025](https://transformer-circuits.pub/2025/january-update/index.html#DL) for more details | `2.0` |

### Activation Normalization Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `norm_activation` | `str` | The activation normalization strategy. Options: `"token-wise"`, `"batch-wise"`, `"dataset-wise"`, `"inference"`. During training, input/label activations are normalized to an average norm of $\sqrt{d_{\text{model}}}$, allowing easier hyperparameter transfer between different model scales | `"dataset-wise"` |

**Normalization strategies:**

- `token-wise`: Norm is directly computed for activation from each token. No averaging is performed.
- `batch-wise`: Norm is computed for each batch, then averaged over the batch dimension.
- `dataset-wise`: Norm is computed from several samples from the activation. Gives a fixed value of average norm for all activations, preserving the linearity of pre-activation encoding and decoding.
- `inference`: No normalization is performed. Produced after calling `standardize_parameters_of_dataset_norm` method, which folds the dataset-wise average norm into the weights and biases.

### Sparsity Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `sparsity_include_decoder_norm` | `bool` | Whether to include the decoder norm term in feature activation gating. If true, pre-activation hidden states will be scaled by the decoder norm before applying the activation function, then scaled back after. This suppresses the training dynamics where the model tries to increase decoder norm in exchange for smaller feature activation magnitude | `True` |

### Performance Optimization Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `use_triton_kernel` | `bool` | Whether to use the Triton SpMM kernel for sparse matrix multiplication. Currently only supported for vanilla SAE | `False` |
| `sparsity_threshold_for_triton_spmm_kernel` | `float` | The sparsity threshold for the Triton SpMM kernel. Only when feature activation sparsity reaches this threshold will the Triton SpMM kernel be used. Useful for JumpReLU or TopK with a k annealing schedule | `0.996` |
