# Sparse Autoencoder (SAE)

Sparse Autoencoders (SAEs) are the foundational architecture for learning interpretable features from language model activations. They decompose neural network activations into sparse, interpretable features that help address the superposition problem.

Given a model activation vector $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$, an SAE first **encodes** it into a high-dimensional sparse latent representation, then **decodes** it back to reconstruct the original activation:

$$
\begin{aligned}
\mathbf{z} &= \sigma(W_E \mathbf{x} + \mathbf{b}_E) \in \mathbb{R}^{d_{\text{SAE}}} \\
\hat{\mathbf{x}} &= W_D \mathbf{z} + \mathbf{b}_D \in \mathbb{R}^{d_{\text{model}}}
\end{aligned}
$$

where $W_E \in \mathbb{R}^{d_{\text{SAE}} \times d_{\text{model}}}$ and $W_D \in \mathbb{R}^{d_{\text{model}} \times d_{\text{SAE}}}$ are the encoder and decoder weight matrices, $\mathbf{b}_E, \mathbf{b}_D$ are bias terms, and $\sigma(\cdot)$ is a sparsity-inducing activation function (e.g., ReLU, TopK). The model is trained to minimize the reconstruction loss $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$ while keeping $\mathbf{z}$ sparse, encouraging each dimension of $\mathbf{z}$ to correspond to a monosemantic feature.

The architecture was introduced in foundational works including [*Sparse Autoencoders Find Highly Interpretable Features in Language Models*](https://arxiv.org/abs/2309.08600) and [*Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*](https://transformer-circuits.pub/2023/monosemantic-features). For detailed architectural specifications and mathematical formulations, please refer to these papers.

## Configuration

SAEs are configured using the `SAEConfig` class. All sparse dictionary models inherit common parameters from `BaseSAEConfig`. See the [Common Configuration Parameters](overview.md#common-configuration-parameters) section for the full list of inherited parameters.

### SAE-Specific Parameters

```python
from lm_saes import SAEConfig
import torch

sae_config = SAEConfig(
    # SAE-specific parameters
    hook_point_in="blocks.6.hook_resid_post",
    hook_point_out="blocks.6.hook_resid_post",  # Same as hook_point_in for SAE
    use_glu_encoder=False,
    
    # Common parameters (documented in Sparse Dictionaries overview)
    d_model=768,
    expansion_factor=8,
    act_fn="topk",
    top_k=64,
    dtype=torch.float32,
    device="cuda",
)
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `hook_point_in` | `str` | Hook point to read activations from. For SAE, this is typically the same as `hook_point_out` | Required |
| `hook_point_out` | `str` | Hook point to write reconstructions to. For SAE, this is typically the same as `hook_point_in` | Required |
| `use_glu_encoder` | `bool` | Whether to use a Gated Linear Unit (GLU) in the encoder. GLU can improve expressiveness but increases parameter count | `False` |

!!! note "SAE vs Transcoder"
    For standard SAEs, `hook_point_in` and `hook_point_out` are identical, meaning the SAE reads from and reconstructs to the same point in the model. When these two hook points differ, the configuration defines a [Transcoder](transcoder.md) instead.

### Initialization Strategy

Proper initialization is crucial for training high-quality SAEs. We recommend the following configuration:

```python
from lm_saes import InitializerConfig

initializer = InitializerConfig(
    bias_init_method="geometric_median",
    grid_search_init_norm=True,
    init_encoder_bias_with_mean_hidden_pre=True,
    # ... (e.g. init_log_jumprelu_threshold_value if use)
)
```

| Parameter | Recommended Value | Description |
|-----------|-------------------|-------------|
| `bias_init_method` | `"geometric_median"` | Initializes the decoder bias using the geometric median of the activation distribution, which is more robust to skewed/biased activations than `"all_zero"` |
| `grid_search_init_norm` | `True` | Performs a grid search to find the optimal encoder/decoder weight scale that minimizes initial MSE loss |
| `init_encoder_bias_with_mean_hidden_pre` | `True` | Initializes the encoder bias with the mean of the pre-activation distribution, which is more robust to skewed/biased activations and stabilizes early training |

#### Initialization for Low-Rank Activations

When training SAEs on low-rank activations (such as attention outputs), dead features become a prevalent problem due to the dimensional collapse in the activation space. As shown in [*Dimensional Collapse in Transformer Attention Outputs: A Challenge for Sparse Dictionary Learning*](https://arxiv.org/abs/2508.16929), attention outputs are confined to a surprisingly low-dimensional subspace (only ~60% of the full space), creating a mismatch between randomly initialized features and the intrinsic geometry of the activation space.

To address this issue, we recommend the following additional configuration:

```python
initializer = InitializerConfig(
    bias_init_method="geometric_median",
    grid_search_init_norm=True,
    init_encoder_bias_with_mean_hidden_pre=True,
    initialize_W_D_with_active_subspace=True,
    d_active_subspace=384,  # Adjust based on effective rank (e.g., 0.5 * d_model)
    # ... (e.g. init_log_jumprelu_threshold_value if use)
)
```

| Parameter | Recommended Value | Description |
|-----------|-------------------|-------------|
| `initialize_W_D_with_active_subspace` | `True` | Constrains decoder features to the active subspace of the activations using PCA or SVD, ensuring features align with the intrinsic geometry |
| `d_active_subspace` | `~0.5 * d_model` | Dimension of the active subspace. Should be adjusted based on the effective rank of your activations. For a model with `d_model=768`, starting with `384` is a good baseline |

This subspace-constrained initialization dramatically reduces dead features in attention output SAEs. The appropriate value for `d_active_subspace` depends on the effective rank of your specific activations and may require some tuning.

## Training

Training an SAE follows the same workflow as described in the [Train SAEs](../train-saes.md) guide. 
