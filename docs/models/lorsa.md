# Low-Rank Sparse Attention (Lorsa)

Low-Rank Sparse Attention (Lorsa) is a specialized sparse dictionary architecture designed to decompose attention layers into interpretable sparse components. Unlike standard SAEs that treat attention as a black box, Lorsa explicitly models the query-key-value structure while maintaining sparsity and interpretability.

Given an input sequence \(X \in \mathbb{R}^{n \times d}\), Lorsa has:

- \(n_{\text{qk\_heads}}\) QK heads, each with projections \(W_q^h, W_k^h \in \mathbb{R}^{d \times d_{\text{qk\_head}}}\)
- \(n_{\text{ov\_heads}}\) rank-1 OV heads, each with projections \(\mathbf{w}_v^i \in \mathbb{R}^{d \times 1}\), \(\mathbf{w}_o^i \in \mathbb{R}^{1 \times d}\)

Every group of \(n_{\text{ov\_heads}} / n_{\text{qk\_heads}}\) consecutive OV heads shares the same QK head. Denote the QK head assigned to OV head \(i\) as \(h(i)\). The forward pass for each OV head \(i\) is:

\[
\begin{aligned}
Q^{h(i)} &= X W_q^{h(i)}, \quad K^{h(i)} = X W_k^{h(i)} \\
A^{h(i)} &= \operatorname{softmax}\!\left(\frac{Q^{h(i)} {(K^{h(i)})}^\top}{\sqrt{d_{\text{qk\_head}}}}\right) \in \mathbb{R}^{n \times n} \\
\tilde{\mathbf{z}}^i &= A^{h(i)}\, (X \mathbf{w}_v^i) \in \mathbb{R}^{n \times 1}
\end{aligned}
\]

The pre-activations across all OV heads are then passed through a sparsity-inducing activation function \(\sigma(\cdot)\):

\[
[\mathbf{z}^0, \ldots, \mathbf{z}^{n_{\text{ov\_heads}}-1}] = \sigma([\tilde{\mathbf{z}}^0, \ldots, \tilde{\mathbf{z}}^{n_{\text{ov\_heads}}-1}])
\]

The final output sums the contributions of all OV heads weighted by their activations:

\[
\hat{Y} = \sum_{i=0}^{n_{\text{ov\_heads}}-1} \mathbf{z}^i\, (\mathbf{w}_o^i)^\top \in \mathbb{R}^{n \times d}
\]

The architecture was introduced in [*Towards Understanding the Nature of Attention with Low-Rank Sparse Decomposition*](https://openreview.net/forum?id=9A2etpDFIB) (ICLR 2026), which proposes using sparse dictionary learning to address *attention superposition*—the challenge of disentangling attention-mediated interactions between features at different token positions. For detailed architectural specifications and mathematical formulations, please refer to this paper.

## Configuration

Lorsa is configured using the `LorsaConfig` class. All sparse dictionary models inherit common parameters from `BaseSAEConfig`. See the [Common Configuration Parameters](overview.md#common-configuration-parameters) section for the full list of inherited parameters.

### Lorsa-Specific Parameters

```python
from lm_saes import LorsaConfig
import torch

lorsa_config = LorsaConfig(
    # Hook points
    hook_point_in="blocks.13.ln1.hook_normalized",
    hook_point_out="blocks.13.hook_attn_out",
    
    # Attention dimensions
    n_qk_heads=16,
    d_qk_head=128,
    n_ctx=2048,
    
    # Positional embeddings
    positional_embedding_type="rotary",
    rotary_dim=128,
    rotary_base=1000000,
    rotary_adjacent_pairs=False,
    rotary_scale=1,
    
    # NTK-aware RoPE (optional)
    use_NTK_by_parts_rope=False,
    NTK_by_parts_factor=1.0,
    NTK_by_parts_low_freq_factor=1.0,
    NTK_by_parts_high_freq_factor=1.0,
    old_context_len=2048,
    
    # Attention settings
    attn_scale=None,
    use_post_qk_ln=True,
    normalization_type="RMS",
    eps=1e-6,
    
    # Common parameters
    d_model=2048,
    expansion_factor=32,
    act_fn="topk",
    top_k=256,
    dtype=torch.float32,
    device="cuda",
)
```

#### Hook Points

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `hook_point_in` | `str` | Input hook point, typically the attention input (e.g., `blocks.L.ln1.hook_normalized`). Must differ from `hook_point_out` | Required |
| `hook_point_out` | `str` | Output hook point, typically the attention output (e.g., `blocks.L.hook_attn_out`). Must differ from `hook_point_in` | Required |

#### Attention Dimensions

We recommend setting `d_qk_head` to match the target model's head dimension. `n_qk_heads` can be freely chosen: a natural starting point is `n_qk_heads = n_heads * expansion_factor` (n_heads is the num of attention heads of target attention layer), though a smaller value is also reasonable if you want to reduce Lorsa's parameter count(not less than `n_heads`).

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `n_qk_heads` | `int` | Number of QK heads. | Required |
| `d_qk_head` | `int` | Dimension per QK head. | Required |
| `n_ctx` | `int` | Maximum context length. | Required |

!!! note "Number of OV Heads"
    The number of OV heads is automatically computed as: `n_ov_heads = expansion_factor * d_model` (same as `d_sae`).

#### Positional Embeddings

It is strongly recommended to copy the positional embedding parameters directly from the target model's implementation. Incorrect settings will make it harder for Lorsa to learn the target attention patterns.

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `positional_embedding_type` | `str` | Type of positional embedding: `"rotary"` or `"none"` | `"rotary"` |
| `rotary_dim` | `int` | Dimension of rotary embeddings (typically `d_qk_head`) | Required |
| `rotary_base` | `int` | Base for rotary embeddings frequency | `10000` |
| `rotary_adjacent_pairs` | `bool` | Whether to apply RoPE on adjacent pairs | `True` |
| `rotary_scale` | `int` | Scaling factor of the head dimension for rotary embeddings | `1` |

#### NTK-Aware RoPE (only for Llama 3.1 and 3.2 herd models)

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `use_NTK_by_parts_rope` | `bool` | Enable NTK-aware RoPE scaling for extended context | `False` |
| `NTK_by_parts_factor` | `float` | NTK scaling factor | `1.0` |
| `NTK_by_parts_low_freq_factor` | `float` | Low-frequency component scaling factor | `1.0` |
| `NTK_by_parts_high_freq_factor` | `float` | High-frequency component scaling factor | `1.0` |
| `old_context_len` | `int` | Original context length before scaling | `2048` |

#### Attention Computation Details

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `attn_scale` | `float | None` | Attention scaling factor. If `None`, uses $\frac{1}{\sqrt{d_{\text{qk\_head}}}}$ | `None` |
| `use_post_qk_ln` | `bool` | Apply LayerNorm/RMSNorm after computing Q and K projections | `False` |
| `normalization_type` | `str | None` | Normalization type: `"LN"` (LayerNorm) or `"RMS"` (RMSNorm). Only used when `use_post_qk_ln=True` | `None` |
| `eps` | `float` | Epsilon for numerical stability in normalization | `1e-6` |

### Initialization Strategy

For Lorsa, initialization from the original model's attention weights is highly recommended:

```python
InitializerConfig(
    grid_search_init_norm=True,
    initialize_lorsa_with_mhsa=True,  # Initialize Q, K from attention weights
    initialize_W_D_with_active_subspace=True,  # Initialize V, O from attention weights
    model_layer=13,  # Specify layer to extract attention weights from
)
```

This initialization helps Lorsa start from a good approximation of the attention computation.

## Training

### Basic Training Setup

Lorsa requires 2D activations with sequence dimension preserved (`ActivationFactoryTarget.ACTIVATIONS_2D`) since it models positional attention patterns:

```python
from lm_saes import (
    TrainLorsaSettings,
    train_lorsa,
    LorsaConfig,
    InitializerConfig,
    TrainerConfig,
    ActivationFactoryConfig,
    ActivationFactoryActivationsSource,
    ActivationFactoryTarget,
    LanguageModelConfig,
)
import torch

settings = TrainLorsaSettings(
    sae=LorsaConfig(
        hook_point_in="blocks.13.ln1.hook_normalized",
        hook_point_out="blocks.13.hook_attn_out",
        # ... other settings ...
    ),
    initializer=InitializerConfig(
        grid_search_init_norm=True,
        initialize_lorsa_with_mhsa=True,  # Initialize with original attention weights
        initialize_W_D_with_active_subspace=True,
        model_layer=13,
    ),
    trainer=TrainerConfig(
        lr=2e-4,
        total_training_tokens=800_000_000,
        initial_k=256,
        k_warmup_steps=1500,
        log_frequency=1000,
        exp_result_path="results/lorsa",
    ),
    activation_factory=ActivationFactoryConfig(
        sources=[
            ActivationFactoryActivationsSource(
                path="path/to/cached/activations",
                name="lorsa-activations",
                device="cuda",
            )
        ],
        target=ActivationFactoryTarget.ACTIVATIONS_2D,  # Preserve sequence dimension
        hook_points=[
            "blocks.13.ln1.hook_normalized",
            "blocks.13.hook_attn_out",
        ],
        batch_size=16,  # Batch size is per-sequence, not per-token
    ),
    sae_name="qwen-lorsa",
    sae_series="qwen-interpretability",
    model_name="Qwen/Qwen3-1.7B",
    model=LanguageModelConfig(
        model_name="Qwen/Qwen3-1.7B",
        device="cuda",
        dtype=torch.float16,
        model_from_pretrained_path="path/to/model",
    ),
    data_parallel_size=1,
    model_parallel_size=1,
)

train_lorsa(settings)
```

### Important Training Considerations

1. **Sequence batching**: Since Lorsa operates on sequences, `batch_size` in `ActivationFactoryConfig` represents the number of sequences (not tokens). The effective token batch size is `batch_size * n_ctx`.

2. **Memory requirements**: Lorsa stores attention patterns and requires more memory than standard SAEs. Consider using parallelism (see [distributed-guidelines](../distributed-guidelines.md)) reducing batch size.

3. **Context length**: Ensure `n_ctx` in `LorsaConfig` matches the `context_size` in `ActivationFactoryConfig` during activation generation.
