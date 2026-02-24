# Transcoder (Per-Layer Transcoder)

Transcoders, also known as Per-Layer Transcoders (PLTs), are a variant of Sparse Autoencoders that read activations from one hook point and reconstruct at a different hook point within the same layer. This enables the decomposition of specific computational units, such as MLP sublayers, into interpretable sparse features. Unlike standard SAEs where `hook_point_in == hook_point_out`, transcoders have different input and output hook points. This allows them to faithfully approximate a computational unit (like an MLP layer) with a wider, sparsely-activating layer, making fine-grained circuit analysis more tractable.

Transcoders were introduced in the following papers: [*Automatically Identifying Local and Global Circuits with Linear Computation Graphs*](https://arxiv.org/abs/2405.13868) (Ge et al., 2024) and [*Transcoders Find Interpretable LLM Feature Circuits*](https://arxiv.org/abs/2406.11944) (Dunefsky et al., 2024). These works demonstrate that transcoders can effectively decompose MLP computations into interpretable circuits while maintaining reconstruction fidelity. For detailed architectural specifications and mathematical formulations, please refer to these papers.

## Configuration

Transcoders use the same `SAEConfig` class as standard SAEs. All sparse dictionary models inherit common parameters from `BaseSAEConfig`. See the [Common Configuration Parameters](overview.md#common-configuration-parameters) section for the full list of inherited parameters.

### Transcoder-Specific Parameters

```python
from lm_saes import SAEConfig
import torch

transcoder_config = SAEConfig(
    # Transcoder-specific: different hook points
    hook_point_in="blocks.6.ln2.hook_normalized",  # Input to MLP
    hook_point_out="blocks.6.hook_mlp_out",        # Output from MLP
    use_glu_encoder=False,
    
    # Common parameters (documented in Sparse Dictionaries overview)
    d_model=768,
    expansion_factor=32,
    act_fn="topk",
    top_k=64,
    dtype=torch.float32,
    device="cuda",
)
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `hook_point_in` | `str` | Hook point before the computational unit (e.g., `blocks.L.ln2.hook_normalized` for MLP input). Must differ from `hook_point_out` for transcoders | Required |
| `hook_point_out` | `str` | Hook point after the computational unit (e.g., `blocks.L.hook_mlp_out` for MLP output). Must differ from `hook_point_in` for transcoders | Required |
| `use_glu_encoder` | `bool` | Whether to use a Gated Linear Unit (GLU) in the encoder. GLU can improve expressiveness but increases parameter count | `False` |

!!! important "Transcoder vs SAE"
    When `hook_point_in != hook_point_out`, the configuration defines a transcoder rather than a standard SAE. This allows the model to learn the transformation between two different points in the network.

### Initialization Strategy

Proper initialization is crucial for training high-quality transcoders. We recommend the following configuration:

```python
from lm_saes import InitializerConfig

initializer = InitializerConfig(
    bias_init_method="geometric_median",
    init_encoder_bias_with_mean_hidden_pre=True,
    init_encoder_with_decoder_transpose=False,
    grid_search_init_norm=True,
    initialize_tc_with_mlp=True,
    model_layer=6,  # Specify which layer to extract MLP weights from
)
```

| Parameter | Recommended Value | Description |
|-----------|-------------------|-------------|
| `bias_init_method` | `"geometric_median"` | Initializes the decoder bias using the geometric median of the activation distribution, which is more robust to skewed/biased activations than `"all_zero"` |
| `init_encoder_bias_with_mean_hidden_pre` | `True` | Initializes the encoder bias with the mean of the pre-activation distribution, which is more robust to skewed/biased activations and stabilizes early training |
| `init_encoder_with_decoder_transpose` | `False` | Disables encoder initialization from decoder transpose. This is typically set to `False` when training transcoder |
| `grid_search_init_norm` | `True` | Performs a grid search to find the optimal encoder/decoder weight scale that minimizes initial MSE loss |
| `initialize_tc_with_mlp` | `True` | Initializes the transcoder decoder weights with the corresponding MLP layer weights. This helps the transcoder start from a good approximation of the MLP computation |
| `model_layer` | Layer index | Specifies which layer to extract MLP weights from. Should match the layer number in your `hook_point_in`/`hook_point_out` configuration |

This initialization strategy is particularly effective for transcoders decomposing MLP sublayers, as it allows the transcoder to start from a good approximation of the target computation and converge faster during training.


## Training

Training a Transcoder follows the same workflow as described in the [Train SAEs](../train-saes.md) guide. 
