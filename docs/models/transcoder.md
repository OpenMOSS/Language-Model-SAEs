# Transcoder (Per-Layer Transcoder)

Transcoders, also known as Per-Layer Transcoders (PLTs), are a variant of Sparse Autoencoders that read activations from one hook point and reconstruct at a different hook point within the same layer. Transcoders are architecturally identical to Sparse Autoencoders, but use different activations for inputs and labels. This enables the decomposition of specific computational units, such as MLP sublayers, into interpretable sparse features.

Transcoders were proposed in [Anthropic Circuits Update - January 2024](https://transformer-circuits.pub/2024/jan-update/index.html#predict-future), and then explored in [*Transcoders Find Interpretable LLM Feature Circuits*](https://arxiv.org/abs/2406.11944) (Dunefsky et al., 2024) and [*Automatically Identifying Local and Global Circuits with Linear Computation Graphs*](https://arxiv.org/abs/2405.13868) (Ge et al., 2024), which demonstrate that transcoders can effectively decompose MLP computations into interpretable circuits while maintaining reconstruction fidelity.

## Configuration

Transcoders use the same [`SAEConfig`][lm_saes.SAEConfig] and [`InitializerConfig`][lm_saes.InitializerConfig] as standard SAEs. See the [SAE configuration guide](sae.md#configuration) for the full parameter reference.

The only essential difference is that `hook_point_in` and `hook_point_out` should point to **different** locations—typically the input and output of the MLP sublayer you want to decompose:

```python
transcoder_config = SAEConfig(
    hook_point_in="blocks.6.ln2.hook_normalized",  # before MLP
    hook_point_out="blocks.6.hook_mlp_out",        # after MLP
    ...
)
```

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
