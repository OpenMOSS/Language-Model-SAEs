import pytest
import torch

from llamascopium.models.clt import CLTConfig, CrossLayerTranscoder


@pytest.mark.parametrize("sparsity_include_decoder_norm", [True, False])
@pytest.mark.parametrize("keep_only_decoders", [False, True])
def test_encode_single_layer_matches_encode(sparsity_include_decoder_norm: bool, keep_only_decoders: bool):
    """Encoding one layer must give that layer's slice of the full encoding, also after
    `keep_only_decoders_for_layer_from`, which feature analysis applies before encoding."""
    torch.manual_seed(0)
    n_layers, d_model = 3, 8
    cfg = CLTConfig(
        hook_points_in=[f"in{i}" for i in range(n_layers)],
        hook_points_out=[f"out{i}" for i in range(n_layers)],
        d_model=d_model,
        expansion_factor=2,
        device="cpu",
        dtype=torch.float32,
        act_fn="relu",
        sparsity_include_decoder_norm=sparsity_include_decoder_norm,
    )
    x = torch.randn(5, n_layers, d_model)

    for layer in range(n_layers):
        clt = CrossLayerTranscoder(cfg)
        with torch.no_grad():
            for param in clt.parameters():
                param.normal_(0, 0.5)
            expected = clt.encode(x)[:, layer]
            if keep_only_decoders:
                clt.keep_only_decoders_for_layer_from(layer)
            actual = clt.encode_single_layer(x[:, layer], layer=layer)
        torch.testing.assert_close(actual, expected)
