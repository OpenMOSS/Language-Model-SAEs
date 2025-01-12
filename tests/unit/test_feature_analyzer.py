import pytest
import torch
from pytest_mock import MockerFixture

from lm_saes.analysis.feature_analyzer import FeatureAnalyzer
from lm_saes.config import FeatureAnalyzerConfig, SAEConfig
from lm_saes.sae import SparseAutoEncoder


@pytest.fixture
def feature_analyzer_config() -> FeatureAnalyzerConfig:
    return FeatureAnalyzerConfig(
        total_analyzing_tokens=1000,
        enable_sampling=True,
        sample_weight_exponent=2.0,
        subsamples={
            "top": {"n_samples": 2, "proportion": 1.0},
            "mid": {"n_samples": 2, "proportion": 0.5},
        },
    )


@pytest.fixture
def feature_analyzer(feature_analyzer_config: FeatureAnalyzerConfig) -> FeatureAnalyzer:
    return FeatureAnalyzer(feature_analyzer_config)


def test_process_batch(feature_analyzer: FeatureAnalyzer):
    """Test _process_batch method with sampling enabled."""
    # Setup test data
    feature_acts = torch.tensor(
        [
            [[1.0, 0.5], [0.3, 0.8]],  # batch 0
            [[0.2, 0.9], [0.4, 0.6]],  # batch 1
        ]
    )  # shape: (2, 2, 2) - (batch_size, context_size, d_sae)

    discrete_meta = {
        "dataset": torch.tensor([0, 1]),  # batch_size
        "context": torch.tensor([0, 1]),  # batch_size
    }

    sample_result = {
        "top": None,
        "mid": None,
    }

    max_feature_acts = torch.tensor([0.8, 0.7])  # d_sae

    # Run method
    result = feature_analyzer._process_batch(
        feature_acts=feature_acts,
        discrete_meta=discrete_meta,
        sample_result=sample_result,
        max_feature_acts=max_feature_acts,
    )

    # Verify results
    assert "top" in result
    assert "mid" in result
    assert all(k in result["top"] for k in ["elt", "feature_acts", "dataset", "context"])
    assert result["top"]["feature_acts"].shape[1] == 2  # n_samples
    assert result["top"]["feature_acts"].shape[2] == 2  # context_size


def test_analyze_chunk_no_sampling(
    feature_analyzer_config: FeatureAnalyzerConfig,
    mocker: MockerFixture,
):
    """Test analyze_chunk method with sampling disabled."""
    # Modify config to disable sampling
    feature_analyzer_config.enable_sampling = False
    feature_analyzer = FeatureAnalyzer(feature_analyzer_config)

    # Mock SAE
    mock_sae = mocker.Mock(spec=SparseAutoEncoder)
    mock_sae.cfg = mocker.Mock(spec=SAEConfig)
    mock_sae.cfg.d_sae = 2
    mock_sae.cfg.device = "cpu"
    mock_sae.cfg.dtype = torch.float32
    mock_sae.cfg.hook_point_in = "activations_in"
    mock_sae.cfg.hook_point_out = "activations_out"

    # Create carefully crafted feature activations
    # Shape: (batch_size=2, context_size=2, d_sae=2)
    mock_sae.encode.return_value = torch.tensor(
        # Batch 0
        [
            # Context 0
            [
                [1.0, 0.0],  # Token 0: Feature 0 strongly active, Feature 1 inactive
                [0.0, 0.8],  # Token 1: Feature 0 inactive, Feature 1 strongly active
            ],
            # Context 1
            [
                [0.5, 0.3],  # Token 0: Both features moderately active
                [0.2, 0.6],  # Token 1: Feature 1 more active than Feature 0
            ],
        ]
    )

    # Create mock activation stream
    activation_stream = [
        {
            "activations_in": torch.randn(2, 2, 10),  # (batch_size, context_size, d_model)
            "tokens": torch.randint(0, 1000, (2, 2)),  # (batch_size, context_size)
            "meta": [
                {
                    "dataset": "train",
                    "context": "ctx1",
                },
                {
                    "dataset": "valid",
                    "context": "ctx2",
                },
            ],
        },
    ]

    # Run analysis
    results = feature_analyzer.analyze_chunk(activation_stream, mock_sae)

    # Verify results with sampling disabled
    assert len(results) == 2  # Two features

    # Check feature 0 results
    assert torch.allclose(torch.tensor(results[0]["max_feature_acts"]), torch.tensor(1.0))
    assert results[0]["act_times"] == 3  # Active in 3 positions (1.0, 0.5, 0.2)
    assert results[0]["samplings"][0]["name"] == "top"
    assert torch.allclose(
        torch.tensor(results[0]["samplings"][0]["feature_acts"]), torch.tensor([[1.0, 0.0], [0.5, 0.2]])
    )
    assert results[0]["samplings"][0]["dataset"] == ["train", "valid"]
    assert results[0]["samplings"][0]["context"] == ["ctx1", "ctx2"]
    assert results[0]["samplings"][1]["name"] == "mid"
    assert torch.allclose(torch.tensor(results[0]["samplings"][1]["feature_acts"]), torch.tensor([[0.5, 0.2]]))
    assert results[0]["samplings"][1]["dataset"] == ["valid"]
    assert results[0]["samplings"][1]["context"] == ["ctx2"]

    # Check feature 1 results
    assert torch.allclose(torch.tensor(results[1]["max_feature_acts"]), torch.tensor(0.8))
    assert results[1]["act_times"] == 3  # Active in 3 positions (0.8, 0.3, 0.6)
