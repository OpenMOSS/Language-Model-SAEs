import pytest
import torch
from datasets import Dataset
from pytest_mock import MockerFixture
from transformer_lens import HookedTransformer

from lm_saes.activation.factory import ActivationFactory, ActivationFactoryTarget
from lm_saes.config import ActivationFactoryConfig, ActivationFactoryDatasetSource


@pytest.fixture
def mock_model(mocker: MockerFixture) -> HookedTransformer:
    model = mocker.Mock(spec=HookedTransformer)

    def to_tokens_with_origins_side_effect(x, **kwargs):
        assert "text" in x
        return torch.tensor([[1, 2, 3]]) + len(x["text"])

    model.to_tokens_with_origins.side_effect = to_tokens_with_origins_side_effect

    def run_with_cache_side_effect(tokens, **kwargs):
        seq_len = tokens.shape[0]
        return (
            None,
            {
                "h0": torch.arange(seq_len * 3).reshape(1, seq_len, 3),
                "h1": torch.arange(seq_len * 3, seq_len * 6).reshape(1, seq_len, 3),
            },
        )

    model.run_with_cache_until.side_effect = run_with_cache_side_effect
    model.tokenizer = mocker.Mock()
    model.tokenizer.eos_token_id = 1
    model.tokenizer.pad_token_id = 0
    model.tokenizer.bos_token_id = 2
    return model


@pytest.fixture
def mock_dataset() -> Dataset:
    return Dataset.from_dict({"text": ["Hello world", "Test text", "Another example"]})


@pytest.fixture
def basic_config() -> ActivationFactoryConfig:
    return ActivationFactoryConfig(
        sources=[
            ActivationFactoryDatasetSource(
                name="test_dataset",
                prepend_bos=True,
                sample_weights=1.0,
            )
        ],
        target=ActivationFactoryTarget.BATCHED_ACTIVATIONS_1D,
        hook_points=["h0", "h1"],
        context_size=4,
        batch_size=2,
        buffer_size=None,
    )


def test_activation_factory_initialization(basic_config: ActivationFactoryConfig):
    factory = ActivationFactory(basic_config)
    assert factory.cfg == basic_config
    assert len(factory.pre_aggregation_processors) == 1
    assert factory.post_aggregation_processor is not None
    assert factory.aggregator is not None


def test_activation_factory_tokens_target(
    basic_config: ActivationFactoryConfig,
    mock_model: HookedTransformer,
    mock_dataset: Dataset,
):
    basic_config.target = ActivationFactoryTarget.TOKENS
    factory = ActivationFactory(basic_config)

    result = list(
        factory.process(
            model=mock_model,
            datasets={"test_dataset": (mock_dataset, {"shard_idx": 0, "n_shards": 8})},
        )
    )
    print(result)

    assert len(result) == 3  # One for each input text
    assert "tokens" in result[0]
    assert "meta" in result[0]
    assert result[0]["meta"]["dataset_name"] == "test_dataset"
    assert result[0]["meta"]["shard_idx"] == 0
    assert result[0]["meta"]["n_shards"] == 8
    assert torch.allclose(result[0]["tokens"], torch.tensor([12, 13, 14]))


def test_activation_factory_activations_2d_target(
    basic_config: ActivationFactoryConfig,
    mock_model: HookedTransformer,
    mock_dataset: Dataset,
):
    basic_config.target = ActivationFactoryTarget.ACTIVATIONS_2D
    factory = ActivationFactory(basic_config)

    result = list(factory.process(model=mock_model, datasets={"test_dataset": (mock_dataset, None)}))
    print(result)

    assert len(result) == 3  # One for each input text
    assert all(h in result[0] for h in basic_config.hook_points)
    assert result[0]["meta"]["dataset_name"] == "test_dataset"
    assert result[0]["h0"].shape == (4, 3)
    assert torch.allclose(result[0]["h0"], torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))


def test_activation_factory_activations_1d_target(
    basic_config: ActivationFactoryConfig,
    mock_model: HookedTransformer,
    mock_dataset: Dataset,
):
    basic_config.target = ActivationFactoryTarget.ACTIVATIONS_1D
    factory = ActivationFactory(basic_config)

    result = list(factory.process(model=mock_model, datasets={"test_dataset": (mock_dataset, None)}))

    assert len(result) == 3  # One for each input text
    assert all(h in result[0] for h in basic_config.hook_points)
    assert result[0]["meta"]["dataset_name"] == "test_dataset"
    assert result[0]["h0"].ndim == 2
    assert result[0]["h0"].shape == (3, 3)
    assert torch.allclose(result[0]["h0"], torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))


def test_activation_factory_batched_activations_1d_target(
    basic_config: ActivationFactoryConfig,
    mock_model: HookedTransformer,
    mock_dataset: Dataset,
):
    factory = ActivationFactory(basic_config)

    result = list(factory.process(model=mock_model, datasets={"test_dataset": (mock_dataset, None)}))
    print(result)

    # With batch_size=2 and 3 samples * 3 activations per sample, we expect 5 batches, with 2 samples in the first 4
    # batches and 1 sample in the last batch
    assert len(result) == 5
    assert all(h in result[0] for h in basic_config.hook_points)
    assert tuple([result[i]["h0"].shape[0] for i in range(5)]) == (2, 2, 2, 2, 1)
    assert torch.allclose(result[0]["h0"], torch.tensor([[0, 1, 2], [3, 4, 5]]))
    assert torch.allclose(result[1]["h0"], torch.tensor([[6, 7, 8], [0, 1, 2]]))
    assert torch.allclose(result[2]["h0"], torch.tensor([[3, 4, 5], [6, 7, 8]]))
    assert torch.allclose(result[3]["h0"], torch.tensor([[0, 1, 2], [3, 4, 5]]))
    assert torch.allclose(result[4]["h0"], torch.tensor([[6, 7, 8]]))
    assert "meta" not in result[0]  # Info is removed for batched activations


def test_activation_factory_multiple_sources(
    basic_config: ActivationFactoryConfig,
    mock_model: HookedTransformer,
    mock_dataset: Dataset,
):
    # Create config with two sources
    basic_config.sources = [
        ActivationFactoryDatasetSource(
            name="dataset1",
            prepend_bos=True,
            sample_weights=0.7,
        ),
        ActivationFactoryDatasetSource(
            name="dataset2",
            prepend_bos=False,
            sample_weights=0.3,
        ),
    ]
    basic_config.target = ActivationFactoryTarget.ACTIVATIONS_2D

    factory = ActivationFactory(basic_config)

    datasets = {
        "dataset1": (mock_dataset, None),
        "dataset2": (mock_dataset, None),
    }

    result = list(factory.process(model=mock_model, datasets=datasets))

    assert len(result) > 0
    for item in result:
        assert item["meta"]["dataset_name"] in ["dataset1", "dataset2"]


def test_activation_factory_invalid_dataset(
    basic_config: ActivationFactoryConfig,
    mock_model: HookedTransformer,
):
    factory = ActivationFactory(basic_config)

    with pytest.raises(AssertionError, match="Dataset test_dataset not found in `datasets`"):
        list(factory.process(model=mock_model, datasets={}))


def test_activation_factory_missing_model(
    basic_config: ActivationFactoryConfig,
    mock_dataset: Dataset,
):
    factory = ActivationFactory(basic_config)

    with pytest.raises(AssertionError, match="`model` must be provided for dataset sources"):
        list(factory.process(datasets={"test_dataset": mock_dataset}))
