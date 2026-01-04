import json
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from pytest_mock import MockerFixture
from safetensors.torch import save_file

from lm_saes import (
    ActivationFactoryActivationsSource,
    ActivationFactoryConfig,
    ActivationFactoryDatasetSource,
    ActivationFactoryTarget,
)
from lm_saes.activation.factory import ActivationFactory
from lm_saes.backend.language_model import LanguageModel


@pytest.fixture
def mock_model(mocker: MockerFixture) -> LanguageModel:
    model = mocker.Mock(spec=LanguageModel)

    def to_activations_side_effect(raw, hook_points, **kwargs):
        assert "text" in raw
        return {
            "h0": torch.arange(3 * 3).reshape(1, 3, 3),
            "h1": torch.arange(3 * 3, 3 * 6).reshape(1, 3, 3),
            "tokens": torch.arange(3).reshape(1, 3) + 1,
        }

    model.to_activations.side_effect = to_activations_side_effect
    model.eos_token_id = 1
    model.pad_token_id = 0
    model.bos_token_id = 2
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
        target=ActivationFactoryTarget.ACTIVATIONS_1D,
        hook_points=["h0", "h1"],
        context_size=4,
        num_workers=0,
        model_batch_size=1,
        batch_size=None,
        buffer_size=None,
    )


def test_activation_factory_initialization(basic_config: ActivationFactoryConfig):
    factory = ActivationFactory(basic_config)
    assert factory.cfg == basic_config
    assert len(factory.pre_aggregation_processors) == 1
    assert factory.post_aggregation_processor is not None
    assert factory.aggregator is not None


def test_activation_factory_activations_2d_target(
    basic_config: ActivationFactoryConfig,
    mock_model: LanguageModel,
    mock_dataset: Dataset,
):
    basic_config.target = ActivationFactoryTarget.ACTIVATIONS_2D
    factory = ActivationFactory(basic_config)

    result = list(factory.process(model=mock_model, model_name="test", datasets={"test_dataset": (mock_dataset, None)}))

    assert len(result) == 3  # One for each input text
    assert all(h in result[0] for h in basic_config.hook_points)
    assert result[0]["meta"][0]["dataset_name"] == "test_dataset"
    assert result[0]["h0"].shape == (1, 3, 3)
    assert torch.allclose(result[0]["h0"], torch.arange(9).reshape(1, 3, 3))


def test_activation_factory_activations_1d_target(
    basic_config: ActivationFactoryConfig,
    mock_model: LanguageModel,
    mock_dataset: Dataset,
):
    factory = ActivationFactory(basic_config)

    result = list(factory.process(model=mock_model, model_name="test", datasets={"test_dataset": (mock_dataset, None)}))

    assert len(result) == 3  # One for each input text
    assert all(h in result[0] for h in basic_config.hook_points)
    assert result[0]["h0"].shape == (1, 3)
    assert torch.allclose(result[0]["h0"], torch.arange(6, 9).reshape(1, 3))


def test_activation_factory_batched_activations_1d_target(
    basic_config: ActivationFactoryConfig,
    mock_model: LanguageModel,
    mock_dataset: Dataset,
):
    basic_config.batch_size = 2
    factory = ActivationFactory(basic_config)

    result = list(factory.process(model=mock_model, model_name="test", datasets={"test_dataset": (mock_dataset, None)}))

    # With batch_size=2 and 3 samples * 1 activations per sample, we expect 2 batches (2 samples in the first batch and 1
    # sample in the second batch)
    assert len(result) == 2
    assert all(h in result[0] for h in basic_config.hook_points)
    assert tuple(result[i]["h0"].shape[0] for i in range(2)) == (2, 1)
    assert torch.allclose(result[0]["h0"], torch.tensor([[6, 7, 8], [6, 7, 8]]))
    assert torch.allclose(result[1]["h0"], torch.tensor([[6, 7, 8]]))
    assert "meta" not in result[0]  # Info is removed for batched activations


def test_activation_factory_batched_activations_2d_target(
    basic_config: ActivationFactoryConfig,
    mock_model: LanguageModel,
    mock_dataset: Dataset,
):
    basic_config.target = ActivationFactoryTarget.ACTIVATIONS_2D
    basic_config.batch_size = 2
    factory = ActivationFactory(basic_config)

    result = list(factory.process(model=mock_model, model_name="test", datasets={"test_dataset": (mock_dataset, None)}))

    assert len(result) == 2  # [2, 1] samples
    assert all(h in result[0] for h in basic_config.hook_points)
    assert result[0]["meta"][0]["dataset_name"] == "test_dataset"
    assert result[0]["h0"].shape == (2, 3, 3)
    assert torch.allclose(
        result[0]["h0"],
        torch.stack(
            [
                torch.arange(9).reshape(3, 3),
                torch.arange(9).reshape(3, 3),
            ]
        ),
    )
    assert result[1]["h0"].shape == (1, 3, 3)
    assert torch.allclose(result[1]["h0"], torch.arange(9).reshape(1, 3, 3))


def test_activation_factory_multiple_sources(
    basic_config: ActivationFactoryConfig,
    mock_model: LanguageModel,
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

    result = list(factory.process(model=mock_model, model_name="test", datasets=datasets))

    assert len(result) > 0
    for item in result:
        assert item["meta"][0]["dataset_name"] in ["dataset1", "dataset2"]


def test_activation_factory_invalid_dataset(
    basic_config: ActivationFactoryConfig,
    mock_model: LanguageModel,
):
    factory = ActivationFactory(basic_config)

    with pytest.raises(AssertionError, match="Dataset test_dataset not found in `datasets`"):
        list(factory.process(model=mock_model, model_name="test", datasets={}))


def test_activation_factory_missing_model(
    basic_config: ActivationFactoryConfig,
    mock_dataset: Dataset,
    mock_model: LanguageModel,
):
    factory = ActivationFactory(basic_config)

    with pytest.raises(AssertionError, match="`model` must be provided for dataset sources"):
        list(factory.process(datasets={"test_dataset": (mock_dataset, None)}))

    with pytest.raises(AssertionError, match="`model_name` must be provided for dataset sources"):
        list(factory.process(model=mock_model, datasets={"test_dataset": (mock_dataset, None)}))


def test_activation_factory_activations_source(
    mocker: MockerFixture,
    tmp_path: Path,
    basic_config: ActivationFactoryConfig,
):
    # Setup mock activation files
    hook_dir = tmp_path / "h0"
    hook_dir.mkdir()

    # Create sample activation data
    sample_data = {
        "activation": torch.randn(2, 3, 4),  # (n_samples, n_context, d_model)
        "tokens": torch.randint(0, 1000, (2, 3)),  # (n_samples, n_context)
        "meta": [{"context_id": f"ctx_{i}"} for i in range(2)],
    }

    # Create mock files
    for i in range(2):
        chunk_path = hook_dir / f"chunk-{i}.safetensors"
        meta_path = hook_dir / f"chunk-{i}.meta.json"

        # Save activation data using safetensors
        save_file({"activation": sample_data["activation"], "tokens": sample_data["tokens"]}, chunk_path)
        # Save meta data
        with open(meta_path, "w") as f:
            json.dump(sample_data["meta"], f)

    # Configure factory to use activation source
    basic_config.sources = [
        ActivationFactoryActivationsSource(
            name="test_activations",
            path=str(tmp_path),
            sample_weights=1.0,
            device="cpu",
            dtype=None,
            num_workers=0,
            prefetch=None,
        )
    ]
    basic_config.target = ActivationFactoryTarget.ACTIVATIONS_2D
    basic_config.hook_points = ["h0"]

    # Initialize factory and process data
    factory = ActivationFactory(basic_config)
    result = list(factory.process())

    # Verify results
    assert len(result) == 2  # 2 chunks * 2 samples
    for item in result:
        # Check activation shape and content
        assert "h0" in item
        assert item["h0"].shape == (2, 3, 4)  # Shape should be (batch_size, context_size, embedding_dim)
        # Check tokens
        assert "tokens" in item
        assert item["tokens"].shape == (2, 3)  # Shape should be (batch_size, context_size)
        # Check metadata
        assert "meta" in item
        assert item["meta"][0]["context_id"] in ["ctx_0", "ctx_1"]


def test_activation_factory_activations_source_invalid_target(basic_config: ActivationFactoryConfig):
    # Configure factory with invalid target
    basic_config.sources = [
        ActivationFactoryActivationsSource(
            name="test_activations",
            path="dummy/path",
            sample_weights=1.0,
            device="cpu",
            dtype=None,
            num_workers=0,
            prefetch=2,
        )
    ]
    basic_config.target = ActivationFactoryTarget.TOKENS  # Too low level target for activation source

    # Should raise error when initializing factory
    with pytest.raises(ValueError, match="Activations sources are only supported for target >= ACTIVATIONS_2D"):
        ActivationFactory(basic_config)


def test_before_aggregation_interceptor(
    basic_config: ActivationFactoryConfig,
    mock_model: LanguageModel,
    mock_dataset: Dataset,
):
    """Test the before_aggregation_interceptor parameter."""
    basic_config.target = ActivationFactoryTarget.ACTIVATIONS_2D

    # Create an interceptor that adds a source_idx field to the data
    def interceptor(data: dict, source_idx: int) -> dict:
        data["source_idx"] = source_idx
        return data

    factory = ActivationFactory(basic_config, before_aggregation_interceptor=interceptor)

    result = list(factory.process(model=mock_model, model_name="test", datasets={"test_dataset": (mock_dataset, None)}))

    assert len(result) > 0
    for item in result:
        assert "source_idx" in item
        assert item["source_idx"] == 0  # Only one source with index 0
