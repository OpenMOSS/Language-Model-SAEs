import torch
from datasets import Dataset
from pytest_mock import MockerFixture

from lm_saes.activation.processors.activation import (
    ActivationBatchler,
    ActivationBuffer,
    ActivationGenerator,
    ActivationTransformer,
)
from lm_saes.activation.processors.huggingface import HuggingFaceDatasetLoader
from lm_saes.backend.language_model import LanguageModel


def test_huggingface_dataset_loader():
    # Create a mock dataset
    data = {"text": ["Hello, world!", "This is a test.", "Another example.", "Yet another one."]}
    dataset = Dataset.from_dict(data)

    # Test without info
    loader = HuggingFaceDatasetLoader(batch_size=2)
    result = list(loader.process(dataset))
    assert len(result) == 4  # Should be flattened
    assert all(isinstance(x, dict) for x in result)
    assert all("text" in x for x in result)
    assert all("meta" not in x for x in result)

    # Test with info
    loader_with_info = HuggingFaceDatasetLoader(batch_size=2, with_info=True)
    result_with_info = list(loader_with_info.process(dataset))
    assert len(result_with_info) == 4
    assert all("meta" in x for x in result_with_info)
    assert all("context_idx" in x["meta"] for x in result_with_info)
    assert all("dataset_name" not in x["meta"] for x in result_with_info)

    # Test with info and dataset_name
    loader_with_info = HuggingFaceDatasetLoader(batch_size=2, with_info=True)
    result_with_info = list(loader_with_info.process(dataset, dataset_name="test"))
    assert len(result_with_info) == 4
    assert all("meta" in x for x in result_with_info)
    assert all("context_idx" in x["meta"] for x in result_with_info)
    assert all("dataset_name" in x["meta"] for x in result_with_info)
    assert all(x["meta"]["dataset_name"] == "test" for x in result_with_info)

    # Test multiple workers
    loader_with_info = HuggingFaceDatasetLoader(batch_size=2, with_info=True, num_workers=2)
    result_with_info = list(loader_with_info.process(dataset))
    assert len(result_with_info) == 4
    assert all("meta" in x for x in result_with_info)
    assert all("context_idx" in x["meta"] for x in result_with_info)

    # Test with metadata
    metadata = {"source": "test_source", "split": "test_split"}
    loader_with_info = HuggingFaceDatasetLoader(batch_size=2, with_info=True)
    result_with_info = list(loader_with_info.process(dataset, metadata=metadata))
    assert len(result_with_info) == 4
    assert all("meta" in x for x in result_with_info)
    assert all("source" in x["meta"] for x in result_with_info)
    assert all("split" in x["meta"] for x in result_with_info)
    assert all(x["meta"]["source"] == "test_source" for x in result_with_info)
    assert all(x["meta"]["split"] == "test_split" for x in result_with_info)


def test_activation_processors(mocker: MockerFixture):
    # Mock LanguageModel
    mock_model = mocker.Mock(spec=LanguageModel)
    mock_model.preprocess_raw_data.side_effect = lambda x: x
    mock_model.to_activations.return_value = {
        "h0": torch.arange(9).reshape(1, 3, 3),
        "h1": torch.arange(9, 18).reshape(1, 3, 3),
        "tokens": torch.tensor([[1, 2, 3]]),
    }

    # Test ActivationGenerator
    hook_points = ["h0", "h1"]
    generator = ActivationGenerator(hook_points=hook_points, batch_size=1)
    input_data = [{"text": "test text", "meta": {"some": "meta"}}]
    result = list(generator.process(input_data, model=mock_model, model_name="test"))
    assert len(result) == 1
    assert all(h in result[0] for h in hook_points)
    assert torch.allclose(result[0]["h0"], torch.arange(9).reshape(1, 3, 3))
    assert torch.allclose(result[0]["h1"], torch.arange(9, 18).reshape(1, 3, 3))
    assert result[0]["meta"][0]["model_name"] == "test"

    # Test ActivationTransformer
    transformer = ActivationTransformer()
    mock_model.eos_token_id = 1
    mock_model.pad_token_id = 0
    mock_model.bos_token_id = 2

    input_activations = [
        {
            "tokens": torch.tensor([1, 3, 2]),
            "h0": torch.arange(9).reshape(3, 3),
            "h1": torch.arange(9, 18).reshape(3, 3),
        }
    ]
    result = list(transformer.process(input_activations, model=mock_model, ignore_token_ids=[1, 2]))
    assert len(result) == 1
    assert all(h in result[0] for h in hook_points)
    assert torch.allclose(result[0]["h0"], torch.tensor([[3, 4, 5]]))
    assert torch.allclose(result[0]["h1"], torch.tensor([[12, 13, 14]]))

    # Test ActivationTransformer with ignore_token_ids
    transformer_with_ignore = ActivationTransformer()
    input_activations = [
        {
            "tokens": torch.tensor([1, 3, 2]),
            "h0": torch.arange(9).reshape(3, 3),
            "h1": torch.arange(9, 18).reshape(3, 3),
        }
    ]
    result = list(transformer_with_ignore.process(input_activations, ignore_token_ids=[3]))
    assert len(result) == 1
    assert all(h in result[0] for h in hook_points)
    # Now we should only have the activations for tokens 1 and 2, not 3
    assert torch.allclose(result[0]["h0"], torch.tensor([[0, 1, 2], [6, 7, 8]]))
    assert torch.allclose(result[0]["h1"], torch.tensor([[9, 10, 11], [15, 16, 17]]))


def test_activation_buffer(mocker: MockerFixture):
    buffer = ActivationBuffer()

    # Test empty buffer
    assert len(buffer) == 0

    # Test concatenation
    activations = {"h0": torch.arange(9).reshape(3, 3), "h1": torch.arange(9, 18).reshape(3, 3)}
    buffer = buffer.cat(activations)
    assert len(buffer) == 3
    assert torch.allclose(buffer.buffer[0]["h0"], torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    assert torch.allclose(buffer.buffer[0]["h1"], torch.tensor([[9, 10, 11], [12, 13, 14], [15, 16, 17]]))

    # Test batch yielding
    batch, buffer = buffer.yield_batch(2)
    assert len(batch) == 2  # Two hook points
    assert torch.allclose(batch["h0"], torch.tensor([[0, 1, 2], [3, 4, 5]]))
    assert torch.allclose(batch["h1"], torch.tensor([[9, 10, 11], [12, 13, 14]]))
    assert len(buffer) == 1
    assert torch.allclose(buffer.buffer[0]["h0"], torch.tensor([[6, 7, 8]]))
    assert torch.allclose(buffer.buffer[0]["h1"], torch.tensor([[15, 16, 17]]))

    # Test concatenation
    buffer = buffer.cat(activations)
    assert len(buffer) == 4
    assert torch.allclose(buffer.buffer[0]["h0"], torch.tensor([[6, 7, 8]]))
    assert torch.allclose(buffer.buffer[0]["h1"], torch.tensor([[15, 16, 17]]))
    assert torch.allclose(buffer.buffer[1]["h0"], torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    assert torch.allclose(buffer.buffer[1]["h1"], torch.tensor([[9, 10, 11], [12, 13, 14], [15, 16, 17]]))

    # Test batch yielding
    batch, buffer = buffer.yield_batch(2)
    assert len(batch) == 2
    assert torch.allclose(batch["h0"], torch.tensor([[6, 7, 8], [0, 1, 2]]))
    assert torch.allclose(batch["h1"], torch.tensor([[15, 16, 17], [9, 10, 11]]))
    assert len(buffer) == 2
    assert torch.allclose(buffer.buffer[0]["h0"], torch.tensor([[3, 4, 5], [6, 7, 8]]))
    assert torch.allclose(buffer.buffer[0]["h1"], torch.tensor([[12, 13, 14], [15, 16, 17]]))

    # Mock shuffle
    buffer = ActivationBuffer(buffer=[activations])
    mocker.patch("torch.randperm", return_value=torch.tensor([1, 0, 2]))

    # Test shuffle
    buffer = buffer.shuffle()
    assert len(buffer) == 3
    assert torch.allclose(buffer.buffer[0]["h0"], torch.tensor([[3, 4, 5], [0, 1, 2], [6, 7, 8]]))
    assert torch.allclose(buffer.buffer[0]["h1"], torch.tensor([[12, 13, 14], [9, 10, 11], [15, 16, 17]]))

    # Test with various generator seeds
    buffer = ActivationBuffer(buffer=[activations])
    buffer.generator.manual_seed(42)
    shuffled_buffer = buffer.shuffle()
    assert len(shuffled_buffer) == 3


def test_activation_batchler(mocker: MockerFixture):
    # Create input data
    input_data = []
    for i in range(5):  # 5 samples
        input_data.append(
            {
                "h0": torch.tensor([[3, 4, 5]]) + i * 10,
                "h1": torch.tensor([[12, 13, 14]]) + i * 10,
                "tokens": torch.tensor([[1, 2, 3]]),  # Make sure this is a 2D tensor with shape [1, 3]
            }
        )

    # Test non-shuffled batchler
    batchler = ActivationBatchler(batch_size=2)
    result = list(batchler.process(input_data))
    assert len(result) == 3  # Should create 2 full batches and 1 partial batch
    assert "h0" in result[0] and "h1" in result[0]
    assert torch.allclose(result[0]["h0"], torch.tensor([[3, 4, 5], [13, 14, 15]]))
    assert torch.allclose(result[0]["h1"], torch.tensor([[12, 13, 14], [22, 23, 24]]))
    assert torch.allclose(result[1]["h0"], torch.tensor([[23, 24, 25], [33, 34, 35]]))
    assert torch.allclose(result[1]["h1"], torch.tensor([[32, 33, 34], [42, 43, 44]]))
    assert torch.allclose(result[2]["h0"], torch.tensor([[43, 44, 45]]))
    assert torch.allclose(result[2]["h1"], torch.tensor([[52, 53, 54]]))
    assert "tokens" in result[0]  # Tokens are kept
    assert torch.allclose(result[0]["tokens"], torch.tensor([[1, 2, 3], [1, 2, 3]]))

    # Mock shuffle
    mocker.patch("torch.randperm", return_value=torch.tensor([1, 0, 2]))

    # Test shuffled batchler with buffer_size
    batchler = ActivationBatchler(batch_size=2, buffer_size=3)
    # In buffer size 3 and batch size 2, the batchler should first read in 3 samples, shuffle them, and yield a batch of 2,
    # so the first batch will contain the 2rd and 1st samples. Then it should read in the remaining 2 samples, shuffle them,
    # and yield all data in two batches, which respectively contain the 4th and 3rd samples and the 5th sample.

    result = list(batchler.process(input_data))
    assert len(result) == 3  # Should create 2 full batches and 1 partial batch
    assert "h0" in result[0] and "h1" in result[0]
    assert torch.allclose(result[0]["h0"], torch.tensor([[13, 14, 15], [3, 4, 5]]))
    assert torch.allclose(result[0]["h1"], torch.tensor([[22, 23, 24], [12, 13, 14]]))
    assert torch.allclose(result[1]["h0"], torch.tensor([[33, 34, 35], [23, 24, 25]]))
    assert torch.allclose(result[1]["h1"], torch.tensor([[42, 43, 44], [32, 33, 34]]))
    assert torch.allclose(result[2]["h0"], torch.tensor([[43, 44, 45]]))
    assert torch.allclose(result[2]["h1"], torch.tensor([[52, 53, 54]]))
    # Verify tokens are kept and shuffled correctly
    assert "tokens" in result[0]
    assert torch.allclose(result[0]["tokens"], torch.tensor([[1, 2, 3], [1, 2, 3]]))

    # Test with buffer_shuffle_config
    from lm_saes import BufferShuffleConfig

    buffer_shuffle_config = BufferShuffleConfig(enabled=True, method="full", perm_seed=42, generator_device="cpu")

    batchler = ActivationBatchler(batch_size=2, buffer_size=3, buffer_shuffle_config=buffer_shuffle_config)

    # Mock shuffle
    mocker.patch("torch.randperm", return_value=torch.tensor([2, 0, 1]))

    result = list(batchler.process(input_data))
    assert len(result) == 3
    assert "h0" in result[0] and "h1" in result[0]
    assert "tokens" in result[0]  # Tokens should be present
