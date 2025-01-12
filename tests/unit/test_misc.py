import math
from typing import Iterator

import pytest
import torch

from lm_saes.utils.misc import calculate_activation_norm


class TestCalculateActivationNorm:
    @pytest.fixture
    def mock_activation_stream(self) -> Iterator[dict[str, torch.Tensor]]:
        """Creates a mock activation stream with known values for testing."""

        def stream_generator():
            # Create 10 batches of activations
            for _ in range(10):
                yield {
                    "layer1": torch.ones(4, 16),  # norm will be sqrt(16)
                    "layer2": torch.ones(4, 16) * 2,  # norm will be sqrt(16) * 2
                }

        return stream_generator()

    def test_basic_functionality(self, mock_activation_stream):
        """Test basic functionality with default batch_num."""
        result = calculate_activation_norm(mock_activation_stream)

        assert isinstance(result, dict)
        assert "layer1" in result
        assert "layer2" in result

        # Expected norms:
        # layer1: sqrt(16) ≈ 4
        # layer2: sqrt(16) * 2 ≈ 8.0
        assert pytest.approx(result["layer1"], rel=1e-4) == 4.0
        assert pytest.approx(result["layer2"], rel=1e-4) == 8.0

    def test_custom_batch_num(self, mock_activation_stream):
        """Test with custom batch_num parameter."""
        result = calculate_activation_norm(mock_activation_stream, batch_num=3)

        # Should still give same results as we're averaging
        assert pytest.approx(result["layer1"], rel=1e-4) == 4.0
        assert pytest.approx(result["layer2"], rel=1e-4) == 8.0

    def test_empty_stream(self):
        """Test behavior with empty activation stream."""
        empty_stream = iter([])
        with pytest.raises(StopIteration):
            calculate_activation_norm(empty_stream)

    def test_single_batch(self):
        """Test with a single batch of activations."""

        def single_batch_stream():
            yield {"single": torch.ones(2, 4)}  # norm will be 2.0

        result = calculate_activation_norm(single_batch_stream(), batch_num=1)
        assert pytest.approx(result["single"], rel=1e-4) == 2.0

    def test_zero_tensors(self):
        """Test with zero tensors."""

        def zero_stream():
            for _ in range(10):
                yield {"zeros": torch.zeros(2, 4)}

        result = calculate_activation_norm(zero_stream())
        assert result["zeros"] == 0.0

    def test_mixed_values(self):
        """Test with mixed positive/negative values."""

        def mixed_stream():
            for i in range(10):
                yield {"mixed": torch.tensor([[1.0, -2.0], [3.0, -4.0], [3.0, -2.0], [9.0, -4.0]]) * (i + 1)}

        result = calculate_activation_norm(mixed_stream(), batch_num=10)
        assert (
            pytest.approx(result["mixed"], rel=1e-4) == ((math.sqrt(5) + 5 + math.sqrt(13) + math.sqrt(97)) / 4) * 5.5
        )
