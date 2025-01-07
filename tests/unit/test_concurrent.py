import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from lm_saes.utils.concurrent import BackgroundGenerator


def test_basic_iteration():
    """Test basic iteration through a simple list."""
    data = [1, 2, 3, 4, 5]
    bg = BackgroundGenerator(data)
    assert list(bg) == data


def test_empty_generator():
    """Test behavior with an empty generator."""
    bg = BackgroundGenerator([])
    with pytest.raises(StopIteration):
        next(bg)


def test_max_prefetch():
    """Test that max_prefetch limits the queue size."""

    def slow_generator():
        for i in range(3):
            time.sleep(0.1)  # Simulate slow generation
            yield i

    bg = BackgroundGenerator(slow_generator(), max_prefetch=1)
    assert bg.queue.maxsize == 1

    # Consume all items
    assert list(bg) == [0, 1, 2]


def test_exception_propagation():
    """Test that exceptions from the generator are properly propagated."""

    def failing_generator():
        yield 1
        raise ValueError("Test error")
        yield 2  # This will never be reached

    bg = BackgroundGenerator(failing_generator())

    # First item should be retrieved normally
    assert next(bg) == 1

    # Next call should raise the ValueError
    with pytest.raises(ValueError, match="Test error"):
        next(bg)


def test_custom_executor():
    """Test using a custom executor."""
    data = [1, 2, 3]
    with ThreadPoolExecutor(max_workers=2) as executor:
        bg = BackgroundGenerator(data, executor=executor)
        assert list(bg) == data


def test_generator_cleanup(mocker):
    """Test proper cleanup of resources."""
    # Mock ThreadPoolExecutor
    mock_executor = mocker.Mock(spec=ThreadPoolExecutor)
    mock_executor.submit.return_value = mocker.Mock()

    data = [1, 2, 3]
    bg = BackgroundGenerator(data, executor=mock_executor)

    # Simulate deletion
    bg.__del__()

    # Verify that shutdown wasn't called since it's not an owned executor
    mock_executor.shutdown.assert_not_called()


def test_large_dataset():
    """Test handling of a larger dataset."""
    large_data = range(1000)
    bg = BackgroundGenerator(large_data, max_prefetch=10)
    assert list(bg) == list(large_data)


def test_multiple_iterations():
    """Test that the generator can only be consumed once."""
    data = [1, 2, 3]
    bg = BackgroundGenerator(data)

    # First iteration should work
    assert list(bg) == data

    # Second iteration should yield no items
    assert list(bg) == []
