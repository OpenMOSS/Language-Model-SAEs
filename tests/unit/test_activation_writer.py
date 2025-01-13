import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from lm_saes.activation.writer import ActivationWriter
from lm_saes.config import ActivationWriterConfig


@pytest.fixture
def mock_config() -> ActivationWriterConfig:
    return ActivationWriterConfig(cache_dir="test_cache", hook_points=["h0", "h1"], n_samples_per_chunk=2, format="pt")


@pytest.fixture
def writer(mock_config: ActivationWriterConfig, tmp_path: Path) -> ActivationWriter:
    mock_config.cache_dir = str(tmp_path)
    return ActivationWriter(mock_config)


@pytest.fixture
def parallel_writer(mock_config: ActivationWriterConfig, tmp_path: Path) -> ActivationWriter:
    """Create a writer configured for parallel writing."""
    mock_config.cache_dir = str(tmp_path)
    mock_config.num_workers = 2
    return ActivationWriter(mock_config)


@pytest.fixture
def sample_data() -> list[dict]:
    return [
        {
            "tokens": torch.tensor([1, 2, 3]),
            "meta": {"context_idx": 1},
            "h0": torch.ones(3, 4),
            "h1": torch.zeros(3, 4),
        },
        {
            "tokens": torch.tensor([4, 5, 6]),
            "meta": {"context_idx": 2},
            "h0": torch.ones(3, 4) * 2,
            "h1": torch.zeros(3, 4) * 2,
        },
        {
            "tokens": torch.tensor([7, 8, 9]),
            "meta": {"context_idx": 3},
            "h0": torch.ones(3, 4) * 3,
            "h1": torch.zeros(3, 4) * 3,
        },
    ]


def test_initialization(writer: ActivationWriter, tmp_path: Path):
    """Test that directories are created correctly during initialization."""
    for hook_point in writer.cfg.hook_points:
        assert (tmp_path / hook_point).exists()
        assert (tmp_path / hook_point).is_dir()


def test_process_pytorch_format(writer: ActivationWriter, sample_data: list[dict], tmp_path: Path):
    """Test processing data and saving in PyTorch format."""
    writer.process(sample_data)

    # Should create 2 chunks (2 samples in first, 1 in second)
    for hook_point in writer.cfg.hook_points:
        chunk0_path = tmp_path / hook_point / f"chunk-{0:08d}.pt"
        chunk1_path = tmp_path / hook_point / f"chunk-{1:08d}.pt"

        assert chunk0_path.exists()
        assert chunk1_path.exists()

        # Verify content of first chunk
        chunk0_data = torch.load(chunk0_path, weights_only=True)
        assert isinstance(chunk0_data, dict)
        assert "activation" in chunk0_data
        assert "tokens" in chunk0_data
        assert "meta" in chunk0_data
        assert len(chunk0_data["meta"]) == 2
        assert chunk0_data["activation"].shape == (2, 3, 4)
        assert chunk0_data["tokens"].shape == (2, 3)


def test_process_safetensors_format(mock_config: ActivationWriterConfig, sample_data: list[dict], tmp_path: Path):
    """Test processing data and saving in safetensors format."""
    mock_config.format = "safetensors"
    mock_config.cache_dir = str(tmp_path)
    writer = ActivationWriter(mock_config)

    writer.process(sample_data)

    # Verify first chunk of first hook point
    chunk0_path = tmp_path / "h0" / f"chunk-{0:08d}.safetensors"
    assert chunk0_path.exists()

    # Verify meta file exists
    meta_path = chunk0_path.with_suffix(".meta.json")
    assert meta_path.exists()

    chunk0_data = load_file(chunk0_path)
    assert "activation" in chunk0_data
    assert "tokens" in chunk0_data

    chunk0_meta = json.load(open(meta_path, "r"))
    assert len(chunk0_meta) == 2
    assert chunk0_meta[0]["context_idx"] == 1
    assert chunk0_meta[1]["context_idx"] == 2


def test_no_batching(writer: ActivationWriter, sample_data: list[dict], tmp_path: Path):
    """Test processing data without batching."""
    writer.cfg.n_samples_per_chunk = None
    writer.process(sample_data)

    for hook_point in writer.cfg.hook_points:
        for i, sample in enumerate(sample_data):
            chunk_path = tmp_path / hook_point / f"chunk-{i:08d}.pt"
            assert chunk_path.exists()

            chunk_data = torch.load(chunk_path, weights_only=True)
            assert isinstance(chunk_data, dict)
            assert "activation" in chunk_data
            assert "tokens" in chunk_data
            assert "meta" in chunk_data
            assert chunk_data["meta"] == sample["meta"]
            assert chunk_data["activation"].shape == (3, 4)
            assert chunk_data["tokens"].shape == (3,)


def test_invalid_format(mock_config: ActivationWriterConfig, sample_data: list[dict], tmp_path: Path):
    """Test that invalid format raises ValueError."""
    mock_config.format = "invalid"
    mock_config.cache_dir = str(tmp_path)
    writer = ActivationWriter(mock_config)

    with pytest.raises(ValueError, match="Invalid format: invalid"):
        writer.process(sample_data)


def test_missing_required_fields(writer: ActivationWriter):
    """Test that missing required fields raises AssertionError."""
    invalid_data = [{"tokens": torch.tensor([1, 2, 3])}]  # Missing meta and hook points

    with pytest.raises(AssertionError):
        writer.process(invalid_data)


def test_parallel_process(parallel_writer: ActivationWriter, sample_data: list[dict], tmp_path: Path):
    """Test processing data in parallel mode."""
    parallel_writer.process(sample_data)

    # Should create 2 chunks (2 samples in first, 1 in second)
    for hook_point in parallel_writer.cfg.hook_points:
        chunk0_path = tmp_path / hook_point / f"chunk-{0:08d}.pt"
        chunk1_path = tmp_path / hook_point / f"chunk-{1:08d}.pt"

        assert chunk0_path.exists()
        assert chunk1_path.exists()

        # Verify content of first chunk
        chunk0_data = torch.load(chunk0_path, weights_only=True)
        assert isinstance(chunk0_data, dict)
        assert "activation" in chunk0_data
        assert "tokens" in chunk0_data
        assert "meta" in chunk0_data
        assert len(chunk0_data["meta"]) == 2
        assert torch.allclose(
            chunk0_data["activation"], torch.stack([sample_data[0][hook_point], sample_data[1][hook_point]])
        )
        assert chunk0_data["tokens"].shape == (2, 3)

        chunk1_data = torch.load(chunk1_path, weights_only=True)
        assert isinstance(chunk1_data, dict)
        assert "activation" in chunk1_data
        assert "tokens" in chunk1_data
        assert "meta" in chunk1_data
        assert len(chunk1_data["meta"]) == 1
        assert torch.allclose(chunk1_data["activation"], sample_data[2][hook_point])
        assert chunk1_data["tokens"].shape == (1, 3)


def test_custom_executor(mock_config: ActivationWriterConfig, sample_data: list[dict], tmp_path: Path):
    """Test using a custom executor."""
    mock_config.cache_dir = str(tmp_path)
    mock_config.num_workers = 2

    with ThreadPoolExecutor(max_workers=3) as executor:
        writer = ActivationWriter(mock_config, executor=executor)
        writer.process(sample_data)

    # Verify files were written correctly
    for hook_point in mock_config.hook_points:
        chunk0_path = tmp_path / hook_point / f"chunk-{0:08d}.pt"
        assert chunk0_path.exists()


def test_parallel_safetensors_format(mock_config: ActivationWriterConfig, sample_data: list[dict], tmp_path: Path):
    """Test parallel processing with safetensors format."""
    mock_config.format = "safetensors"
    mock_config.cache_dir = str(tmp_path)
    mock_config.num_workers = 2
    writer = ActivationWriter(mock_config)

    writer.process(sample_data)

    # Verify first chunk of first hook point
    chunk0_path = tmp_path / "h0" / f"chunk-{0:08d}.safetensors"
    assert chunk0_path.exists()

    # Verify meta file exists
    meta_path = chunk0_path.with_suffix(".meta.json")
    assert meta_path.exists()

    chunk0_data = load_file(chunk0_path)
    assert "activation" in chunk0_data
    assert "tokens" in chunk0_data

    chunk0_meta = json.load(open(meta_path, "r"))
    assert len(chunk0_meta) == 2
    assert chunk0_meta[0]["context_idx"] == 1
    assert chunk0_meta[1]["context_idx"] == 2


def test_parallel_writer_cleanup(mock_config: ActivationWriterConfig, tmp_path: Path):
    """Test that the executor is properly cleaned up when writer owns it."""
    mock_config.cache_dir = str(tmp_path)
    mock_config.num_workers = 2
    writer = ActivationWriter(mock_config)

    assert writer._owned_executor
    assert not writer.executor._shutdown

    # Trigger cleanup
    del writer
