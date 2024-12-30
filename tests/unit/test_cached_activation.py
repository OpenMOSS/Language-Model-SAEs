from pathlib import Path

import pytest
import torch
from pytest_mock import MockerFixture

from lm_saes.activation.processors.cached_activation import (
    CachedActivationLoader,
    ChunkInfo,
)


@pytest.fixture
def sample_activation():
    """Create a sample activation tensor."""
    return torch.randn(2, 3, 4)  # (n_samples, n_context, d_model)


@pytest.fixture
def sample_tokens():
    """Create sample token indices."""
    return torch.randint(0, 1000, (2, 3))  # (n_samples, n_context)


@pytest.fixture
def sample_info():
    """Create sample info list."""
    return [{"context_id": f"ctx_{i}"} for i in range(2)]


def create_fake_pt_file(fs, path: Path, activation, tokens, info):
    """Helper to create a fake .pt file with test data."""
    fs.create_file(
        path,
        contents="",  # Contents don't matter as we'll mock torch.load
    )


def test_chunk_info_from_path():
    """Test ChunkInfo.from_path with different filename formats."""
    # Test sharded format
    sharded = ChunkInfo.from_path(Path("shard-1-chunk-2.pt"))
    assert sharded.shard_id == 1
    assert sharded.chunk_id == 2

    # Test non-sharded format
    non_sharded = ChunkInfo.from_path(Path("chunk-3.pt"))
    assert non_sharded.shard_id == 0
    assert non_sharded.chunk_id == 3

    # Test invalid format
    with pytest.raises(ValueError):
        ChunkInfo.from_path(Path("invalid.pt"))


def test_cached_activation_loader(fs, mocker: MockerFixture, sample_activation, sample_tokens, sample_info):
    """Test CachedActivationLoader with a fake filesystem."""
    # Setup test directory structure
    cache_dir = Path("/cache")
    hook_points = ["hook1", "hook2"]

    for hook in hook_points:
        hook_dir = cache_dir / hook
        fs.create_dir(hook_dir)

        # Create both sharded and non-sharded files
        files = [
            hook_dir / "shard-0-chunk-0.pt",
            hook_dir / "shard-0-chunk-1.pt",
            hook_dir / "chunk-2.pt",
        ]

        for file in files:
            create_fake_pt_file(fs, file, sample_activation, sample_tokens, sample_info)

    # Mock torch.load to return test data
    def mock_torch_load(path, **kwargs):
        return {
            "activation": sample_activation,
            "tokens": sample_tokens,
            "meta": sample_info,
        }

    mocker.patch("torch.load", side_effect=mock_torch_load)

    # Initialize loader and process data
    loader = CachedActivationLoader(cache_dir, hook_points)
    results = list(loader.process())

    # Verify results
    assert len(results) == 6  # 3 chunks * 2 samples per chunk

    for i, result in enumerate(results):
        # Check if all hook points are present
        for hook in hook_points:
            assert hook in result
            assert torch.equal(result[hook], sample_activation[i % 2])

        # Check tokens and info
        assert torch.equal(result["tokens"], sample_tokens[i % 2])
        assert result["meta"] == sample_info[i % 2]


def test_cached_activation_loader_missing_dir(fs):
    """Test CachedActivationLoader with missing directory."""
    with pytest.raises(FileNotFoundError):
        loader = CachedActivationLoader("/nonexistent", ["hook1"])
        list(loader.process())


def test_cached_activation_loader_mismatched_chunks(
    fs, mocker: MockerFixture, sample_activation, sample_tokens, sample_info
):
    """Test CachedActivationLoader with mismatched chunk counts."""
    # Setup directories with different numbers of chunks
    cache_dir = Path("/cache")

    # hook1 has 2 chunks
    hook1_dir = cache_dir / "hook1"
    fs.create_dir(hook1_dir)
    create_fake_pt_file(fs, hook1_dir / "chunk-0.pt", sample_activation, sample_tokens, sample_info)
    create_fake_pt_file(fs, hook1_dir / "chunk-1.pt", sample_activation, sample_tokens, sample_info)

    # hook2 has 1 chunk
    hook2_dir = cache_dir / "hook2"
    fs.create_dir(hook2_dir)
    create_fake_pt_file(fs, hook2_dir / "chunk-0.pt", sample_activation, sample_tokens, sample_info)

    # Mock torch.load
    mocker.patch(
        "torch.load",
        return_value={
            "activation": sample_activation,
            "tokens": sample_tokens,
            "meta": sample_info,
        },
    )

    # Should raise ValueError due to mismatched chunk counts
    with pytest.raises(
        ValueError,
        match="Hook points have different numbers of chunks: {'hook1': 2, 'hook2': 1}. All hook points must have the same number of chunks.",
    ):
        loader = CachedActivationLoader(cache_dir, ["hook1", "hook2"])
        list(loader.process())


def test_cached_activation_loader_invalid_data(fs, mocker: MockerFixture):
    """Test CachedActivationLoader with invalid data format."""
    cache_dir = Path("/cache")
    hook_dir = cache_dir / "hook1"
    fs.create_dir(hook_dir)
    create_fake_pt_file(fs, hook_dir / "chunk-0.pt", None, None, None)

    # Mock torch.load to return invalid data
    mocker.patch("torch.load", return_value={"invalid": "data"})

    loader = CachedActivationLoader(cache_dir, ["hook1"])
    with pytest.raises(
        AssertionError,
        match="Loading cached activation /cache/hook1/chunk-0.pt error: missing 'activation' field",
    ):
        list(loader.process())
