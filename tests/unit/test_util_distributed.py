import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from lm_saes.utils.misc import get_mesh_dim_size, get_mesh_rank


def setup_distributed():
    """Initialize distributed training with torchrun."""
    # torchrun sets these environment variables automatically
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Initialize the process group
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    return device, rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def test_get_mesh_dim_size_none():
    """Test get_mesh_dim_size with None device mesh."""
    result = get_mesh_dim_size(None, "data")
    assert result == 1, f"Expected 1 for None device mesh, got {result}"


def test_get_mesh_rank_none():
    """Test get_mesh_rank with None device mesh."""
    result = get_mesh_rank(None)
    assert result == 0, f"Expected 0 for None device mesh, got {result}"


def test_get_mesh_dim_size_and_rank_2d_distributed():
    """Test get_mesh_dim_size and get_mesh_rank with 2D device mesh."""
    device, rank, world_size = setup_distributed()
    print(f"Running 2D test on device {device}, rank {rank}, world size {world_size}")
    assert world_size == 8, "This test requires 8 processes"

    # Test with 2D mesh (2x1)
    device_mesh = init_device_mesh(
        device_type="cuda" if torch.cuda.is_available() else "cpu", mesh_shape=[2, 2, 2], mesh_dim_names=["a", "b", "c"]
    )

    # Test get_mesh_dim_size for both dimensions
    a_dim_size = get_mesh_dim_size(device_mesh, "a")
    b_dim_size = get_mesh_dim_size(device_mesh, "b")
    c_dim_size = get_mesh_dim_size(device_mesh, "c")

    assert a_dim_size == 2, f"Expected a dimension size 2, got {a_dim_size}"
    assert b_dim_size == 2, f"Expected b dimension size 2, got {b_dim_size}"
    assert c_dim_size == 2, f"Expected c dimension size 2, got {c_dim_size}"

    # Test get_mesh_rank
    mesh_rank = get_mesh_rank(device_mesh)
    assert mesh_rank == rank, f"Expected mesh rank {rank}, got {mesh_rank}"

    sub_mesh = device_mesh["b", "c"]
    sub_mesh_rank = get_mesh_rank(sub_mesh)
    assert sub_mesh_rank == rank % 4, f"Expected sub mesh rank {rank % 4}, got {sub_mesh_rank}"
    b_dim_size = get_mesh_dim_size(sub_mesh, "b")
    assert b_dim_size == 2, f"Expected b dimension size 2, got {b_dim_size}"
    c_dim_size = get_mesh_dim_size(sub_mesh, "c")
    assert c_dim_size == 2, f"Expected c dimension size 2, got {c_dim_size}"

    sub_mesh = device_mesh["a", "c"]
    sub_mesh_rank = get_mesh_rank(sub_mesh)
    if rank in [0, 1, 4, 5]:
        assert sub_mesh_rank == [0, 1, 4, 5].index(rank), (
            f"Expected sub mesh rank {[0, 1, 4, 5].index(rank)}, got {sub_mesh_rank}"
        )
    else:
        assert sub_mesh_rank == [2, 3, 6, 7].index(rank), (
            f"Expected sub mesh rank {[2, 3, 6, 7].index(rank)}, got {sub_mesh_rank}"
        )
    a_dim_size = get_mesh_dim_size(sub_mesh, "a")
    assert a_dim_size == 2, f"Expected a dimension size 2, got {a_dim_size}"
    c_dim_size = get_mesh_dim_size(sub_mesh, "c")
    assert c_dim_size == 2, f"Expected c dimension size 2, got {c_dim_size}"

    cleanup_distributed()


if __name__ == "__main__":
    # Test non-distributed functions first
    test_get_mesh_dim_size_none()
    test_get_mesh_rank_none()

    # Test distributed functions
    test_get_mesh_dim_size_and_rank_2d_distributed()
