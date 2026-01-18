import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from lm_saes.activation.processors.activation import ActivationBuffer
from lm_saes.utils.distributed import DimMap

pytest.skip("Skipping distributed tests", allow_module_level=True)


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


def test_activation_buffer_distributed():
    device, rank, world_size = setup_distributed()
    print(f"Running test on device {device}, rank {rank}, world size {world_size}")
    assert world_size == 2, "This test requires 2 processes"

    # Initialize device mesh
    device_mesh = init_device_mesh(
        device_type="cuda" if torch.cuda.is_available() else "cpu", mesh_shape=[world_size], mesh_dim_names=["data"]
    )

    # Initialize activation buffer
    buffer = ActivationBuffer(device_mesh=device_mesh)

    a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]], device=device)
    a = DimMap({"data": 0}).distribute(a, device_mesh)
    buffer = buffer.cat({"a": a})
    batch, buffer = buffer.yield_batch(2)

    if rank == 0:
        assert isinstance(batch["a"], DTensor) and torch.allclose(
            batch["a"].to_local(), torch.tensor([[1, 2, 3]], device=device)
        )
    else:
        assert isinstance(batch["a"], DTensor) and torch.allclose(
            batch["a"].to_local(), torch.tensor([[10, 11, 12]], device=device)
        )

    a = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
    a = DimMap({"data": 0}).distribute(a, device_mesh)
    buffer = buffer.cat({"a": a})

    remaining_buffer = buffer.consume()
    if rank == 0:
        assert isinstance(remaining_buffer["a"], DTensor) and torch.allclose(
            remaining_buffer["a"].to_local(), torch.tensor([[4, 5, 6], [7, 8, 9], [1, 2, 3]], device=device)
        )
    else:
        assert isinstance(remaining_buffer["a"], DTensor) and torch.allclose(
            remaining_buffer["a"].to_local(), torch.tensor([[13, 14, 15], [16, 17, 18], [4, 5, 6]], device=device)
        )

    cleanup_distributed()


if __name__ == "__main__":
    test_activation_buffer_distributed()
