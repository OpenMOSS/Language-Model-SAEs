import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from llamascopium.testing import distributed_test
from llamascopium.utils.distributed.ops import nonzero


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@distributed_test(nproc_per_node=2, backend="nccl")
def test_nonzero_dtensor_matches_torch_nonzero():
    world_size = int(dist.get_world_size())
    device = "cuda" if torch.cuda.is_available() and dist.get_backend() == "nccl" else "cpu"
    device_mesh = init_device_mesh(device, (world_size,), mesh_dim_names=("model",))

    x = torch.tensor(
        [
            [0, 1, 0],
            [2, 0, 3],
            [0, 0, 0],
            [4, 5, 0],
        ],
        device=device,
    )
    expected = torch.nonzero(x, as_tuple=False)
    x_dtensor = distribute_tensor(x, device_mesh, [Shard(0)])

    result = nonzero(x_dtensor)

    assert isinstance(result, DTensor)
    assert all(isinstance(placement, Replicate) for placement in result.placements)
    assert result.shape == expected.shape
    assert torch.equal(result.to_local().cpu(), expected.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@distributed_test(nproc_per_node=2, backend="nccl")
def test_nonzero_dtensor_handles_empty_local_results():
    world_size = int(dist.get_world_size())
    device = "cuda" if torch.cuda.is_available() and dist.get_backend() == "nccl" else "cpu"
    device_mesh = init_device_mesh(device, (world_size,), mesh_dim_names=("model",))

    x = torch.tensor(
        [
            [0, 1, 0],
            [2, 0, 3],
            [0, 0, 0],
            [0, 0, 0],
        ],
        device=device,
    )
    expected = torch.nonzero(x, as_tuple=False)
    x_dtensor = distribute_tensor(x, device_mesh, [Shard(0)])

    result = nonzero(x_dtensor)

    assert isinstance(result, DTensor)
    assert all(isinstance(placement, Replicate) for placement in result.placements)
    assert torch.equal(result.to_local().cpu(), expected.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@distributed_test(nproc_per_node=2, backend="nccl")
def test_nonzero_dtensor_rejects_as_tuple():
    world_size = int(dist.get_world_size())
    device = "cuda" if torch.cuda.is_available() and dist.get_backend() == "nccl" else "cpu"
    device_mesh = init_device_mesh(device, (world_size,), mesh_dim_names=("model",))

    x = torch.tensor(
        [
            [0, 1],
            [2, 0],
        ],
        device=device,
    )
    x_dtensor = distribute_tensor(x, device_mesh, [Shard(0)])

    with pytest.raises(AssertionError, match="as_tuple is not supported for DTensor"):
        nonzero(x_dtensor, as_tuple=True)
