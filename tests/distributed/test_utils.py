import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from lm_saes.testing import distributed_test
from lm_saes.utils.distributed.utils import execute_and_broadcast


def get_rank_specific_value(val):
    return f"rank_{dist.get_rank()}_{val}"


@distributed_test(nproc_per_node=2, backend="gloo")
def test_execute_and_broadcast():
    world_size = int(dist.get_world_size())
    device = "cuda" if torch.cuda.is_available() and dist.get_backend() == "nccl" else "cpu"
    # Create a simple 1D device mesh
    device_mesh = init_device_mesh(device, (world_size,), mesh_dim_names=("model",))

    val = "test"

    # Case 1: device_mesh is None
    # Function executes on all ranks independently
    res = execute_and_broadcast(get_rank_specific_value, None, val=val)
    assert res == f"rank_{dist.get_rank()}_{val}"

    # Case 2: Broadcast from primary rank (dim_name="sweep", not in mesh)
    # "sweep" is not in ("model",), so is_primary_rank checks if ALL coords are 0.
    # Only Rank 0 is primary.
    # Group covers all ranks.
    # Rank 0 executes and broadcasts to Rank 1.
    res = execute_and_broadcast(get_rank_specific_value, device_mesh, mesh_dim_name="sweep", val=val)
    # Everyone receives Rank 0's result
    assert res == f"rank_0_{val}"

    # Case 3: Execute in parallel groups (dim_name="model", in mesh)
    # "model" is in mesh. is_primary_rank excludes "model" coord from check.
    # Remaining coords: []. all([]) is True. Everyone is primary.
    # Group is created excluding "model" dim.
    # Since "model" is the only dim, group definition dims are [].
    # This means groups of size 1 (each rank is its own group).
    # Everyone executes and broadcasts to themselves.
    res = execute_and_broadcast(get_rank_specific_value, device_mesh, mesh_dim_name="model", val=val)
    assert res == f"rank_{dist.get_rank()}_{val}"
