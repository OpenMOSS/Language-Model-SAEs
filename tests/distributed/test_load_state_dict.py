import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor
from torch.nn import Module

from lm_saes import (
    CLTConfig,
    CrossCoder,
    CrossCoderConfig,
    CrossLayerTranscoder,
    LorsaConfig,
    LowRankSparseAttention,
    SAEConfig,
    SparseAutoEncoder,
)
from lm_saes.testing import distributed_test
from lm_saes.utils.distributed.dimmap import DimMap


def run_load_test(model: Module, device_mesh: DeviceMesh, sharded: bool = True) -> None:
    device = "cuda" if torch.cuda.is_available() and dist.get_backend() == "nccl" else "cpu"
    dim_maps = model.dim_maps()
    state_dict = {}
    ref_values = {}

    for name, param in model.named_parameters():
        ref_tensor = torch.randn(param.shape, dtype=torch.float32)
        ref_values[name] = ref_tensor

        if sharded and name in dim_maps:
            state_dict[name] = dim_maps[name].distribute(ref_tensor.to(device), device_mesh)
        else:
            state_dict[name] = DimMap({}).distribute(ref_tensor.to(device), device_mesh)

    model.load_state_dict(state_dict, strict=False)

    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), f"{name} should be a DTensor"
        full_param = param.full_tensor().cpu()
        assert torch.allclose(full_param, ref_values[name]), f"{name} values mismatch"


@distributed_test(nproc_per_node=2, backend="gloo")
def test_sae_load():
    torch.manual_seed(42)
    world_size = int(dist.get_world_size())
    device = "cuda" if torch.cuda.is_available() and dist.get_backend() == "nccl" else "cpu"
    device_mesh = init_device_mesh(device, (world_size,), mesh_dim_names=("model",))

    sae_cfg = SAEConfig(
        d_model=8,
        expansion_factor=2,
        hook_point_in="layer_0",
        hook_point_out="layer_0",
        device=device,
    )
    sae = SparseAutoEncoder(sae_cfg, device_mesh=device_mesh)
    run_load_test(sae, device_mesh, sharded=True)
    run_load_test(sae, device_mesh, sharded=False)


@distributed_test(nproc_per_node=2, backend="gloo")
def test_crosscoder_load():
    torch.manual_seed(42)
    world_size = int(dist.get_world_size())
    device = "cuda" if torch.cuda.is_available() and dist.get_backend() == "nccl" else "cpu"
    device_mesh = init_device_mesh(device, (world_size,), mesh_dim_names=("model",))

    cc_cfg = CrossCoderConfig(
        d_model=8,
        expansion_factor=2,
        hook_points=["head_0", "head_1"],
        device=device,
    )
    cc = CrossCoder(cc_cfg, device_mesh=device_mesh)
    run_load_test(cc, device_mesh, sharded=True)


@distributed_test(nproc_per_node=2, backend="gloo")
def test_clt_load():
    torch.manual_seed(42)
    world_size = int(dist.get_world_size())
    device = "cuda" if torch.cuda.is_available() and dist.get_backend() == "nccl" else "cpu"
    device_mesh = init_device_mesh(device, (world_size,), mesh_dim_names=("model",))

    clt_cfg = CLTConfig(
        d_model=8,
        expansion_factor=2,
        hook_points_in=["layer_0", "layer_1"],
        hook_points_out=["layer_0", "layer_1"],
        device=device,
    )
    clt = CrossLayerTranscoder(clt_cfg, device_mesh=device_mesh)
    run_load_test(clt, device_mesh, sharded=True)


@distributed_test(nproc_per_node=2, backend="gloo")
def test_lorsa_load():
    torch.manual_seed(42)
    world_size = int(dist.get_world_size())
    device = "cuda" if torch.cuda.is_available() and dist.get_backend() == "nccl" else "cpu"
    device_mesh = init_device_mesh(device, (world_size,), mesh_dim_names=("model",))

    lorsa_cfg = LorsaConfig(
        d_model=8,
        expansion_factor=2,
        hook_point_in="layer_0",
        hook_point_out="layer_1",
        n_qk_heads=2,
        d_qk_head=4,
        rotary_dim=4,
        n_ctx=16,
        device=device,
    )
    lorsa = LowRankSparseAttention(lorsa_cfg, device_mesh=device_mesh)
    run_load_test(lorsa, device_mesh, sharded=True)
