import json
import os
import tempfile

import pytest
import safetensors.torch as safe
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from lm_saes import SAEConfig, SparseAutoEncoder
from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.testing import distributed_test


@pytest.fixture
def temp_sae_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a dummy config
        config = {
            "sae_type": "sae",
            "d_model": 8,
            "expansion_factor": 2,
            "hook_point_in": "layer_0",
            "hook_point_out": "layer_0",
            "device": "cpu",
            "dtype": "float32",
        }
        with open(os.path.join(tmpdirname, "config.json"), "w") as f:
            json.dump(config, f)

        # Create dummy weights
        sae_cfg = SAEConfig(**config)
        sae = SparseAutoEncoder(sae_cfg)

        # Use exact parameters to verify loading correctness
        with torch.no_grad():
            sae.W_E.fill_(1.0)
            sae.b_E.fill_(2.0)
            sae.W_D.fill_(3.0)
            sae.b_D.fill_(4.0)

        # Save as safetensors
        safe.save_file(sae.state_dict(), os.path.join(tmpdirname, "sae_weights.safetensors"))

        # Save as pt
        torch.save({"sae": sae.state_dict()}, os.path.join(tmpdirname, "sae_weights.pt"))

        yield tmpdirname


def test_from_local_single_device(temp_sae_dir):
    # Test safetensors loading
    model = AbstractSparseAutoEncoder.from_local(temp_sae_dir)
    assert isinstance(model, SparseAutoEncoder)

    # Verify loaded parameters
    assert torch.allclose(model.W_E, torch.ones_like(model.W_E))
    assert torch.allclose(model.b_E, torch.full_like(model.b_E, 2.0))
    assert torch.allclose(model.W_D, torch.full_like(model.W_D, 3.0))
    assert torch.allclose(model.b_D, torch.full_like(model.b_D, 4.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@distributed_test(nproc_per_node=2, backend="nccl")
def test_from_local_distributed(temp_sae_dir):
    # This test receives temp_sae_dir from the parent process via distributed_test wrapper.
    # Note: temp_sae_dir must be picklable. Path string is picklable.

    world_size = int(dist.get_world_size())
    device = "cuda" if torch.cuda.is_available() and dist.get_backend() == "nccl" else "cpu"
    device_mesh = init_device_mesh(device, (world_size,), mesh_dim_names=("model",))

    # Check if we can access the directory
    assert os.path.exists(temp_sae_dir)
    assert os.path.exists(os.path.join(temp_sae_dir, "config.json"))

    model = AbstractSparseAutoEncoder.from_local(temp_sae_dir, device_mesh=device_mesh)
    assert isinstance(model, SparseAutoEncoder)

    # Verify loaded parameters and sharding
    assert torch.allclose(model.W_E.full_tensor(), torch.ones(model.W_E.shape, device=model.W_E.device))
    assert torch.allclose(model.b_E.full_tensor(), torch.full(model.b_E.shape, 2.0, device=model.b_E.device))
    assert torch.allclose(model.W_D.full_tensor(), torch.full(model.W_D.shape, 3.0, device=model.W_D.device))
    assert torch.allclose(model.b_D.full_tensor(), torch.full(model.b_D.shape, 4.0, device=model.b_D.device))
