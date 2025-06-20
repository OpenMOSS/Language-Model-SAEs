"""
Tests for Cross Layer Transcoder (CLT) in distributed settings.
"""
import os
import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from lm_saes import CrossLayerTranscoder, CLTConfig


def setup_distributed():
    """Initialize distributed training with torchrun."""
    # torchrun sets these environment variables automatically
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
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


def create_test_config(device: str) -> CLTConfig:
    """Create a simple test configuration."""
    return CLTConfig(
        sae_type="clt",
        d_model=4,
        expansion_factor=2,  # d_sae = 8
        n_layers=2,
        hook_points_in=["layer_0_in", "layer_1_in"],
        hook_points_out=["layer_0_out", "layer_1_out"],
        use_decoder_bias=True,
        act_fn="relu",
        apply_decoder_bias_to_pre_encoder=False,
        norm_activation="inference",
        sparsity_include_decoder_norm=True,
        force_unit_decoder_norm=False,
        device=device,
        dtype=torch.float32,
    )


def create_simple_batch(device: str) -> dict[str, torch.Tensor]:
    """Create simple test data."""
    batch_size = 2
    seq_len = 3
    d_model = 4
    
    batch = {
        "layer_0_in": torch.ones(batch_size, seq_len, d_model, device=device),
        "layer_1_in": torch.zeros(batch_size, seq_len, d_model, device=device),
        "layer_0_out": torch.ones(batch_size, seq_len, d_model, device=device),
        "layer_1_out": torch.zeros(batch_size, seq_len, d_model, device=device),
    }
    return batch


def test_distributed_clt_tensor_parallel():
    """Test CLT model with tensor parallel sharding across devices."""
    device, rank, world_size = setup_distributed()
    
    try:
        # Create device mesh for tensor parallel
        device_list = list(range(world_size))
        if torch.cuda.is_available():
            device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tp",))
        else:
            device_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
        
        # Create config and model with device mesh
        config = create_test_config(device)
        clt_model = CrossLayerTranscoder(config, device_mesh=device_mesh)
        
        # Initialize parameters to specific values using distributed tensors
        with torch.no_grad():
            # For distributed tensors, we need to work with local tensors and then redistribute
            # Initialize encoders - these are sharded along d_sae dimension
            if hasattr(clt_model.W_E, 'device_mesh'):
                # Convert to local, modify, then redistribute
                W_E_local = clt_model.W_E.to_local()
                W_E_local[0, :, :] = 0.1
                W_E_local[1, :, :] = -0.2
                clt_model.W_E.data = torch.distributed.tensor.DTensor.from_local(
                    W_E_local, device_mesh=device_mesh, placements=clt_model.W_E.placements
                )
                
                b_E_local = clt_model.b_E.to_local()
                b_E_local[0, :] = 0.0
                b_E_local[1, :] = -0.1
                clt_model.b_E.data = torch.distributed.tensor.DTensor.from_local(
                    b_E_local, device_mesh=device_mesh, placements=clt_model.b_E.placements
                )
                
                # Initialize decoders and their biases - W_D sharded along d_sae, b_D replicated
                # Set decoder weights for each layer group
                # W_D[0] contains decoder for (0,0): layer 0 to layer 0
                W_D_0_local = clt_model.W_D[0].to_local()
                W_D_0_local[0, :, :] = 0.3
                clt_model.W_D[0].data = torch.distributed.tensor.DTensor.from_local(
                    W_D_0_local, device_mesh=device_mesh, placements=clt_model.W_D[0].placements
                )
                
                # W_D[1] contains decoders for (0,1) and (1,1): layer 0,1 to layer 1
                W_D_1_local = clt_model.W_D[1].to_local()
                W_D_1_local[0, :, :] = -0.4  # Decoder (0,1): layer 0 to layer 1
                W_D_1_local[1, :, :] = 0.5   # Decoder (1,1): layer 1 to layer 1
                clt_model.W_D[1].data = torch.distributed.tensor.DTensor.from_local(
                    W_D_1_local, device_mesh=device_mesh, placements=clt_model.W_D[1].placements
                )
                
                b_D_local = clt_model.b_D.to_local()
                b_D_local[0, :] = -0.2  # Decoder 0: (0,0) - layer 0 to layer 0
                b_D_local[1, :] = -0.3  # Decoder 1: (0,1) - layer 0 to layer 1
                b_D_local[2, :] = -0.4  # Decoder 2: (1,1) - layer 1 to layer 1
                clt_model.b_D.data = torch.distributed.tensor.DTensor.from_local(
                    b_D_local, device_mesh=device_mesh, placements=clt_model.b_D.placements
                )
            else:
                # Fallback for non-distributed case
                clt_model.W_E.data[0, :, :] = 0.1
                clt_model.b_E.data[0, :] = 0.0
                clt_model.W_E.data[1, :, :] = -0.2
                clt_model.b_E.data[1, :] = -0.1
                clt_model.W_D.data[0, :, :] = 0.3
                clt_model.b_D.data[0, :] = -0.2
                clt_model.W_D.data[1, :, :] = -0.4
                clt_model.b_D.data[1, :] = -0.3
                clt_model.W_D.data[2, :, :] = 0.5
                clt_model.b_D.data[2, :] = -0.4
        
        # Wait for all processes to synchronize
        dist.barrier()
        
        # Test forward pass with tensor parallel model
        simple_batch = create_simple_batch(device)
        input_tensor, _ = clt_model.prepare_input(simple_batch)
        
        # Forward pass
        output = clt_model(input_tensor)
        
        # Verify output shape and values
        assert output.shape == (2, 3, 2, 4)
        # Create expected output with correct shape (batch=2, seq_len=3, n_layers=2, d_model=4)
        expected_output = torch.zeros(2, 3, 2, 4, device=device)
        expected_output[:, :, 0, :] = 0.76   # Layer 0 output
        expected_output[:, :, 1, :] = -1.98  # Layer 1 output
        
        assert torch.allclose(output, expected_output, atol=1e-6)
        
        # Test loss computation
        loss = clt_model.compute_loss(simple_batch, return_aux_data=False)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar loss
        
        # Test backward pass
        loss.backward()
        
        # Check that gradients are computed for distributed tensors
        for name, param in clt_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
        
        # Test parameter sharding - verify that parameters are actually distributed
        if hasattr(clt_model.W_E, 'device_mesh'):
            print(f"Rank {rank}: W_E is distributed tensor with shape {clt_model.W_E.shape}")
            print(f"Rank {rank}: W_E local shape {clt_model.W_E.to_local().shape}")
        if hasattr(clt_model.W_D, 'device_mesh'):
            print(f"Rank {rank}: W_D is distributed tensor with shape {clt_model.W_D.shape}")
            print(f"Rank {rank}: W_D local shape {clt_model.W_D.to_local().shape}")
        
        print(f"Rank {rank}: All tensor parallel tests passed!")
        
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    # Run tests directly - torchrun will handle process spawning
    print("Running tensor parallel CLT test...")
    test_distributed_clt_tensor_parallel()
