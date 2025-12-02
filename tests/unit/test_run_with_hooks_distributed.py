"""
Tests for run_with_hooks with DTensor support.

Run with: torchrun --nproc_per_node=2 tests/unit/test_run_with_hooks_distributed.py
"""

import os
from typing import Callable, Union, cast

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor

from lm_saes.utils.distributed.dimmap import DimMap


def setup_distributed():
    """Initialize distributed training with torchrun."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    return device, rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


class MockHookPoint:
    """Mock hook point for testing."""

    def __init__(self, name: str):
        self.name = name


class MockModel:
    """Mock model with run_with_hooks interface (TransformerLens style)."""

    def __init__(self, device: str):
        self.device = device
        self.hook_points = {
            "layer_0": MockHookPoint("layer_0"),
            "layer_1": MockHookPoint("layer_1"),
        }

    def run_with_hooks(
        self,
        input_tensor: torch.Tensor,
        fwd_hooks: list[tuple[str, Callable]] = [],
        bwd_hooks: list[tuple[str, Callable]] = [],
    ) -> torch.Tensor:
        """
        Simplified run_with_hooks that applies forward hooks to intermediate activations.
        Hook signature: (activation, hook) -> activation
        """
        x = input_tensor

        # Simulate layer 0
        x = x * 2  # Some transformation
        for hook_name, hook_fn in fwd_hooks:
            if hook_name == "layer_0":
                result = hook_fn(x, self.hook_points["layer_0"])
                if result is not None:
                    x = result

        # Simulate layer 1
        x = x + 1  # Some transformation
        for hook_name, hook_fn in fwd_hooks:
            if hook_name == "layer_1":
                result = hook_fn(x, self.hook_points["layer_1"])
                if result is not None:
                    x = result

        return x


def run_with_hooks_distributed(
    model: MockModel,
    input_tensor: torch.Tensor,
    device_mesh: DeviceMesh | None,
    fwd_hooks: list[tuple[Union[str, Callable], Callable]] = [],
    bwd_hooks: list[tuple[Union[str, Callable], Callable]] = [],
) -> torch.Tensor:
    """
    Decoupled version of run_with_hooks for testing.
    Wraps hooks to handle DTensor <-> local tensor conversion.
    """
    if device_mesh is None:
        return model.run_with_hooks(input_tensor, fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks)

    def to_tensor(input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, DTensor):
            assert input.placements == DimMap({"data": 0}).placements(cast(DeviceMesh, device_mesh))
            return input.to_local()
        else:
            return input

    def to_dtensor(input: torch.Tensor) -> torch.Tensor:
        return (
            DTensor.from_local(
                input,
                device_mesh=device_mesh,
                placements=DimMap({"data": 0}).placements(cast(DeviceMesh, device_mesh)),
            )
            if isinstance(input, torch.Tensor)
            else input
        )

    def wrap_hook_for_local(hook_fn):
        def wrapped_hook_fn(*args, **kwargs):
            args = pytree.tree_map(to_dtensor, args)
            kwargs = pytree.tree_map(to_dtensor, kwargs)
            return pytree.tree_map(to_tensor, hook_fn(*args, **kwargs))

        return wrapped_hook_fn

    wrapped_fwd_hooks = [(name, wrap_hook_for_local(hook)) for name, hook in fwd_hooks]
    wrapped_bwd_hooks = [(name, wrap_hook_for_local(hook)) for name, hook in bwd_hooks]

    return model.run_with_hooks(input_tensor, fwd_hooks=wrapped_fwd_hooks, bwd_hooks=wrapped_bwd_hooks)


def test_hook_receives_dtensor():
    """Test that hooks receive DTensor and can return DTensor."""
    device, rank, world_size = setup_distributed()

    try:
        if torch.cuda.is_available():
            device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("data",))
        else:
            device_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("data",))

        model = MockModel(device)

        # Create local input tensor (each rank has its own shard)
        # Total batch size = 4, each rank gets 2
        local_batch_size = 2
        seq_len = 3
        d_model = 4
        local_input = torch.ones(local_batch_size, seq_len, d_model, device=device) * (rank + 1)

        received_types = []

        def hook_that_checks_dtensor(activation, hook):
            """Hook that verifies it receives DTensor and returns modified DTensor."""
            received_types.append(type(activation).__name__)

            # Verify activation is DTensor
            assert isinstance(activation, DTensor), f"Expected DTensor, got {type(activation)}"

            # Verify shape is the full (global) shape
            expected_global_shape = (local_batch_size * world_size, seq_len, d_model)
            assert activation.shape == expected_global_shape, (
                f"Expected global shape {expected_global_shape}, got {activation.shape}"
            )

            # Modify and return DTensor
            return activation * 0.5

        output = run_with_hooks_distributed(
            model,
            local_input,
            device_mesh,
            fwd_hooks=[("layer_0", hook_that_checks_dtensor)],
        )

        # Verify hook was called
        assert len(received_types) == 1
        assert received_types[0] == "DTensor"

        # Verify output is local tensor
        assert not isinstance(output, DTensor), "Output should be local tensor"
        assert output.shape == (local_batch_size, seq_len, d_model)

        print(f"Rank {rank}: test_hook_receives_dtensor passed!")

    finally:
        cleanup_distributed()


def test_hook_returning_none():
    """Test that hooks can return None (no modification)."""
    device, rank, world_size = setup_distributed()

    try:
        if torch.cuda.is_available():
            device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("data",))
        else:
            device_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("data",))

        model = MockModel(device)
        local_input = torch.ones(2, 3, 4, device=device)

        def hook_returning_none(activation, hook):
            """Hook that returns None (passes through)."""
            return None

        # This should not raise an error
        output = run_with_hooks_distributed(
            model,
            local_input,
            device_mesh,
            fwd_hooks=[("layer_0", hook_returning_none)],
        )

        # Without modification: input * 2 + 1 = 3
        expected = torch.ones(2, 3, 4, device=device) * 3
        assert torch.allclose(output, expected), f"Expected {expected}, got {output}"

        print(f"Rank {rank}: test_hook_returning_none passed!")

    finally:
        cleanup_distributed()


def test_multiple_hooks():
    """Test multiple hooks on different layers."""
    device, rank, world_size = setup_distributed()

    try:
        if torch.cuda.is_available():
            device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("data",))
        else:
            device_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("data",))

        model = MockModel(device)
        local_input = torch.ones(2, 3, 4, device=device)

        hook_calls = []

        def hook_layer_0(activation, hook):
            hook_calls.append("layer_0")
            assert isinstance(activation, DTensor)
            return activation * 2  # Double the activation

        def hook_layer_1(activation, hook):
            hook_calls.append("layer_1")
            assert isinstance(activation, DTensor)
            return activation - 1  # Subtract 1

        output = run_with_hooks_distributed(
            model,
            local_input,
            device_mesh,
            fwd_hooks=[("layer_0", hook_layer_0), ("layer_1", hook_layer_1)],
        )

        # Verify both hooks were called
        assert hook_calls == ["layer_0", "layer_1"]

        # Calculate expected output:
        # input = 1
        # after layer_0 transform: 1 * 2 = 2
        # after hook_layer_0: 2 * 2 = 4
        # after layer_1 transform: 4 + 1 = 5
        # after hook_layer_1: 5 - 1 = 4
        expected = torch.ones(2, 3, 4, device=device) * 4
        assert torch.allclose(output, expected), f"Expected {expected}, got {output}"

        print(f"Rank {rank}: test_multiple_hooks passed!")

    finally:
        cleanup_distributed()


def test_without_device_mesh():
    """Test that function works without device mesh (non-distributed case)."""
    device, rank, world_size = setup_distributed()

    try:
        model = MockModel(device)
        local_input = torch.ones(2, 3, 4, device=device)

        def simple_hook(activation, hook):
            # In non-distributed case, activation is regular tensor
            assert not isinstance(activation, DTensor)
            return activation * 0.5

        output = run_with_hooks_distributed(
            model,
            local_input,
            device_mesh=None,  # No distributed
            fwd_hooks=[("layer_0", simple_hook)],
        )

        # input * 2 * 0.5 + 1 = 2
        expected = torch.ones(2, 3, 4, device=device) * 2
        assert torch.allclose(output, expected)

        print(f"Rank {rank}: test_without_device_mesh passed!")

    finally:
        cleanup_distributed()


def test_dtensor_operations_in_hook():
    """Test that DTensor operations work correctly inside hooks."""
    device, rank, world_size = setup_distributed()

    try:
        if torch.cuda.is_available():
            device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("data",))
        else:
            device_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("data",))

        model = MockModel(device)
        local_input = torch.arange(24, device=device, dtype=torch.float).reshape(2, 3, 4) + rank * 24

        def hook_with_dtensor_ops(activation, hook):
            """Hook that performs DTensor-aware operations."""
            assert isinstance(activation, DTensor)

            # Get global mean (should work across all shards)
            global_mean = activation.mean()

            # Normalize by global mean
            return activation - global_mean

        output = run_with_hooks_distributed(
            model,
            local_input,
            device_mesh,
            fwd_hooks=[("layer_0", hook_with_dtensor_ops)],
        )

        assert not isinstance(output, DTensor)
        print(f"Rank {rank}: test_dtensor_operations_in_hook passed!")
        print(f"Rank {rank}: Output mean = {output.mean().item():.4f}")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    print("Running run_with_hooks distributed tests...")
    print("=" * 50)

    print("\n1. Testing hook receives DTensor...")
    test_hook_receives_dtensor()

    print("\n2. Testing hook returning None...")
    test_hook_returning_none()

    print("\n3. Testing multiple hooks...")
    test_multiple_hooks()

    print("\n4. Testing without device mesh...")
    test_without_device_mesh()

    print("\n5. Testing DTensor operations in hook...")
    test_dtensor_operations_in_hook()

    print("\n" + "=" * 50)
    print("All tests passed!")
