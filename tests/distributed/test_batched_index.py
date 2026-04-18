import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from llamascopium.testing import distributed_test
from llamascopium.utils.distributed.ops import batch_index, multi_batch_index


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@distributed_test(nproc_per_node=2, backend="nccl")
def test_batch_index_no_batch():
    device_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("model",))

    x = distribute_tensor(
        torch.tensor(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
        device_mesh,
        [Shard(1)],
    ).requires_grad_(True)
    indices = distribute_tensor(torch.tensor([[0, 0], [1, 2], [2, 3]], device="cuda"), device_mesh, [Replicate()])

    result = batch_index(x, indices, preserve_order=True)
    result.sum().backward()

    assert torch.allclose(result.full_tensor(), torch.tensor([0, 6, 11], device="cuda", dtype=torch.float32))
    assert torch.allclose(
        x.grad.full_tensor(),
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@distributed_test(nproc_per_node=2, backend="nccl")
def test_batch_index_with_batch():
    device_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("model",))

    x = distribute_tensor(
        torch.tensor(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
        device_mesh,
        [Shard(1)],
    ).requires_grad_(True)
    indices = distribute_tensor(torch.tensor([[0], [1], [2]], device="cuda"), device_mesh, [Replicate()])

    result = batch_index(x, indices, n_batch_dims=1, preserve_order=True)
    result.sum().backward()

    assert torch.allclose(
        result.full_tensor(), torch.tensor([[0, 1, 2], [4, 5, 6], [8, 9, 10]], device="cuda", dtype=torch.float32)
    )
    assert torch.allclose(
        x.grad.full_tensor(),
        torch.tensor(
            [
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 0],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@distributed_test(nproc_per_node=2, backend="nccl")
def test_batch_index_no_full_index():
    device_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("model",))

    x = distribute_tensor(
        torch.tensor(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
        device_mesh,
        [Shard(0)],
    ).requires_grad_(True)
    indices = distribute_tensor(torch.tensor([[0], [2], [3]], device="cuda"), device_mesh, [Replicate()])

    result = batch_index(x, indices, n_batch_dims=0, preserve_order=True)
    result.sum().backward()

    assert torch.allclose(
        result.full_tensor(),
        torch.tensor([[0, 1, 2, 3], [8, 9, 10, 11], [12, 13, 14, 15]], device="cuda", dtype=torch.float32),
    )
    assert torch.allclose(
        x.grad.full_tensor(),
        torch.tensor(
            [
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@distributed_test(nproc_per_node=4, backend="nccl")
def test_batch_index_shard2():
    device_mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("a", "b"))

    x = distribute_tensor(
        torch.tensor(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
        device_mesh,
        [Shard(0), Shard(1)],
    ).requires_grad_(True)

    indices = distribute_tensor(
        torch.tensor([[0, 1], [0, 3], [2, 2]], device="cuda"), device_mesh, [Replicate(), Replicate()]
    )

    result = batch_index(x, indices, preserve_order=True)
    result.sum().backward()

    assert torch.allclose(
        result.full_tensor(),
        torch.tensor([1, 3, 10], device="cuda", dtype=torch.float32),
    )
    assert torch.allclose(
        x.grad.full_tensor(),
        torch.tensor(
            [
                [0, 1, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@distributed_test(nproc_per_node=2, backend="nccl")
def test_batch_index_dim3():
    device_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("model",))

    x = distribute_tensor(
        torch.tensor(
            [
                [[0, 1], [2, 3]],
                [[4, 5], [6, 7]],
                [[8, 9], [10, 11]],
                [[12, 13], [14, 15]],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
        device_mesh,
        [Shard(2)],
    ).requires_grad_(True)

    indices = distribute_tensor(torch.tensor([[0, 1], [1, 0], [1, 1]], device="cuda"), device_mesh, [Replicate()])

    result = batch_index(x, indices, n_batch_dims=1, preserve_order=True)
    result.sum().backward()

    print(result.full_tensor())

    assert torch.allclose(
        result.full_tensor(),
        torch.tensor([[1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15]], device="cuda", dtype=torch.float32),
    )
    assert torch.allclose(
        x.grad.full_tensor(),
        torch.tensor(
            [
                [[0, 1], [1, 1]],
                [[0, 1], [1, 1]],
                [[0, 1], [1, 1]],
                [[0, 1], [1, 1]],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@distributed_test(nproc_per_node=2, backend="nccl")
def test_batch_index_dim3_no_reserve_order():
    device_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("model",))

    x = distribute_tensor(
        torch.tensor(
            [
                [[0, 1], [2, 3]],
                [[4, 5], [6, 7]],
                [[8, 9], [10, 11]],
                [[12, 13], [14, 15]],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
        device_mesh,
        [Shard(2)],
    ).requires_grad_(True)

    indices = distribute_tensor(torch.tensor([[0, 1], [1, 0], [1, 1]], device="cuda"), device_mesh, [Replicate()])

    result = batch_index(x, indices, n_batch_dims=1, preserve_order=False)
    result.sum().backward()

    print(result.full_tensor())

    assert torch.allclose(
        result.full_tensor(),
        torch.tensor([[2, 1, 3], [6, 5, 7], [10, 9, 11], [14, 13, 15]], device="cuda", dtype=torch.float32),
    )
    assert torch.allclose(
        x.grad.full_tensor(),
        torch.tensor(
            [
                [[0, 1], [1, 1]],
                [[0, 1], [1, 1]],
                [[0, 1], [1, 1]],
                [[0, 1], [1, 1]],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@distributed_test(nproc_per_node=2, backend="nccl")
def test_multi_batch_index_no_batch():
    device_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("model",))

    x1 = distribute_tensor(
        torch.tensor(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
        device_mesh,
        [Shard(1)],
    ).requires_grad_(True)

    x2 = distribute_tensor(
        torch.tensor(
            [
                [10, 11, 12, 13],
                [14, 15, 16, 17],
                [18, 19, 20, 21],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
        device_mesh,
        [Shard(1)],
    ).requires_grad_(True)

    indices1 = distribute_tensor(torch.tensor([[0, 0], [1, 2], [2, 3]], device="cuda"), device_mesh, [Replicate()])
    indices2 = distribute_tensor(torch.tensor([[0, 1], [1, 1], [2, 2]], device="cuda"), device_mesh, [Replicate()])

    results = multi_batch_index([(x1, indices1), (x2, indices2)], preserve_order=True)

    loss = results[0].sum() + results[1].sum()
    loss.backward()

    assert torch.allclose(results[0].full_tensor(), torch.tensor([0, 6, 11], device="cuda", dtype=torch.float32))
    assert torch.allclose(results[1].full_tensor(), torch.tensor([11, 15, 20], device="cuda", dtype=torch.float32))

    assert torch.allclose(
        x1.grad.full_tensor(),
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
    )
    assert torch.allclose(
        x2.grad.full_tensor(),
        torch.tensor(
            [
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@distributed_test(nproc_per_node=2, backend="nccl")
def test_multi_batch_index_with_batch():
    device_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("model",))

    x1 = distribute_tensor(
        torch.tensor(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
        device_mesh,
        [Shard(1)],
    ).requires_grad_(True)

    x2 = distribute_tensor(
        torch.tensor(
            [
                [10, 11, 12, 13],
                [14, 15, 16, 17],
                [18, 19, 20, 21],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
        device_mesh,
        [Shard(1)],
    ).requires_grad_(True)

    indices1 = distribute_tensor(torch.tensor([[0], [1], [2]], device="cuda"), device_mesh, [Replicate()])
    indices2 = distribute_tensor(torch.tensor([[1], [0], [3]], device="cuda"), device_mesh, [Replicate()])

    results = multi_batch_index([(x1, indices1), (x2, indices2)], n_batch_dims=1, preserve_order=True)

    loss = results[0].sum() + results[1].sum()
    loss.backward()

    assert torch.allclose(
        results[0].full_tensor(), torch.tensor([[0, 1, 2], [4, 5, 6], [8, 9, 10]], device="cuda", dtype=torch.float32)
    )
    assert torch.allclose(
        results[1].full_tensor(),
        torch.tensor([[11, 10, 13], [15, 14, 17], [19, 18, 21]], device="cuda", dtype=torch.float32),
    )
    assert torch.allclose(
        x1.grad.full_tensor(),
        torch.tensor(
            [
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 0],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
    )
    assert torch.allclose(
        x2.grad.full_tensor(),
        torch.tensor(
            [
                [1, 1, 0, 1],
                [1, 1, 0, 1],
                [1, 1, 0, 1],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@distributed_test(nproc_per_node=2, backend="nccl")
def test_multi_batch_index_dim3():
    device_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("model",))

    x1 = distribute_tensor(
        torch.tensor(
            [
                [[0, 1], [2, 3]],
                [[4, 5], [6, 7]],
                [[8, 9], [10, 11]],
                [[12, 13], [14, 15]],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
        device_mesh,
        [Shard(2)],
    ).requires_grad_(True)

    x2 = distribute_tensor(
        torch.tensor(
            [
                [[10, 11], [12, 13]],
                [[14, 15], [16, 17]],
                [[18, 19], [20, 21]],
                [[22, 23], [24, 25]],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
        device_mesh,
        [Shard(2)],
    ).requires_grad_(True)

    indices1 = distribute_tensor(torch.tensor([[0, 1], [1, 0], [1, 1]], device="cuda"), device_mesh, [Replicate()])
    indices2 = distribute_tensor(torch.tensor([[1, 0], [0, 1], [0, 0]], device="cuda"), device_mesh, [Replicate()])

    results = multi_batch_index([(x1, indices1), (x2, indices2)], n_batch_dims=1, preserve_order=True)

    loss = results[0].sum() + results[1].sum()
    loss.backward()

    assert torch.allclose(
        results[0].full_tensor(),
        torch.tensor([[1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15]], device="cuda", dtype=torch.float32),
    )
    assert torch.allclose(
        results[1].full_tensor(),
        torch.tensor([[12, 11, 10], [16, 15, 14], [20, 19, 18], [24, 23, 22]], device="cuda", dtype=torch.float32),
    )

    assert torch.allclose(
        x1.grad.full_tensor(),
        torch.tensor(
            [
                [[0, 1], [1, 1]],
                [[0, 1], [1, 1]],
                [[0, 1], [1, 1]],
                [[0, 1], [1, 1]],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
    )
    assert torch.allclose(
        x2.grad.full_tensor(),
        torch.tensor(
            [
                [[1, 1], [1, 0]],
                [[1, 1], [1, 0]],
                [[1, 1], [1, 0]],
                [[1, 1], [1, 0]],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@distributed_test(nproc_per_node=2, backend="nccl")
def test_multi_batch_index_dim3_no_reserve_order():
    device_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("model",))

    x1 = distribute_tensor(
        torch.tensor(
            [
                [[0, 1], [2, 3]],
                [[4, 5], [6, 7]],
                [[8, 9], [10, 11]],
                [[12, 13], [14, 15]],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
        device_mesh,
        [Shard(2)],
    ).requires_grad_(True)

    x2 = distribute_tensor(
        torch.tensor(
            [
                [[10, 11], [12, 13]],
                [[14, 15], [16, 17]],
                [[18, 19], [20, 21]],
                [[22, 23], [24, 25]],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
        device_mesh,
        [Shard(2)],
    ).requires_grad_(True)

    indices1 = distribute_tensor(torch.tensor([[0, 1], [1, 0], [1, 1]], device="cuda"), device_mesh, [Replicate()])
    indices2 = distribute_tensor(torch.tensor([[1, 0], [0, 1], [0, 0]], device="cuda"), device_mesh, [Replicate()])

    results = multi_batch_index([(x1, indices1), (x2, indices2)], n_batch_dims=1, preserve_order=False)

    loss = results[0].sum() + results[1].sum()
    loss.backward()

    assert torch.allclose(
        results[0].full_tensor(),
        torch.tensor([[2, 1, 3], [6, 5, 7], [10, 9, 11], [14, 13, 15]], device="cuda", dtype=torch.float32),
    )
    assert torch.allclose(
        results[1].full_tensor(),
        torch.tensor([[12, 10, 11], [16, 14, 15], [20, 18, 19], [24, 22, 23]], device="cuda", dtype=torch.float32),
    )

    assert torch.allclose(
        x1.grad.full_tensor(),
        torch.tensor(
            [
                [[0, 1], [1, 1]],
                [[0, 1], [1, 1]],
                [[0, 1], [1, 1]],
                [[0, 1], [1, 1]],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
    )
    assert torch.allclose(
        x2.grad.full_tensor(),
        torch.tensor(
            [
                [[1, 1], [1, 0]],
                [[1, 1], [1, 0]],
                [[1, 1], [1, 0]],
                [[1, 1], [1, 0]],
            ],
            device="cuda",
            dtype=torch.float32,
        ),
    )
