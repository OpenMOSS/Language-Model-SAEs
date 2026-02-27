# Distributed Guidelines

A fundamental advantage of `Language-Model-SAEs` is its support of distributed setup, including data parallelism (DP), tensor parallelism (TP), some special parallelism strategy for some specific models, and their arbitrary combination. These strategies avoid OOM and accelerate model computation, making it possible for training arbitrarily large sparse dictionaries for frontier models.

## How does it work

!!! note

    If you don't care about the under-the-hood implementation of our distributed settings, feel free to skip this section. You can still use the distributed settings to speed up everything with ease.

We mainly take advantage of PyTorch [DeviceMesh](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html) and [DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html#pytorch-dtensor-distributed-tensor) to organize distributed storage/computation and collective communication.

### DeviceMesh

[DeviceMesh](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html) is a multi-dimensional mesh structure that manages the distribution of computation across your devices (typically GPUs). Each cell in the mesh represents a single device, and each dimension of the mesh corresponds to a specific parallelism strategy.

<figure markdown="span">
  ![Illustration of DeviceMesh](assets/images/device-mesh.svg)
  <figcaption>An illustration of DeviceMesh with 8 GPUs arranged in a 2×4 grid. This configuration uses Data Parallelism (DP=2) along one dimension and Tensor Parallelism (TP=4) along the other.</figcaption>
</figure>

DeviceMesh provides a standardized framework for implementing multi-dimensional parallelism. For each parallelism strategy, sharding and communication operations occur exclusively along the corresponding dimension of the mesh.

In the example above, the input data is split into 2 shards. GPUs 0-3 process the first shard, while GPUs 4-7 process the second shard. And for the TP dimension, model parameters are partitioned into 4 shards, distributed across GPUs within each data-parallel group (e.g., GPUs 0, 1, 2, 3 each hold one shard of the model).

This mesh abstraction allows you to compose different parallelism strategies cleanly, with each strategy operating independently along its designated dimension.

A DeviceMesh can be created by:

```python
device_mesh = init_device_mesh(
    device_type="cuda",
    mesh_shape=(8,),
    mesh_dim_names=("model",),
)
```

### DTensor

Built on top of DeviceMesh, [DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html#pytorch-dtensor-distributed-tensor) provides an abstraction layer that enables you to work with distributed tensors from a global perspective.

DTensor requires your code to follow the SPMD (Single Program, Multiple Data) paradigm, meaning the same program executes across all processes. Under this model, any tensor created at a specific point in the program has a corresponding tensor in every other process at that same point.

When using regular tensors in distributed settings, these per-process tensors exist independently with no explicit relationship or coordination between them. DTensor addresses this by providing a unified, global view: it logically represents a single large tensor containing all the data across all processes, which is then automatically sharded and distributed to each process according to its `Placement` specifications.

A DTensor can be created by:

```python
local_tensor = torch.randn(2, 4)
dtensor = DTensor.from_local(
    local_tensor,
    device_mesh=device_mesh,
    placements=(Shard(0)),
) # This `dtensor` stores per-device data just as the `local_tensor`. 
# It just provides more information on how the tensor is organized across device globally.
```

More factory method of DTensor can be found at [torch.distributed.tensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html#different-ways-to-create-a-dtensor).

### Determine the Placements of DTensor

The term "dimension" is indeed overloaded. It can refer to a column or a row in a DeviceMesh, or in a tensor in its mathematical meaning. When it comes to "sharding" a DTensor, it actually relates to both of the interpretations. For each of the dimension of DeviceMesh we want the DTensor to shard across, we must select a tensor dimension to perform the sharding.

However, we cannot randomly pick tensor dimensions to shard the tensors on. Suppose we are to perform distributed matrix multiplication with tensors $a \in M \times K$ and $b \in K \times N$. It's only possible to efficiently accelerate the computation when both the tensors are sharded on the middle dimension $K$ or both are sharded in the outer dimension $M$ and $N$, in which case every local device has the required data to perform its block matrix multiplication.

Thus, we need to carefully determine: _for each mesh dimension, which tensor dimension, if any, should be sharded on?_ DTensor uses its `placements` to specify how shardings correspond to each mesh dimension:

```python
# Shard tensor dim 0 along the first mesh dimension, and replicate along the second mesh dimension.
placements = (Shard(0), Replicate())
```

However, this placement tuple are tightly coupling to mesh topology. If the mesh changes, e.g., from `("data", "model")` to just `("model",)` when running on a single node without data parallelism, the placement tuple should be changed correspondingly. 

To make every weight and activation in our codebase flexible to different mesh, we specify their `DimMap` to dynamically compute the placements, taking inspiration from JAX's [PartitionSpec](https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.PartitionSpec). `DimMap` maps *mesh dimension names* to *tensor dimensions*, and any mesh dimension absent from the map is implicitly replicated:

```python
DimMap({"data": 0})            # Shard tensor dim 0 along mesh "data"; replicate elsewhere
DimMap({"model": 1})           # Shard tensor dim 1 along mesh "model"; replicate elsewhere
DimMap({"data": 0, "model": 1}) # Shard along both
DimMap({})                      # Fully replicated
```

Given a concrete `DeviceMesh`, `DimMap.placements(device_mesh)` generates the correct positional `Placement` tuple dynamically.

### Application in Language-Model-SAEs

Ideally the above system can inherently solve multi-dimensional parallelism, including DP and TP -- we just need to provide a DeviceMesh, and specify the DimMap of each leaf node tensor (input and weight), and the tensor operations (matrix multiplications and others) will be automatically accelerated. 

But in practice, this cannot be the full story. Often a [primitive](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml) of PyTorch [does not know how it should deal with DTensor properly](https://github.com/pytorch/pytorch/issues?q=is%3Aissue%20state%3Aopen%20does%20not%20have%20a%20sharding%20strategy%20registered), or the implementation is not performant in distributed cases (run slowly or costs unnecessary extra GPU memory). So there're still a number of corner cases in which we need to convert the DTensors back to local tensor and operate on them manually.

## Accelerate Your Training/Analyzing

The use of distributed strategies is just as simple as other libraries: for the runners we've provided, just specify `data_parallel_size` and `model_parallel_size` in the settings, and launch your experiment via `torchrun`. The total number of processes must equal `data_parallel_size × model_parallel_size`.

**Generate activations** with 8 GPUs (8-way data parallelism):

```bash
uv run torchrun --nproc-per-node=8 examples/generate_pythia_activation_1d.py \
    --size 160m --layer 6 --activation_path /data/activations
```

```python
settings = GenerateActivationsSettings(
    ...,
    data_parallel_size=8,
)
```

**Train SAEs** with 8 GPUs (2-way DP × 4-way TP):

```bash
uv run torchrun --nproc-per-node=8 examples/train_pythia_sae_topk.py
```

```python
settings = TrainSAESettings(
    ...,
    data_parallel_size=2,
    model_parallel_size=4,
)
```

**Analyze SAEs** with 4 GPUs (4-way TP):

```bash
uv run torchrun --nproc-per-node=4 examples/analyze_pythia_sae.py \
    --sae_path /path/to/sae
```

```python
settings = AnalyzeSAESettings(
    ...,
    model_parallel_size=4,
)
```

For custom runners, you may create the `DeviceMesh` yourself and pass it to modules like `ActivationFactory`. Most modules in `Language-Model-SAEs` support `DeviceMesh` inherently.