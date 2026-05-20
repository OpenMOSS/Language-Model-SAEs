# Codes for Evolution of Concepts in Language Model Pre-Training

## Install the Environment

We use [uv](https://docs.astral.sh/uv/getting-started/installation/) as the dependency manager. Install `uv`, and run:

```bash
uv sync --extra default
```

to fetch all dependencies.

## Replicate the Crosscoders

To replicate our key results, you need to generate Pythia model activations, train the crosscoders, and analyze the crosscoders.

### Requirements

The following instructions assume you have access to a GPU cluster with at least 16 NVIDIA A100s/H100s or better GPUs, with CUDA version 12.8. With some simple modifications (e.g. change all `"cuda"` to `"npu"`, and install the environment by `uv sync --extra npu`), these codes can also run on an NPU cluster with at least 32 Ascend 910B or better NPUs. The cluster should have a large disk space (>200T) to save all model activations.

Our scripts also require you have a [subset](https://huggingface.co/datasets/Hzfinfdu/SlimPajama-3B) of the SlimPajama dataset saved by `dataset.save_to_disk()` at `~/data/SlimPajama-3B`, and all Pythia model checkpoints at `~/models/pythia-{size}-all/step{step}`, where `size` can be `160m` or `6.9b`. You can change the paths in the scripts to your own paths.

### Generate Activations

Two types of model activations are required for training and analyzing crosscoders:

1. **1D Activations:** Activations where the context dimension folds into the batch dimension and re-shuffled. Typically with the shape of `(batch, d_model)`. Use for crosscoder training.
2. **2D Activations:** Activations where the context dimension is reserved. Typically with the shape of `(batch, n_context, d_model)`. Use for crosscoder analyzing.

To generate 1D activations of Pythia-160M, run:

```bash
uv run torchrun --nproc-per-node=8 generate-pythia-activations-1d.py --size 160m --layer 6
```

This will take up ~40T disk space.

To generate 2D activations of Pythia-160M, run:

```bash
uv run torchrun --nproc-per-node=8 generate-pythia-activations-2d.py --size 160m --layer 6
```

To generate 1D activations of Pythia-6.9B, run:

```bash
uv run torchrun --nproc-per-node=8 generate-pythia-activations-1d.py --size 6.9b --layer 16
```

This will take up ~170T disk space.

To generate 2D activations of Pythia-160M, run:

```bash
uv run torchrun --nproc-per-node=8 generate-pythia-activations-2d.py --size 6.9b --layer 16
```

### Training Crosscoders

To train crosscoders on Pythia-160M, run:

```bash
uv run torchrun --nproc-per-node=8 train-pythia-crosscoders.py --init_encoder_factor 1 --lr 5e-5 --l1_coefficient 0.3 --jumprelu_lr_factor 0.1 --layer 6 --expansion_factor 32 --batch_size 2048
```

To train crosscoders on Pythia-6.9B, run:

```bash
uv run torchrun --nproc-per-node=8 --nnodes=2 train-pythia-crosscoders.py --init_encoder_factor 1 --lr 1e-5 --l1_coefficient 0.3 --jumprelu_lr_factor 0.3 --layer 16 --expansion_factor 8 --batch_size 2048 --size 6.9b # Require 2 nodes
```

You can modify the `expansion_factor` to get crosscoders with different dictionary sizes, and modify the `l1_coefficient` to move the trade-off between sparsity and reconstruction fidelity.

### Analyze Crosscoders

To analyze trained crosscoders, you should first have a MongoDB instance run at `localhost:27017`, and run

```bash
uv run analyze-pythia-crosscoder.py --name <crosscoder-name> --batch-size 16
```

where `<crosscoder-name>` is the name of your trained crosscoder. Results will be saved to the MongoDB. Afterwards, you can use our visualization tool to view the features:

```bash
cd ../../ui
bun install
bun run dev
```
