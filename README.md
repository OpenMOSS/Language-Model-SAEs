# Language-Model-SAEs

> [!IMPORTANT]
> Currently the examples are outdated and some parallelism strategies are not working due to lack of bandwidth. We are working on better organizing recent updates and will make everything work ASAP.

## News

- 2025.9.23 We leverage **Crosscoder** to track feature evolution across pre-training snapshots. Link: [Evolution of Concepts in Language Model Pre-Training](https://www.arxiv.org/abs/2509.17196).

- 2025.8.23 We identify a prevalent low-rank structure in attention outputs as the key cause of dead features, and propose **Active Subspace Initialization** to improve sparse dictionary learning on these low-rank activations. Link: [Attention Layers Add Into Low-Dimensional Residual Subspaces](https://arxiv.org/abs/2508.16929).

- 2025.4.29 We introduce **Low-Rank Sparse Attention (Lorsa)** to attack attention superposition, extracting tens of thousands of true attention units from LLM attention layers. Link: [Towards Understanding the Nature of Attention with Low-Rank Sparse Decomposition](https://arxiv.org/abs/2504.20938).

- 2024.10.29 We introduce **Llama Scope**, our first contribution to the open-source Sparse Autoencoder ecosystem. Stay tuned! Link: [Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders](http://arxiv.org/abs/2410.20526).

- 2024.10.9 Transformers and Mambas are mechanistically similar in both feature and circuit level. Can we follow this line and find **universal motifs and fundamental differences between language model architectures**? Link: [Towards Universality: Studying Mechanistic Similarity Across Language Model Architectures](https://arxiv.org/pdf/2410.06672).

- 2024.5.22 We propose hierarchical tracing, a promising method to **scale up sparse feature circuit analysis** to industrial size language models! Link: [Automatically Identifying Local and Global Circuits with Linear Computation Graphs](https://arxiv.org/pdf/2405.13868).

- 2024.2.19 Our first attempt on SAE-based circuit analysis for Othello-GPT leads us to **an example of Attention Superposition in the wild**! Link: [Dictionary learning improves patch-free circuit discovery in mechanistic interpretability: A case study on othello-gpt](https://arxiv.org/pdf/2402.12201).

## Installation

We use [uv](https://docs.astral.sh/uv/) to manage the dependencies, which is an alternative to [poetry](https://python-poetry.org/) or [pdm](https://pdm-project.org/). To install the required packages, just install [uv](https://docs.astral.sh/uv/getting-started/installation/), and run the following command:

```bash
uv sync --extra default
```

This will install all the required packages for the codebase in `.venv` directory. For Ascend NPU support, run

```bash
uv sync --extra npu
```

A forked version of `TransformerLens` is also included in the dependencies to provide the necessary tools for analyzing features.

If you want to use the visualization tools, you also need to install the required packages for the frontend, which uses [bun](https://bun.sh/) for dependency management. Follow the instructions on the website to install it, and then run the following command:

```bash
cd ui
bun install
```

`bun` is not well-supported on Windows, so you may need to use WSL or other Linux-based solutions to run the frontend, or consider using a different package manager, such as `pnpm` or `yarn`.

## Launch an Experiment

The guidelines and examples for launching experiments are generally outdated. At this moment, you may explore `src/lm_saes/runners` folder for the interface for generating activations and training & analyzing SAE variants. For analyzing SAEs, a MongoDB instance is required. More instructions will be provided in near future.

## Visualizing the Learned Dictionary

The analysis results will be saved using MongoDB, and you can use the provided visualization tools to visualize the learned dictionary. First, start the FastAPI server by running the following command:

```bash
uvicorn server.app:app --port 24577 --env-file server/.env
```

Then, copy the `ui/.env.example` file to `ui/.env` and modify the `VITE_BACKEND_URL` to fit your server settings (by default, it's `http://localhost:24577`), and start the frontend by running the following command:

```bash
cd ui
bun dev --port 24576
```

That's it! You can now go to `http://localhost:24576` to visualize the learned dictionary and its features.

## Development

We highly welcome contributions to this project. If you have any questions or suggestions, feel free to open an issue or a pull request. We are looking forward to hearing from you!

TODO: Add development guidelines

## Acknowledgement

The design of the pipeline (including the configuration and some training details) is highly inspired by the [mats_sae_training
](https://github.com/jbloomAus/mats_sae_training) project (now known as [SAELens](https://github.com/jbloomAus/SAELens)) and heavily relies on the [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) library. We thank the authors for their great work.

## Citation

Please cite this library as:

```
@misc{Ge2024OpenMossSAEs,
    title  = {OpenMoss Language Model Sparse Autoencoders},
    author = {Xuyang Ge, Fukang Zhu, Junxuan Wang, Wentao Shu, Lingjie Chen, Zhengfu He},
    url    = {https://github.com/OpenMOSS/Language-Model-SAEs},
    year   = {2024}
}
```
