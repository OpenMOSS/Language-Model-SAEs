# Language-Model-SAEs

This repo aims to provide a general codebase for conducting dictionary-learning-based mechanistic interpretability research on Language Models (LMs). It powers a configurable pipeline for training and evaluating GPT-2 dictionaries, and provides a set of tools (mainly a React-based webpage) for analyzing and visualizing the learned dictionaries.

The design of the pipeline (including the configuration and some training detail) is highly inspired by the [mats_sae_training
](https://github.com/jbloomAus/mats_sae_training) project and heavily relies on the [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) library. We thank the authors for their great work.

## Getting Started with Mechanistic Interpretability and Dictionary Learning

If you are new to the concept of mechanistic interpretability and dictionary learning, we recommend you to start from the following paper:

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593)
- [Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task](https://arxiv.org/abs/2210.13382)
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
- [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600)

Furthermore, to dive deeper into the inner activations of LMs, it's recommended to get familiar with the [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) library.

## Installation

Currently, the codebase use [pdm](https://pdm-project.org/) to manage the dependencies, which is an alternative to [poetry](https://python-poetry.org/). To install the required packages, just install `pdm`, and run the following command:

```bash
pdm install
```

This will install all the required packages for the core codebase. Note that if you're in a conda environment, `pdm` will directly take the current environment as the virtual environment for current project, and remove all the packages that are not in the `pyproject.toml` file. So make sure to create a new conda environment (or just deactivate conda, this will use virtualenv by default) before running the above command. A forked version of `TransformerLens` is also included in the dependencies to provide the necessary tools for analyzing features.

If you want to use the visualization tools, you also need to install the required packages for the frontend, which uses [bun](https://bun.sh/) for dependency management. Follow the instructions on the website to install it, and then run the following command:

```bash
cd ui
bun install
```

It's worth noting that `bun` is not well-supported on Windows, so you may need to use WSL or other Linux-based solutions to run the frontend, or consider using a different package manager, such as `pnpm` or `yarn`.

## Launch an Experiment

We provide both a programmatic and a configuration-based way to launch an experiment. The configuration-based way is more flexible and recommended for most users. You can find the configuration files in the [examples/configuration](https://github.com/OpenMOSS/Language-Model-SAEs/tree/main/examples/configuration) directory, and modify them to fit your needs. The programmatic way is more suitable for advanced users who want to customize the training process, and you can find the example scripts in the [examples/programmatic](https://github.com/OpenMOSS/Language-Model-SAEs/tree/main/examples/programmatic) directory.

To simply begin a training process, you can run the following command:

```bash
lm-saes train examples/configuration/train.toml
```

which will start the training process using the configuration file [examples/configuration/train.toml](https://github.com/OpenMOSS/Language-Model-SAEs/tree/main/examples/configuration/train.toml).

To analyze a trained dictionary, you can run the following command:

```bash
lm-saes analyze examples/configuration/analyze.toml --sae <path_to_sae_model>
```

which will start the analysis process using the configuration file [examples/configuration/analyze.toml](https://github.com/OpenMOSS/Language-Model-SAEs/tree/main/examples/configuration/analyze.toml). The analysis process requires a trained SAE model, which can be obtained from the training process. You may need launch a MongoDB server to store the analysis results, and you can modify the MongoDB settings in the configuration file.

Generally, our configuration-based pipeline uses outer layer settings as default of the inner layer settings. This is beneficial for easily building deeply nested configurations, where sub-configurations can be reused (such as device and dtype settings). More detail will be provided future.

## Visualizing the Learned Dictionary

The analysis results will be saved using MongoDB, and you can use the provided visualization tools to visualize the learned dictionary. First, start the FastAPI server by running the following command:

```bash
uvicorn server.app:app --port 24577
# You may want to modify some environmental settings in server/.env.example to server/.env, and run with these environmental variables:
# uvicorn server.app:app --port 24577 --env-file server/.env
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
