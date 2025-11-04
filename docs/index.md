# Language Model SAEs

Welcome to the documentation for **Language Model SAEs** - a library for training and analyzing Sparse Autoencoders (SAEs) on language models.

## Overview

Sparse Autoencoders (SAEs) are neural network models used to extract interpretable features from language models. They help address the superposition problem in neural networks by learning sparse, interpretable representations of activations.

This library provides:

- **Scalability**: Our framework is fully distributed with arbitrary combinations of data, model, and head parallelism for both training and analysis. Enjoy training SAEs with millions of features!
- **Flexibility**: We support a wide range of SAE variants, including vanilla SAEs, Lorsa (Low-rank Sparse Attention), CLT (Cross-layer Transcoder), MoLT (Mixture of Linear Transforms), CrossCoder, and more. Each variant can be combined with different activation functions (e.g., ReLU, JumpReLU, TopK, BatchTopK) and sparsity penalties (e.g., L1, Tanh).
- **Easy to Use**: We provide high-level `runners` APIs to quickly launch experiments with simple configurations. Check our [examples](examples) for verified hyperparameters.
- **Visualization**: We provide a unified web interface to visualize learned SAE variants and their features.

## Getting Started
