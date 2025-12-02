# Workthrough

`Language-Model-SAEs` provides a general way to train, analyze and visualize Sparse Autoencoders and their variants. To help you get started quickly, we've included [example scripts]() that guide you through each stage of working with SAEs. This guide begins with a foundational example and progressively introduces the core features and capabilities of the library.

## Training Basic Sparse Autoencoders

A [Sparse Autoencoder]() is trained to reconstruct model activations at specific position. We depend on [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) to take activations out of model forward pass, specified by hook points. To train a vanilla SAE on Pythia 160M Layer 6 output, you can create the following `TrainSAESetting`:

```python

```