import torch
import matplotlib.pyplot as plt
from typing import Dict
import numpy as np

def draw_feature_density(act_times: torch.Tensor, save_path: str):
    """
    Draw the feature density diagram.

    Args:
    - feature_activation_info: A dictionary containing the following:
        - "act_times": A tensor of shape (n_samples,) containing the activation times;
        - "elt": A tensor of shape (n_samples,) containing the feature activation scores;
        - "feature_acts": A tensor of shape (n_samples, n_tokens) containing the feature activations;
        - "contexts": A tensor of shape (n_samples, n_tokens, n_features) containing the contexts;
        - "positions": A tensor of shape (n_samples, n_tokens) containing the positions;
    - save_path: The path to save the diagram.
    """
    bins = 10 ** (np.arange(0, np.log10(act_times.max().item()) + 0.1, 0.1))
    plt.figure(figsize=(10, 5))
    plt.title("Feature Density")
    plt.xlabel("Activation Times")
    plt.ylabel("Feature Count")
    plt.xscale('log')
    plt.hist(act_times.cpu().numpy(), bins=bins)
    plt.savefig(save_path)