from typing import Literal, Tuple, Union, overload

import torch
from jaxtyping import Float


def compute_geometric_median(x: torch.Tensor, max_iter=1000) -> torch.Tensor:
    """
    Compute the geometric median of a point cloud x.
    The geometric median is the point that minimizes the sum of distances to the other points.
    This function uses Weiszfeld's algorithm to compute the geometric median.

    Args:
        x: Input point cloud. Shape (n_points, n_dims)
        max_iter: Maximum number of iterations

    Returns:
        The geometric median of the point cloud. Shape (n_dims,)
    """

    # Initialize the geometric median as the mean of the points
    y = x.mean(dim=0)

    for _ in range(max_iter):
        # Compute the weights
        w = 1 / (x - y.unsqueeze(0)).norm(dim=-1)

        # Update the geometric median
        y = (w.unsqueeze(-1) * x).sum(dim=0) / w.sum()

    return y


def norm_ratio(a, b):
    a_norm = torch.norm(a, 2, dim=0).mean()
    b_norm = torch.norm(b, 2, dim=0).mean()
    return a_norm / b_norm


@overload
def topk(
    x: Union[
        Float[torch.Tensor, "batch d_sae"],
        Float[torch.Tensor, "batch n_layers d_sae"],
        Float[torch.Tensor, "batch seq_len d_sae"],
    ],
    k: int,
    dim: Union[int, Tuple[int, ...]] = -1,
    *,
    return_threshold: Literal[False] = False,
) -> Union[
    Float[torch.Tensor, "batch d_sae"],
    Float[torch.Tensor, "batch n_layers d_sae"],
    Float[torch.Tensor, "batch seq_len d_sae"],
]: ...


@overload
def topk(
    x: Union[
        Float[torch.Tensor, "batch d_sae"],
        Float[torch.Tensor, "batch n_layers d_sae"],
        Float[torch.Tensor, "batch seq_len d_sae"],
    ],
    k: int,
    dim: Union[int, Tuple[int, ...]] = -1,
    *,
    return_threshold: Literal[True],
) -> tuple[
    Union[
        Float[torch.Tensor, "batch d_sae"],
        Float[torch.Tensor, "batch n_layers d_sae"],
        Float[torch.Tensor, "batch seq_len d_sae"],
    ],
    torch.Tensor,
]: ...


def topk(
    x: Union[
        Float[torch.Tensor, "batch d_sae"],
        Float[torch.Tensor, "batch n_layers d_sae"],
        Float[torch.Tensor, "batch seq_len d_sae"],
    ],
    k: int,
    dim: Union[int, Tuple[int, ...]] = -1,
    *,
    return_threshold: bool = False,
) -> Union[
    Union[
        Float[torch.Tensor, "batch d_sae"],
        Float[torch.Tensor, "batch n_layers d_sae"],
        Float[torch.Tensor, "batch seq_len d_sae"],
    ],
    Tuple[
        Union[
            Float[torch.Tensor, "batch d_sae"],
            Float[torch.Tensor, "batch n_layers d_sae"],
            Float[torch.Tensor, "batch seq_len d_sae"],
        ],
        torch.Tensor,
    ],
]:
    """
    Perform topk operation on specified dimensions, keeping only the top k values.

    Args:
        x: Input tensor of shape (batch, d_sae) or (batch, n_layers, d_sae) or (batch, seq_len, d_sae)
        k: Target number of top elements to keep
        dim: Dimension(s) along which to perform topk operation
        return_threshold: If True, return both the result tensor and the threshold value

    Returns:
        If return_threshold is False, returns the filtered tensor with non-topk values zeroed out.
        If return_threshold is True, returns a tuple of (filtered tensor, threshold).
    """
    with torch.no_grad():
        if isinstance(dim, int):
            dim = (dim,)

        # Ensure all dimensions are positive indices
        dim = tuple(d if d >= 0 else d + x.ndim for d in dim)

        # Compute the dimensions that remain constant (not involved in topk)
        constant_dims = tuple(d for d in range(x.ndim) if d not in dim)

        # Permute tensor so that topk dimensions are at the end
        perm = constant_dims + tuple(dim)
        x_permuted = x.permute(perm)

        # Record shapes for later restoration
        constant_shape = x_permuted.shape[: len(constant_dims)]
        topk_shape = x_permuted.shape[len(constant_dims) :]

        # Flatten the topk dimensions into a single dimension
        x_flat = x_permuted.flatten(start_dim=len(constant_dims))

        topk_values, topk_indices = torch.topk(x_flat, k=k, dim=-1, sorted=False)

        result_flat = torch.zeros_like(x_flat)
        result_flat.scatter_(dim=-1, index=topk_indices, src=topk_values)

        # Restore the original shape
        result_permuted = result_flat.view(*constant_shape, *topk_shape)

        # Inverse permute to restore original dimension order
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        result = result_permuted.permute(inv_perm)

        if return_threshold:
            threshold = topk_values.min(dim=-1).values
            return result, threshold
        else:
            return result
