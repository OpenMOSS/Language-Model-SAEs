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
    x: Float[torch.Tensor, "batch n_layers d_sae"],
    k: int,
    dim: Union[int, Tuple[int, ...]] = -1,
    tolerance: int = 1,
    max_iterations: int = 50,
    *,
    return_threshold: Literal[False] = False,
) -> Float[torch.Tensor, "batch n_layers d_sae"]: ...


@overload
def topk(
    x: Float[torch.Tensor, "batch n_layers d_sae"],
    k: int,
    dim: Union[int, Tuple[int, ...]] = -1,
    tolerance: int = 1,
    max_iterations: int = 50,
    *,
    return_threshold: Literal[True],
) -> tuple[Float[torch.Tensor, "batch n_layers d_sae"], torch.Tensor]: ...


def topk(
    x: Float[torch.Tensor, "batch n_layers d_sae"],
    k: int,
    dim: Union[int, Tuple[int, ...]] = -1,
    tolerance: int = 1,
    max_iterations: int = 50,
    *,
    return_threshold: bool = False,
) -> Union[
    Float[torch.Tensor, "batch n_layers d_sae"], Tuple[Float[torch.Tensor, "batch n_layers d_sae"], torch.Tensor]
]:
    """
    Perform distributed batch kthvalue operation on a DTensor using binary search.

    Args:
        x: Input tensor of shape (batch, n_layers, d_sae)
        k: Target number of top elements to keep
        dim: Dimension(s) along which to perform topk operation
        tolerance: Acceptable range for the number of elements above threshold
        max_iterations: Maximum number of binary search iterations
        return_threshold: If True, return both the result tensor and the threshold value

    Returns:
        If return_threshold is False, returns the filtered tensor.
        If return_threshold is True, returns a tuple of (filtered tensor, threshold).
    """
    with torch.no_grad():
        if isinstance(dim, int):
            dim = (dim,)

        def _ensure_positive_dim(dim: Tuple[int, ...]) -> Tuple[int, ...]:
            """We want to ensure that the dims are positive"""
            return tuple(d if d >= 0 else d + x.ndim for d in dim)

        dim = _ensure_positive_dim(dim)

        constant_dims = tuple(d for d in range(x.ndim) if d not in dim)
        constant_dim_size = tuple(x.size(d) for d in constant_dims)

        k_lower_bound, k_upper_bound = k - tolerance, k + tolerance
        search_low_val = torch.zeros(constant_dim_size, device=x.device)
        search_high_val = torch.full(constant_dim_size, x.max().item(), device=x.device)

        x_flat = x.flatten(start_dim=len(constant_dims))

        threshold = (search_low_val + search_high_val) / 2
        for _ in range(max_iterations):
            threshold = (search_low_val + search_high_val) / 2

            count_above_threshold = (x_flat > threshold.unsqueeze(-1)).sum(-1)
            # All-reduce to get total count across all ranks

            if ((k_lower_bound <= count_above_threshold) * (count_above_threshold <= k_upper_bound)).all():
                break

            to_increase = count_above_threshold > k_upper_bound
            to_decrease = count_above_threshold < k_lower_bound

            if to_increase.any():
                search_low_val = torch.where(to_increase, threshold, search_low_val)
            if to_decrease.any():
                search_high_val = torch.where(to_decrease, threshold, search_high_val)

            # Check for convergence
            if (search_high_val - search_low_val < 1e-6).all():
                break

        while threshold.ndim < x.ndim:
            threshold = threshold[..., None]

    if return_threshold:
        return x * x.ge(threshold), threshold
    else:
        return x * x.ge(threshold)
