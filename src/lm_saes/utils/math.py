import torch


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

@torch.no_grad()
def batch_kthvalue_clt_binary_search(
    x: torch.Tensor,
    k_range: tuple[int, int],
    dim: int = -1,
) -> float:
    """
    Perform batch kthvalue operation on a tensor using binary search.
    
    Args:
        x: Input tensor of shape (batch, n_layers, d_sae) or any 3D tensor
        k_range: Acceptable range for the number of elements above threshold (lower_bound, upper_bound)
        dim: Dimension to operate on (currently not used, assumes 3D tensor)
        
    Returns:
        Threshold value that gives acceptable k elements within the specified range
    """
    if x.dim() != 3:
        raise ValueError("x must be a 3D tensor")
    
    batch_size, n_layers, d_sae = x.shape
    k_range_overall = (k_range[0] * batch_size * n_layers, k_range[1] * batch_size * n_layers)
    
    # Flatten the tensor for easier processing
    x_flat = x.flatten()
    
    # Binary search parameters
    lower_bound, upper_bound = k_range_overall
    search_low = 0.0
    search_high = 10.0
    max_iterations = 50
    tolerance = 1e-6
    
    # First check if 10 is a reasonable upper bound
    count_above_high = (x_flat > search_high).sum()
    
    if count_above_high > upper_bound:
        # Need to increase search space
        search_high *= 2
        count_above_high = (x_flat > search_high).sum()
    
    # Check if we can directly use 0 as threshold
    count_above_0 = (x_flat > 0).sum()
    
    if count_above_0 <= upper_bound:
        # Directly use 0 as threshold
        threshold = 0.0
    else:
        # Binary search for the optimal threshold
        threshold = None
        for iteration in range(max_iterations):
            threshold = (search_low + search_high) / 2
            
            # Count elements above threshold
            count_above_threshold = (x_flat > threshold).sum()
            
            if lower_bound <= count_above_threshold <= upper_bound:
                # Found acceptable threshold
                break
            elif count_above_threshold > upper_bound:
                # Too many elements above threshold, increase threshold
                search_low = threshold
            else:
                # Too few elements above threshold, decrease threshold
                search_high = threshold
            
            # Check for convergence
            if search_high - search_low < tolerance:
                break
        
        # If we didn't find a threshold in range, use the best approximation
        if threshold is None:
            threshold = search_low  # Use the lower bound as a fallback
    
    # Ensure threshold is a scalar value
    if isinstance(threshold, torch.Tensor):
        threshold_value = threshold.item()
    else:
        threshold_value = threshold
    
    return threshold_value
    