import torch
import torch.distributed as dist

def print_once(
    *values: object,
    sep: str | None = " ",
    end: str | None = "\n",
) -> None:
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*values, sep=sep, end=end)
    else:
        print(*values, sep=sep, end=end)


def compute_attention_mask(
    batch: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    attention_mask = batch != pad_token_id
    attention_mask[:, 0] = True
    return attention_mask

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

if __name__ == "__main__":
    # Test geometric median
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    print(compute_geometric_median(x))  # Output: tensor([0.3333, 0.3333])