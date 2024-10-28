import torch


def sort_dict_of_tensor(
    tensor_dict: dict[str, torch.Tensor],
    sort_key: str,
    sort_dim: int = 0,
    descending: bool = True,
):
    """
    Sort a dictionary of tensors by the values of a tensor.

    Args:
        tensor_dict: Dictionary of tensors
        sort_key: Key of the tensor to sort by
        sort_dim: Dimension to sort along
        descending: Whether to sort in descending order

    Returns:
        A dictionary of tensors sorted by the values of the specified tensor
    """
    sorted_idx = tensor_dict[sort_key].argsort(dim=sort_dim, descending=descending)
    return {
        k: v.gather(sort_dim, sorted_idx.unsqueeze(-1).expand_as(v.reshape(*sorted_idx.shape, -1)).reshape_as(v))
        for k, v in tensor_dict.items()
    }


def concat_dict_of_tensor(*dicts: dict[str, torch.Tensor], dim: int = 0) -> dict[str, torch.Tensor]:
    """
    Concatenate a dictionary of tensors along a specified dimension.

    Args:
        *dicts: Dictionaries of tensors
        dim: Dimension to concatenate along

    Returns:
        A dictionary of tensors concatenated along the specified dimension
    """
    return {k: torch.cat([d[k] for d in dicts], dim=dim) for k in dicts[0].keys()}
