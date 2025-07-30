from typing import Any, TypeVar, overload
import torch

T = TypeVar("T")

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
    for k, v in tensor_dict.items():
        tmp_sorted_idx = sorted_idx
        while v.ndim > tmp_sorted_idx.ndim:
            tmp_sorted_idx = tmp_sorted_idx.unsqueeze(-1)
        tensor_dict[k] = v.gather(
            sort_dim,
            tmp_sorted_idx.expand_as(v)
        ) 
    return tensor_dict


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


@overload
def move_dict_of_tensor_to_device(
    tensor_dict: dict[str, torch.Tensor], device: torch.device | str
) -> dict[str, torch.Tensor]: ...


@overload
def move_dict_of_tensor_to_device(tensor_dict: dict[str, Any], device: torch.device | str) -> dict[str, Any]: ...


def move_dict_of_tensor_to_device(tensor_dict: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    """
    Move tensors in a dictionary to specified device, leaving non-tensor values unchanged.

    Args:
        tensor_dict: Dictionary containing tensors and possibly other types
        device: Target device to move tensors to

    Returns:
        Dictionary with tensors moved to specified device
    """
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in tensor_dict.items()}


def batch_size(tensor_dict: dict[str, torch.Tensor]) -> int:
    """
    Get the batch size of a dictionary of tensors.
    """
    return len(tensor_dict[list(tensor_dict.keys())[0]])
