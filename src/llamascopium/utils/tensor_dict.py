from typing import Any, TypeVar, cast, overload

import torch
from einops import repeat
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

T = TypeVar("T")


def sort_dict_of_tensor(
    tensor_dict: dict[str, torch.Tensor],
    sort_key: str,
    sort_dim: int = 0,
    descending: bool = True,
    device_mesh: DeviceMesh | None = None,
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
    if device_mesh is not None:
        assert isinstance(tensor_dict[sort_key], DTensor), (
            "All tensors to sort must be DTensor when device_mesh is provided"
        )
        sorted_idx_local = cast(DTensor, tensor_dict[sort_key]).to_local().argsort(dim=sort_dim, descending=descending)
        for k, v in tensor_dict.items():
            assert isinstance(v, DTensor), "All tensors to sort must be DTensor when device_mesh is provided"
            v_local = v.to_local()
            tensor_dict[k] = DTensor.from_local(
                local_tensor=v_local.gather(
                    sort_dim,
                    repeat(
                        sorted_idx_local, f"... -> ... {' '.join(['1'] * (v_local.ndim - sorted_idx_local.ndim))}"
                    ).expand_as(v_local),
                ),
                device_mesh=device_mesh,
                placements=v.placements,
            )
    else:
        sorted_idx = tensor_dict[sort_key].argsort(dim=sort_dim, descending=descending)
        for k, v in tensor_dict.items():
            tensor_dict[k] = v.gather(
                sort_dim, repeat(sorted_idx, f"... -> ... {' '.join(['1'] * (v.ndim - sorted_idx.ndim))}").expand_as(v)
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
