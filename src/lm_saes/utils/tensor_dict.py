from typing import Any, TypeVar, overload

import torch
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
        assert isinstance(tensor_dict[sort_key], DTensor), "tensor_dict[sort_key] must be a DTensor"
        sorted_idx_local = tensor_dict[sort_key].to_local().argsort(dim=sort_dim, descending=descending)
        for k, v in tensor_dict.items():
            assert isinstance(v, DTensor), "v must be a DTensor"
            v_local = v.to_local()
            tmp_sorted_idx = sorted_idx_local
            while v_local.ndim > tmp_sorted_idx.ndim:
                tmp_sorted_idx = tmp_sorted_idx.unsqueeze(-1)
            tensor_dict[k] = DTensor.from_local(
                local_tensor=v_local.gather(sort_dim, tmp_sorted_idx.expand_as(v_local)),
                device_mesh=device_mesh,
                placements=v.placements,
            )
    else:
        sorted_idx = tensor_dict[sort_key].argsort(dim=sort_dim, descending=descending)
        for k, v in tensor_dict.items():
            tmp_sorted_idx = sorted_idx
            while v.ndim > tmp_sorted_idx.ndim:
                tmp_sorted_idx = tmp_sorted_idx.unsqueeze(-1)
            tensor_dict[k] = v.gather(sort_dim, tmp_sorted_idx.expand_as(v))

        # sort_tensor = tensor_dict[sort_key]
        
        # # If sort_tensor has more dimensions than just sort_dim, we need to aggregate
        # # For example, if sort_tensor is [batch_size, d_sae] and sort_dim=0,
        # # we should aggregate over d_sae to get [batch_size] before sorting
        # if sort_tensor.ndim > 1:
        #     # Aggregate over all dimensions except sort_dim
        #     # Use max aggregation to get the maximum value for each element along sort_dim
        #     other_dims = [i for i in range(sort_tensor.ndim) if i != sort_dim]
        #     if other_dims:
        #         # Aggregate over all other dimensions
        #         sort_values = sort_tensor.max(dim=tuple(other_dims), keepdim=False).values
        #     else:
        #         sort_values = sort_tensor
        # else:
        #     sort_values = sort_tensor
        
        # # Now sort_values should have shape matching sort_dim
        # # Get the sorted indices along sort_dim
        # sorted_idx = sort_values.argsort(dim=sort_dim, descending=descending)
        
        # # Apply the sorting to all tensors in the dictionary
        # for k, v in tensor_dict.items():
        #     # Ensure sorted_idx has compatible dimensions with v
        #     tmp_sorted_idx = sorted_idx
        #     # If v has more dimensions than sorted_idx, unsqueeze sorted_idx
        #     while v.ndim > tmp_sorted_idx.ndim:
        #         tmp_sorted_idx = tmp_sorted_idx.unsqueeze(-1)
        #     # If sorted_idx has more dimensions than v, we need to handle it
        #     # This can happen if sort_key was multidimensional but we aggregated it
        #     # In this case, we should use the first slice along extra dimensions
        #     while tmp_sorted_idx.ndim > v.ndim:
        #         # Take the first slice along the first extra dimension
        #         extra_dims = [i for i in range(tmp_sorted_idx.ndim) if i != sort_dim]
        #         if extra_dims:
        #             tmp_sorted_idx = tmp_sorted_idx.select(extra_dims[0], 0)
        #         else:
        #             break
            
        #     # Expand sorted_idx to match v's shape for gather
        #     # Gather requires indices to match the shape except at sort_dim
        #     if tmp_sorted_idx.shape[sort_dim] == v.shape[sort_dim]:
        #         # Expand sorted_idx to match v's shape
        #         expand_shape = list(v.shape)
        #         expand_shape[sort_dim] = -1  # Keep original size at sort_dim
        #         tmp_sorted_idx = tmp_sorted_idx.expand(expand_shape)
        #         tensor_dict[k] = v.gather(sort_dim, tmp_sorted_idx)
        #     else:
        #         # If dimensions don't match, this is an error case
        #         raise ValueError(
        #             f"Cannot sort tensor {k} with shape {v.shape} using indices "
        #             f"with shape {sorted_idx.shape} at sort_dim {sort_dim}"
        #         )
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
