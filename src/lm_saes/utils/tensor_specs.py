import torch


class TensorSpecs:
    """Infer the specs of a tensor.

    Specs are a tuple of dimension names. Length of the tuple should match the number of dimensions of the tensor.
    """

    @staticmethod
    def feature_acts(tensor: torch.Tensor) -> tuple[str, ...]:
        if tensor.ndim == 2:
            return ("batch", "sae")
        elif tensor.ndim == 3:
            return ("batch", "context", "sae")
        else:
            raise ValueError(f"Cannot infer tensor specs for tensor with {tensor.ndim} dimensions.")

    @staticmethod
    def reconstructed(tensor: torch.Tensor) -> tuple[str, ...]:
        if tensor.ndim == 2:
            return ("batch", "model")
        elif tensor.ndim == 3:
            return ("batch", "context", "model")
        else:
            raise ValueError(f"Cannot infer tensor specs for tensor with {tensor.ndim} dimensions.")

    @staticmethod
    def label(tensor: torch.Tensor) -> tuple[str, ...]:
        return TensorSpecs.reconstructed(tensor)
