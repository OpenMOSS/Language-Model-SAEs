"""Protocol definitions for sparse dictionary capabilities.

These protocols define optional interfaces that sparse dictionary implementations
can adopt. Use ``isinstance`` checks (enabled by ``@runtime_checkable``) to guard
calls to protocol methods.
"""

from typing import Protocol, runtime_checkable

import torch
from torch.distributed.tensor import DTensor


@runtime_checkable
class NormComputing(Protocol):
    """Protocol for weight norm computation.

    Implementors provide ``encoder_norm``, ``decoder_norm``, and ``decoder_bias_norm``
    to compute the L2 norms of their respective weight matrices.
    """

    def encoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Compute the L2 norm of encoder weight rows."""
        ...

    def decoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Compute the L2 norm of decoder weight columns."""
        ...

    def decoder_bias_norm(self) -> torch.Tensor:
        """Compute the L2 norm of the decoder bias."""
        ...

    def decoder_norm_full(self, keepdim: bool = False) -> torch.Tensor:
        """Compute the full norm of the decoder. Converts DTensor to a full tensor if needed."""
        decoder_norm = self.decoder_norm(keepdim=keepdim)
        if not isinstance(decoder_norm, DTensor):
            return decoder_norm
        else:
            return decoder_norm.full_tensor()


@runtime_checkable
class NormConstrainable(Protocol):
    """Protocol for weight norm constraint capabilities.

    Implementors provide methods to set encoder/decoder weights to fixed norms
    and to transform the model to have unit decoder norm.
    """

    def set_decoder_to_fixed_norm(self, value: float, force_exact: bool) -> None:
        """Set the decoder to a fixed norm."""
        ...

    def set_encoder_to_fixed_norm(self, value: float) -> None:
        """Set the encoder to a fixed norm."""
        ...

    def transform_to_unit_decoder_norm(self) -> None:
        """Transform the model to have unit decoder norm."""
        ...


@runtime_checkable
class DatasetNormStandardizable(Protocol):
    """Protocol for dataset-wise norm standardization.

    Implementors fold the dataset-wise average activation norm into the model's
    weights and biases for inference.
    """

    def standardize_parameters_of_dataset_norm(self) -> None:
        """Standardize the parameters of the model to account for dataset_norm during inference."""
        ...


@runtime_checkable
class EncoderInitializable(Protocol):
    """Protocol for encoder initialization from decoder transpose.

    Implementors initialize the encoder weight matrix as the transpose of the
    decoder weight matrix.
    """

    def init_encoder_with_decoder_transpose(self, factor: float = 1.0) -> None:
        """Initialize the encoder with the transpose of the decoder."""
        ...


@runtime_checkable
class ActiveSubspaceInitializable(Protocol):
    """Protocol for decoder initialization from the active subspace of activations.

    Implementors initialize the decoder weight matrix using the principal components
    of the activation distribution.
    """

    def init_W_D_with_active_subspace(self, batch: dict[str, torch.Tensor], d_active_subspace: int) -> None:
        """Initialize the decoder with the active subspace of the activation distribution."""
        ...


@runtime_checkable
class EncoderBiasInitializable(Protocol):
    """Protocol for encoder bias initialization from pre-activation statistics.

    Implementors initialize the encoder bias to the negative mean of the
    pre-activation distribution.
    """

    def init_encoder_bias_with_mean_hidden_pre(self, batch: dict[str, torch.Tensor]) -> None:
        """Initialize the encoder bias with the negative mean of the pre-activation hidden states."""
        ...
