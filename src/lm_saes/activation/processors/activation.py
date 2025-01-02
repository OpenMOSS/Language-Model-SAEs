import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Optional, cast

import torch
from transformer_lens import HookedTransformer

from lm_saes.activation.processors.core import BaseActivationProcessor


class ActivationGenerator(BaseActivationProcessor[Iterable[dict[str, Any]], Iterable[dict[str, Any]]]):
    """Processor for extracting model activations at specified hook points.

    This processor takes an iterable of dictionaries containing tokens and runs them through
    a model to extract activations at specified hook points. The output is a dictionary containing
    the activations at each hook point, the original tokens as "context", and any info field
    from the input.

    Args:
        hook_points (list[str]): List of hook point names to extract activations from
    """

    def __init__(self, hook_points: list[str]):
        self.hook_points = hook_points

    def process(
        self, data: Iterable[dict[str, Any]], *, model: HookedTransformer, **kwargs
    ) -> Iterable[dict[str, Any]]:
        """Process tokens to extract model activations.

        Args:
            data (Iterable[dict[str, Any]]): Input data containing tokens to process
            model (HookedTransformer): Model to extract activations from
            **kwargs: Additional keyword arguments. Not used by this processor.

        Yields:
            dict[str, Any]: Dictionary containing:
                - Activations for each hook point
                - Original tokens as "tokens"
                - Original info field if present in input
        """
        for d in data:
            assert "tokens" in d and isinstance(d["tokens"], torch.Tensor)
            tokens = d["tokens"]
            _, cache = model.run_with_cache_until(tokens, names_filter=self.hook_points, until=self.hook_points[-1])
            ret = {k: cache[k][0] for k in self.hook_points}
            ret = ret | {"tokens": tokens}
            if "meta" in d:
                ret = ret | {"meta": d["meta"]}
            yield ret


class ActivationTransformer(BaseActivationProcessor[Iterable[dict[str, Any]], Iterable[dict[str, Any]]]):
    """Processor for filtering and transforming model activations.

    This processor takes an iterable of dictionaries containing activations and tokens, and filters out
    specified special tokens (like padding, BOS, EOS). The output contains the filtered activations
    at each hook point, the original tokens, and any info field from the input. The "batch" and "context" dimensions
    will be reshaped in one dimension, i.e. (batch, context, d) -> (filtered_batch_and_context, d).

    Args:
        hook_points (list[str]): List of hook point names to process activations from
    """

    def __init__(self, hook_points: list[str]):
        self.hook_points = hook_points

    def process(
        self,
        data: Iterable[dict[str, Any]],
        *,
        ignore_token_ids: Optional[list[int]] = None,
        model: Optional[HookedTransformer] = None,
        **kwargs,
    ) -> Iterable[dict[str, Any]]:
        """Process activations by filtering out specified token types.

        Args:
            data (Iterable[dict[str, Any]]): Input data containing activations and tokens to process
            ignore_token_ids (Optional[list[int]], optional): List of token IDs to filter out. If None and model
                is provided, uses model's special tokens (EOS, PAD, BOS). Defaults to None.
            model (Optional[HookedTransformer], optional): Model to get default special tokens from. Only used
                if ignore_token_ids is None. Defaults to None.
            **kwargs: Additional keyword arguments. Not used by this processor.

        Yields:
            dict[str, Any]: Dictionary containing:
                - Filtered activations for each hook point
                - Original tokens as "tokens"
                - Original info field if present in input
        """
        if ignore_token_ids is None and model is None:
            warnings.warn(
                "Both ignore_token_ids and model are not provided. No tokens (including pad tokens) will be filtered out. If this is intentional, set ignore_token_ids explicitly to an empty list to avoid this warning.",
                UserWarning,
                stacklevel=2,
            )
        if ignore_token_ids is None and model is not None:
            assert model.tokenizer is not None, "Tokenizer is required to obtain ignore_token_ids"
            ignore_token_ids = [
                cast(int, model.tokenizer.eos_token_id),
                cast(int, model.tokenizer.pad_token_id),
                cast(int, model.tokenizer.bos_token_id),
            ]
        if ignore_token_ids is None:
            ignore_token_ids = []
        for d in data:
            assert "tokens" in d and isinstance(d["tokens"], torch.Tensor)
            tokens = d["tokens"]
            mask = torch.ones_like(tokens, dtype=torch.bool)
            for token_id in ignore_token_ids:
                mask &= tokens != token_id
            activations = {k: d[k][mask] for k in self.hook_points}
            activations = activations | {"tokens": tokens}
            if "meta" in d:
                activations = activations | {"meta": d["meta"]}
            yield activations


def shuffle_activations(activations: dict[str, torch.Tensor], hook_points: list[str]) -> dict[str, torch.Tensor]:
    assert all(isinstance(activations[k], torch.Tensor) for k in hook_points)
    assert all(activations[k].shape == activations[hook_points[0]].shape for k in hook_points)
    perm = torch.randperm(activations[hook_points[0]].shape[0])
    return {k: v[perm] for k, v in activations.items()}


@dataclass
class ActivationBuffer:
    """Buffer for storing and manipulating activation tensors.

    This class provides functionality to store activations from multiple hook points,
    concatenate new activations, yield batches, and shuffle the stored data. All operations
    are performed out-of-place.

    Args:
        hook_points (list[str]): List of hook point names to track
        buffer (dict[str, torch.Tensor] | None, optional): Initial buffer of activations. Defaults to None.
    """

    hook_points: list[str]
    buffer: dict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        """Get the number of samples in the buffer.

        Returns:
            int: Number of samples, or 0 if buffer is empty
        """
        if self.buffer is None:
            return 0
        return self.buffer[self.hook_points[0]].shape[0]

    def cat(self, activations: dict[str, torch.Tensor]) -> "ActivationBuffer":
        """Concatenate new activations to the buffer.

        Args:
            activations (dict[str, torch.Tensor]): New activations to add

        Returns:
            ActivationBuffer: New buffer containing concatenated activations
        """
        if self.buffer is None:
            buffer = {k: activations[k] for k in self.hook_points}
        else:
            buffer = {k: torch.cat([self.buffer[k], activations[k]], dim=0) for k in self.hook_points}
        return ActivationBuffer(hook_points=self.hook_points, buffer=buffer)

    def yield_batch(self, batch_size: int) -> tuple[dict[str, torch.Tensor], "ActivationBuffer"]:
        """Extract a batch of samples from the buffer.

        Args:
            batch_size (int): Number of samples to extract

        Returns:
            tuple[dict[str, torch.Tensor], ActivationBuffer]: Tuple containing:
                - Dictionary of extracted batch activations
                - New buffer with remaining samples

        Raises:
            ValueError: If buffer is empty
        """
        if self.buffer is None:
            raise ValueError("Buffer is empty")
        batch = {k: v[:batch_size] for k, v in self.buffer.items()}
        buffer = {k: v[batch_size:] for k, v in self.buffer.items()}
        return batch, ActivationBuffer(hook_points=self.hook_points, buffer=buffer)

    def shuffle(self) -> "ActivationBuffer":
        """Randomly shuffle all samples in the buffer.

        Returns:
            ActivationBuffer: New buffer with shuffled samples

        Raises:
            ValueError: If buffer is empty
        """
        if self.buffer is None:
            raise ValueError("Buffer is empty")
        perm = torch.randperm(self.buffer[self.hook_points[0]].shape[0])
        buffer = {k: v[perm] for k, v in self.buffer.items()}
        return ActivationBuffer(hook_points=self.hook_points, buffer=buffer)


class ActivationBatchler(BaseActivationProcessor[Iterable[dict[str, Any]], Iterable[dict[str, Any]]]):
    """Processor for batching activations.

    This processor takes an iterable of dictionaries containing activations, and batches them.
    The input activations are expected to be of shape (arbitary, d), and the output activations
    will be of shape (batch_size, d). Also, this processor supports performing online shuffling.

    Additional fields, including "tokens" and "meta", will be removed.

    Args:
        hook_points (list[str]): List of hook point names to process
        batch_size (int): Number of samples per batch
        buffer_size (Optional[int], optional): Size of the buffer to perform online shuffling. If specified,
            data will be refilled into the buffer whenever the buffer is less than half full, and then re-shuffled.
    """

    def __init__(self, hook_points: list[str], batch_size: int, buffer_size: Optional[int] = None):
        """Initialize the ActivationBatchler.

        Args:
            hook_points (list[str]): List of hook point names to process
            batch_size (int): Number of samples per batch
            buffer_size (Optional[int], optional): Size of buffer for online shuffling. Defaults to None.
        """
        self.hook_points = hook_points
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def process(self, data: Iterable[dict[str, Any]], **kwargs) -> Iterable[dict[str, Any]]:
        """Process input data by batching activations.

        Takes an iterable of activation dictionaries and yields batches of activations.
        If buffer_size is specified, performs online shuffling by maintaining a buffer
        of samples that gets shuffled whenever it's full.

        Args:
            data (Iterable[dict[str, Any]]): Input iterable of activation dictionaries
            **kwargs: Additional keyword arguments (unused)

        Yields:
            dict[str, Any]: Batched activations

        Raises:
            AssertionError: If hook points are missing or tensors have invalid shapes
        """
        buffer = ActivationBuffer(hook_points=self.hook_points)
        for d in data:
            # Validate input: ensure all hook points exist and are 2D tensors
            assert all(
                (k in d and isinstance(d[k], torch.Tensor) and len(d[k].shape) == 2) for k in self.hook_points
            ), "All hook points must be present and be 2D tensors"

            # Validate input: ensure all tensors have consistent shapes
            assert all(
                d[k].shape == d[self.hook_points[0]].shape for k in self.hook_points
            ), "All tensors must have the same shape"

            # Add new data to buffer
            buffer = buffer.cat(d)

            if self.buffer_size is not None:
                # If buffer is full, shuffle and yield batches until half empty
                if len(buffer) >= self.buffer_size:
                    buffer = buffer.shuffle()
                    while len(buffer) >= self.buffer_size // 2 and len(buffer) >= self.batch_size:
                        # I have no idea why the buffer is Never in the while block and I need to cast it to ActivationBuffer
                        # Perhaps this is a bug with basedpyright
                        batch, buffer = cast(ActivationBuffer, buffer).yield_batch(self.batch_size)
                        yield batch
            else:
                # If no buffer size specified, yield complete batches as they become available
                while len(buffer) >= self.batch_size:
                    # The same issue as above
                    batch, buffer = cast(ActivationBuffer, buffer).yield_batch(self.batch_size)
                    yield batch

        # Yield any remaining samples in batches
        while len(buffer) > 0:
            batch, buffer = buffer.yield_batch(self.batch_size)
            yield batch
