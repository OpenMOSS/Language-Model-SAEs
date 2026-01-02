from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, cast

import torch
from more_itertools import batched
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map
from tqdm import tqdm

from lm_saes.backend.language_model import LanguageModel
from lm_saes.config import BaseConfig
from lm_saes.utils.distributed import DimMap, mesh_dim_size

from .core import BaseActivationProcessor


class BufferShuffleConfig(BaseConfig):
    perm_seed: int = 42
    """ Perm seed for aligned permutation for generating activations. If `None`, will not use manual seed for Generator. """
    generator_device: str | None = None
    """ The device to be assigned for the torch.Generator. If 'None', generator will be initialized on cpu as pytorch default. """


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

    buffer: list[dict[str, Any]] = field(default_factory=list)
    device_mesh: Optional[DeviceMesh] = None
    generator: torch.Generator = torch.Generator()  # Generator passed from ActivationBatchler

    def __len__(self) -> int:
        """Get the number of samples in the buffer.

        Returns:
            int: Number of samples, or 0 if buffer is empty
        """
        return sum(len(list(d.values())[0]) for d in self.buffer)

    def cat(self, activations: dict[str, Any]) -> "ActivationBuffer":
        """Concatenate new activations to the buffer.

        Args:
            activations (dict[str, torch.Tensor]): New activations to add

        Returns:
            ActivationBuffer: New buffer containing concatenated activations
        """
        return ActivationBuffer(
            buffer=self.buffer + [activations], generator=self.generator, device_mesh=self.device_mesh
        )

    def consume(self) -> dict[str, torch.Tensor | list[Any]]:
        """Consume the buffer and return the activations as a dictionary."""

        def _cat(xs: list[torch.Tensor]) -> torch.Tensor:
            """Concatenate a list of tensors. Mainly for distributed setting.
            For non-distributed setting, this is just torch.cat(xs, dim=0).
            """

            if self.device_mesh is not None:
                assert all(isinstance(x, DTensor) and x.device_mesh == self.device_mesh for x in xs)
                xs_local = [cast(DTensor, x).to_local() for x in xs]
            else:
                xs_local = xs

            if len(xs_local) == 1:
                catted = xs_local[0]
            else:
                catted = torch.cat(xs_local, dim=0)

            if self.device_mesh is not None:
                return DTensor.from_local(
                    catted, device_mesh=self.device_mesh, placements=cast(DTensor, xs[0]).placements
                )
            else:
                return catted

        return {
            k: _cat([d[k] for d in self.buffer])
            if isinstance(self.buffer[0][k], torch.Tensor)
            else sum([d[k] for d in self.buffer], [])
            for k in self.buffer[0].keys()
        }

    def yield_batch(self, batch_size: int) -> tuple[dict[str, torch.Tensor | list[Any]], "ActivationBuffer"]:
        """Extract a batch of samples from the buffer.

        Args:
            batch_size (int): Number of samples to extract

        Returns:
            tuple[dict[str, torch.Tensor], ActivationBuffer]: Tuple containing:
                - Dictionary of extracted batch activations
                - New buffer with remaining samples
        """
        if self.__len__() == 0:
            raise ValueError("Buffer is empty")
        data = self.consume()

        def _split(
            x: torch.Tensor | list[Any], batch_size: int
        ) -> tuple[torch.Tensor | list[Any], torch.Tensor | list[Any]]:
            """Split the tensor into a batch and a buffer. Mainly for distributed setting.
            For non-distributed setting, this is just x[:batch_size] and x[batch_size:].
            """

            dp_size = mesh_dim_size(self.device_mesh, "data")
            local_batch_size = batch_size // dp_size  # For non-distributed setting, this is just batch_size

            if self.device_mesh is not None and isinstance(x, DTensor):
                assert (
                    x.device_mesh == self.device_mesh
                    and DimMap.from_placements(x.placements, self.device_mesh).to_dict().get("data", 0) == 0
                )
                assert batch_size % dp_size == 0, "Batch size must be divisible by data parallel size"

                x_local = x.to_local()
                batch_tensor = x_local[:local_batch_size]
                buffer_tensor = x_local[local_batch_size:]

                # Convert back to DTensor with same placements
                batch_dtensor = DTensor.from_local(batch_tensor, device_mesh=x.device_mesh, placements=x.placements)
                buffer_dtensor = DTensor.from_local(buffer_tensor, device_mesh=x.device_mesh, placements=x.placements)
                return batch_dtensor, buffer_dtensor
            else:
                return x[:local_batch_size], x[local_batch_size:]

        splitted = {k: _split(v, batch_size) for k, v in data.items()}
        batch = {k: v[0] for k, v in splitted.items()}
        buffer = {k: v[1] for k, v in splitted.items()}
        return batch, ActivationBuffer(buffer=[buffer], generator=self.generator, device_mesh=self.device_mesh)

    def shuffle(self) -> "ActivationBuffer":
        """Randomly shuffle all samples in the buffer.

        Returns:
            ActivationBuffer: New buffer with shuffled samples
        """
        assert self.device_mesh is None, "Shuffling is not supported for distributed setting"

        data = self.consume()
        assert all(isinstance(data[k], torch.Tensor) for k in data.keys()), (
            "All data must be tensors to perform shuffling"
        )
        data = cast(dict[str, torch.Tensor], data)

        # Use the passed generator for shuffling
        perm = torch.randperm(
            data[list(data.keys())[0]].shape[0], generator=self.generator, device=self.generator.device
        )
        buffer = {k: v[perm] for k, v in data.items()}
        return ActivationBuffer(buffer=[buffer], generator=self.generator, device_mesh=self.device_mesh)


class ActivationGenerator(BaseActivationProcessor[Iterable[dict[str, Any]], Iterable[dict[str, Any]]]):
    """Processor for extracting model activations at specified hook points.

    This processor takes an iterable of dictionaries containing tokens and runs them through
    a model to extract activations at specified hook points. The output is a dictionary containing
    the activations at each hook point, the original tokens as "context", and any info field
    from the input.

    Args:
        hook_points (list[str]): List of hook point names to extract activations from
        batch_size (Optional[int], optional): Size of the batch to run through the model at a time.
            If None, will keep the original data structure of the tokens (batched or not). If specified,
            will batch the tokens to the specified size, so if the tokens are already batched, the size
            should be divisible by the original batch size.
    """

    def __init__(self, hook_points: list[str], batch_size: int, n_context: Optional[int] = None):
        self.hook_points = hook_points
        self.batch_size = batch_size
        self.n_context = n_context

    def batched(self, data: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        for d in batched(data, self.batch_size):
            keys = d[0].keys()
            yield {k: [dd[k] for dd in d] for k in keys}

    @torch.no_grad()
    def process(
        self,
        data: Iterable[dict[str, Any]],
        *,
        model: LanguageModel,
        model_name: str,
        **kwargs,
    ) -> Iterable[dict[str, Any]]:
        """Process tokens to extract model activations.

        Args:
            data (Iterable[dict[str, Any]]): Input data containing tokens to process
            model (LanguageModel): Model to extract activations from
            model_name (str): Name of the model. Save to metadata.
            **kwargs: Additional keyword arguments. Not used by this processor.

        Yields:
            dict[str, Any]: Dictionary containing:
                - Activations for each hook point
                - Original tokens as "tokens"
                - Original info field if present in input
        """
        for d in self.batched(data):
            # for specific models like LLaDA, we need to preprocess the raw data, e.g. add mask tokens to the raw[text] and replace the raw[text] with the masked text
            d = model.preprocess_raw_data(d)
            activations = model.to_activations(d, self.hook_points, n_context=self.n_context)
            # merge meta information
            existing_meta = d.get("meta", [{} for _ in range(len(d["text"]))])
            activations = {
                **activations,
                "meta": [{"model_name": model_name} | existing_meta[i] for i in range(len(existing_meta))],
            }
            yield activations


class ActivationTransformer(BaseActivationProcessor[Iterable[dict[str, Any]], Iterable[dict[str, Any]]]):
    """Processor for filtering and transforming model activations.

    This processor takes an iterable of dictionaries containing activations and tokens, and filters out
    specified special tokens (like padding, BOS, EOS). The output contains the filtered activations
    at each hook point, the original tokens, and any info field from the input. The "batch" and "context" dimensions
    will be reshaped in one dimension, i.e. (batch, context, d) -> (filtered_batch_and_context, d).

    Args:
        hook_points (list[str]): List of hook point names to process activations from
    """

    def process(
        self,
        data: Iterable[dict[str, Any]],
        *,
        ignore_token_ids: Optional[list[int]] = None,
        **kwargs,
    ) -> Iterable[dict[str, Any]]:
        """Process activations by filtering out specified token types.

        Args:
            data (Iterable[dict[str, Any]]): Input data containing activations and tokens to process
            ignore_token_ids (Optional[list[int]], optional): List of token IDs to filter out. If None, filter out masked tokens.
            **kwargs: Additional keyword arguments. Not used by this processor.

        Yields:
            dict[str, Any]: Dictionary containing:
                - Filtered activations for each hook point
                - Original tokens as "tokens"
                - Original info field if present in input
        """
        for d in data:
            assert "tokens" in d and isinstance(d["tokens"], torch.Tensor)
            tokens = d["tokens"]

            if ignore_token_ids is not None:
                mask = (
                    cast(
                        torch.Tensor,
                        local_map(
                            lambda x: torch.isin(x, torch.tensor(ignore_token_ids).to(x.device), invert=True),
                            out_placements=DimMap({"data": 0}).placements(tokens.device_mesh),
                        )(tokens),
                    )
                    if isinstance(tokens, DTensor)
                    else torch.isin(tokens, torch.tensor(ignore_token_ids).to(tokens.device), invert=True)
                )
            else:
                mask = d["mask"].bool()

            if isinstance(mask, DTensor):
                # Check if mask is all true
                # TODO: Actually, this assertion is not necessary for tp settings. Remove it in future.
                assert mask.to_local().all(), "Mask must be all true for distributed tensors"
                activations = {k: v for k, v in d.items() if isinstance(v, torch.Tensor)}  # Drop meta
            else:
                activations = {k: v[mask] for k, v in d.items() if isinstance(v, torch.Tensor)}  # Drop meta

            yield activations


def shuffle_activations(activations: dict[str, torch.Tensor], hook_points: list[str]) -> dict[str, torch.Tensor]:
    assert all(isinstance(activations[k], torch.Tensor) for k in hook_points)
    assert all(activations[k].shape == activations[hook_points[0]].shape for k in hook_points)
    perm = torch.randperm(activations[hook_points[0]].shape[0])
    return {k: v[perm] for k, v in activations.items()}


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

    def __init__(
        self,
        batch_size: int,
        buffer_size: Optional[int] = None,
        buffer_shuffle_config: Optional[BufferShuffleConfig] = None,
        device_mesh: Optional[DeviceMesh] = None,
    ):
        """Initialize the ActivationBatchler.

        Args:
            hook_points (list[str]): List of hook point names to process
            batch_size (int): Number of samples per batch
            buffer_size (Optional[int], optional): Size of buffer for online shuffling. Defaults to None.
        """
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device_mesh = device_mesh
        self.perm_generator = torch.Generator()
        if buffer_shuffle_config is not None:
            self.perm_generator = torch.Generator(buffer_shuffle_config.generator_device)
            self.perm_generator.manual_seed(buffer_shuffle_config.perm_seed)  # Set seed if provided

    def process(self, data: Iterable[dict[str, Any]], **kwargs) -> Iterable[dict[str, torch.Tensor]]:
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
        buffer = ActivationBuffer(generator=self.perm_generator, device_mesh=self.device_mesh)
        pbar = tqdm(total=self.buffer_size, desc="Buffer monitor", miniters=1, disable=True)
        dp_size = mesh_dim_size(self.device_mesh, "data")
        for d in data:

            def get_batch_size(x):
                return len(x) if isinstance(x, DTensor) else len(x) * dp_size

            # Validate input: ensure all tensors and lists have consistent shapes
            assert all(get_batch_size(d[k]) == get_batch_size(d[next(iter(d.keys()))]) for k in d.keys()), (
                f"All tensors and lists must have the same batch size, {[(k, len(d[k])) for k in d.keys()]}"
            )

            # Add new data to buffer
            buffer = buffer.cat(d)
            pbar.update(len(buffer) - pbar.n)

            if self.buffer_size is not None:
                # If buffer is full, shuffle and yield batches until half empty
                if len(buffer) >= self.buffer_size:
                    pbar.set_postfix({"Shuffling": True})
                    buffer = buffer.shuffle()
                    pbar.set_postfix({"Shuffling": False})
                    while len(buffer) >= self.buffer_size // 2 and len(buffer) >= self.batch_size:
                        # I have no idea why the buffer is Never in the while block and I need to cast it to ActivationBuffer
                        # Perhaps this is a bug with basedpyright
                        batch, buffer = cast(ActivationBuffer, buffer).yield_batch(self.batch_size)
                        pbar.update(len(buffer) - pbar.n)
                        yield cast(dict[str, torch.Tensor], batch)
            else:
                # If no buffer size specified, yield complete batches as they become available
                while len(buffer) >= self.batch_size:
                    # The same issue as above
                    batch, buffer = cast(ActivationBuffer, buffer).yield_batch(self.batch_size)
                    pbar.update(len(buffer) - pbar.n)
                    yield cast(dict[str, torch.Tensor], batch)

        # Yield any remaining samples in batches
        while len(buffer) > 0:
            batch, buffer = buffer.yield_batch(self.batch_size)
            pbar.update(len(buffer) - pbar.n)
            yield cast(dict[str, torch.Tensor], batch)
        pbar.close()


class OverrideDtypeProcessor(BaseActivationProcessor[Iterable[dict[str, Any]], Iterable[dict[str, Any]]]):
    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype

    def process(self, data: Iterable[dict[str, Any]], **kwargs: Any) -> Iterable[dict[str, Any]]:
        for activation in data:
            for key, value in activation.items():
                if isinstance(value, torch.Tensor):
                    activation[key] = value.to(self.dtype)
            yield activation
