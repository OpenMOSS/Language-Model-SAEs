from enum import Enum
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    Iterable,
    Iterator,
    Optional,
    Sequence,
)

import numpy as np
import torch
from datasets import Dataset
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainSerializer,
    WithJsonSchema,
)

from lm_saes.backend.language_model import LanguageModel
from lm_saes.config import BaseConfig
from lm_saes.utils.misc import (
    convert_str_to_torch_dtype,
    convert_torch_dtype_to_str,
)

from .processors.activation import (
    ActivationBatchler,
    ActivationGenerator,
    ActivationTransformer,
    BufferShuffleConfig,
    OverrideDtypeProcessor,
)
from .processors.cached_activation import CachedActivationLoader
from .processors.core import BaseActivationProcessor
from .processors.huggingface import HuggingFaceDatasetLoader


class ActivationFactorySource(BaseModel):
    type: str
    """ The type of the source. Used to determine the source of activations in deserialization. """
    name: str
    sample_weights: float = 1.0
    """ The sample weights to use for the source. Will be used to randomly pull tokens from the source when multiple sources are present. """


class ActivationFactoryDatasetSource(ActivationFactorySource):
    type: str = "dataset"
    is_dataset_tokenized: bool = False
    """ Whether the dataset is tokenized. Non-tokenized datasets should have records with fields `text`, `images`, etc. Tokenized datasets should have records with fields `tokens`, which could contain either padded or non-padded tokens. """
    prepend_bos: bool = True
    """ Whether to prepend the BOS token to each record when tokenizing. """


class ActivationFactoryActivationsSource(ActivationFactorySource):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # allow parsing torch.dtype

    type: str = "activations"
    path: str | dict[str, str]
    """ The path to the cached activations. """
    device: str = "cpu"
    """ The device to load the activations on. """
    dtype: Optional[
        Annotated[
            torch.dtype,
            BeforeValidator(lambda v: convert_str_to_torch_dtype(v) if isinstance(v, str) else v),
            PlainSerializer(convert_torch_dtype_to_str),
            WithJsonSchema(
                {
                    "type": "string",
                },
                mode="serialization",
            ),
        ]
    ] = None
    """ We might want to convert presaved bf16 activations to fp32"""
    num_workers: int = 4
    """ The number of workers to use for loading the activations. """
    prefetch: int | None = 8
    """ The number of chunks to prefetch."""


class ActivationFactoryTarget(Enum):
    TOKENS = "tokens"
    """ Output non-padded and non-truncated tokens. """
    ACTIVATIONS_2D = "activations-2d"
    """ Output activations in `(batch_size, seq_len, d_model)` shape. Tokens are padded and truncated to the same length. """
    ACTIVATIONS_1D = "activations-1d"
    """ Output activations in `(n_filtered_tokens, d_model)` shape. Tokens are filtered in this stage. """

    @property
    def stage(self) -> int:
        return {
            ActivationFactoryTarget.TOKENS: 0,
            ActivationFactoryTarget.ACTIVATIONS_2D: 1,
            ActivationFactoryTarget.ACTIVATIONS_1D: 2,
        }[self]

    def __lt__(self, other: "ActivationFactoryTarget") -> bool:
        return self.stage < other.stage

    def __le__(self, other: "ActivationFactoryTarget") -> bool:
        return self.stage <= other.stage


class ActivationFactoryConfig(BaseConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # allow parsing torch.dtype

    sources: list[ActivationFactoryDatasetSource | ActivationFactoryActivationsSource]
    """ List of sources to use for activations. Can be a dataset or a path to activations. """
    target: ActivationFactoryTarget
    """ The target to produce. """
    hook_points: list[str]
    """ The hook points to capture activations from. """
    batch_size: int
    """ The batch size to use for outputting activations. """
    num_workers: int = 4
    """ The number of workers to use for loading the dataset. """
    context_size: int | None = None
    """ The context size to use for generating activations. All tokens will be padded or truncated to this size. If `None`, will not pad or truncate tokens. This may lead to some error when re-batching activations of different context sizes."""
    model_batch_size: int = 1
    """ The batch size to use for model forward pass when generating activations. """
    override_dtype: Optional[
        Annotated[
            torch.dtype,
            BeforeValidator(lambda v: convert_str_to_torch_dtype(v) if isinstance(v, str) else v),
            PlainSerializer(convert_torch_dtype_to_str),
            WithJsonSchema(
                {
                    "type": "string",
                },
                mode="serialization",
            ),
        ]
    ] = None
    """ The dtype to use for outputting activations. If `None`, will not override the dtype. """
    buffer_size: int | None = None
    """ Buffer size for online shuffling. If `None`, no shuffling will be performed. """
    buffer_shuffle: BufferShuffleConfig | None = None
    """" Manual seed and device of generator for generating randomperm in buffer. """
    ignore_token_ids: list[int] | None = None
    """ Tokens to ignore in the activations. """


class ActivationFactory:
    """Factory class for generating activation data from different sources.

    This class handles loading data from datasets or activation files, processing it through
    a pipeline of processors, and aggregating the results based on configured weights.

    The overall pipeline is like a tree, where multiple chains collect data from different sources,
    and then aggregated together, which in detail is:
    1. Pre-aggregation processors: Process data from each source through a series of processors.
    2. Aggregator: Aggregate the processed data streams.
    3. Post-aggregation processor: Process the aggregated data through a final processor.
    """

    def __init__(
        self,
        cfg: ActivationFactoryConfig,
        before_aggregation_interceptor: Callable[[dict[str, Any], int], dict[str, Any]] | None = None,
        device_mesh: Optional[Any] = None,
    ):
        """Initialize the factory with the given configuration.

        Args:
            cfg: Configuration object specifying data sources, processing pipeline and output format
        """
        self.cfg = cfg
        self.device_mesh = device_mesh

        self.pre_aggregation_processors = self.build_pre_aggregation_processors()
        self.post_aggregation_processor = self.build_post_aggregation_processor()
        self.aggregator = self.build_aggregator()
        self.before_aggregation_interceptor = before_aggregation_interceptor

    def _build_pre_aggregation_dataset_source_processors(
        self, dataset_source: ActivationFactoryDatasetSource, source_idx: int
    ):
        loader = HuggingFaceDatasetLoader(
            batch_size=1,
            with_info=True,
            show_progress=True,
            num_workers=self.cfg.num_workers,
        )

        processors_optional: Sequence[
            BaseActivationProcessor[Iterable[dict[str, Any]], Iterable[dict[str, Any]]] | None
        ] = [
            ActivationGenerator(
                hook_points=self.cfg.hook_points, batch_size=self.cfg.model_batch_size, n_context=self.cfg.context_size
            )
            if self.cfg.target >= ActivationFactoryTarget.ACTIVATIONS_2D
            else None,
            ActivationTransformer() if self.cfg.target >= ActivationFactoryTarget.ACTIVATIONS_1D else None,
        ]

        # Create processors up to required stage
        processors: Sequence[BaseActivationProcessor[Iterable[dict[str, Any]], Iterable[dict[str, Any]]]] = [
            processor for processor in processors_optional if processor is not None
        ]

        def process_dataset(**kwargs: Any):
            """Process a single dataset through the pipeline.

            Args:
                **kwargs: Must contain 'datasets' dict and 'model' transformer

            Returns:
                Stream of processed data
            """
            datasets: dict[str, tuple[Dataset, Optional[dict[str, Any]]]] | None = kwargs.get("datasets")
            assert datasets is not None, "`datasets` must be provided for dataset sources"
            model: LanguageModel | None = kwargs.get("model")
            assert model is not None, "`model` must be provided for dataset sources"
            model_name: str | None = kwargs.get("model_name")
            assert model_name is not None, "`model_name` must be provided for dataset sources"

            dataset = datasets.get(dataset_source.name)
            assert dataset is not None, f"Dataset {dataset_source.name} not found in `datasets`"
            dataset, metadata = dataset

            stream = loader.process(dataset, dataset_name=dataset_source.name, metadata=metadata)

            for processor in processors:
                stream = processor.process(
                    stream, model=model, model_name=model_name, ignore_token_ids=self.cfg.ignore_token_ids
                )

            if self.before_aggregation_interceptor is not None:
                before_aggregation_interceptor = self.before_aggregation_interceptor  # capture the function

                def interceptor(x: dict[str, Any]):
                    return before_aggregation_interceptor(x, source_idx)

                stream = map(interceptor, stream)

            return stream

        return process_dataset

    def _build_pre_aggregation_activations_source_processors(
        self, activations_source: ActivationFactoryActivationsSource, source_idx: int
    ):
        if self.cfg.target < ActivationFactoryTarget.ACTIVATIONS_2D:
            raise ValueError("Activations sources are only supported for target >= ACTIVATIONS_2D")

        cache_dirs = (
            activations_source.path
            if isinstance(activations_source.path, dict)
            else {hook_point: Path(activations_source.path) / hook_point for hook_point in self.cfg.hook_points}
        )

        loader = CachedActivationLoader(
            cache_dirs=cache_dirs,
            device=activations_source.device,
            dtype=activations_source.dtype,
            num_workers=activations_source.num_workers,
            prefetch_factor=activations_source.prefetch,
            device_mesh=self.device_mesh,
        )

        processors = [ActivationTransformer()] if self.cfg.target >= ActivationFactoryTarget.ACTIVATIONS_1D else []

        def process_activations(**kwargs: Any):
            """Process a single activations source through the pipeline.

            Args:
                **kwargs: Could contain 'model' for providing tokenizer info for special tokens filtering

            Returns:
                Stream of processed data
            """
            model: LanguageModel | None = kwargs.get("model")

            stream = loader.process()
            for processor in processors:
                stream = processor.process(stream, ignore_token_ids=self.cfg.ignore_token_ids, model=model)

            if self.before_aggregation_interceptor is not None:
                before_aggregation_interceptor = self.before_aggregation_interceptor  # capture the function

                def interceptor(x: dict[str, Any]):
                    return before_aggregation_interceptor(x, source_idx)

                stream = map(interceptor, stream)

            return stream

        return process_activations

    def build_pre_aggregation_processors(self):
        """Build processors that run before aggregation for each data source.

        Returns:
            List of callables that process data from each source
        """
        # Split sources by type
        dataset_sources = [source for source in self.cfg.sources if isinstance(source, ActivationFactoryDatasetSource)]
        activations_sources = [
            source for source in self.cfg.sources if isinstance(source, ActivationFactoryActivationsSource)
        ]

        pre_aggregation_processors = [
            self._build_pre_aggregation_dataset_source_processors(source, i) for i, source in enumerate(dataset_sources)
        ] + [
            self._build_pre_aggregation_activations_source_processors(source, i + len(dataset_sources))
            for i, source in enumerate(activations_sources)
        ]
        return pre_aggregation_processors

    def build_post_aggregation_processor(self):
        """Build processor that runs after aggregation.

        Args:
            cfg: Factory configuration object

        Returns:
            Callable that processes aggregated data
        """

        def build_batchler():
            """Create batchler for batched activations."""
            assert self.cfg.batch_size is not None, "Batch size must be provided for outputting batched activations"
            return ActivationBatchler(
                batch_size=self.cfg.batch_size,
                buffer_size=self.cfg.buffer_size,
                buffer_shuffle_config=self.cfg.buffer_shuffle,
                device_mesh=self.device_mesh,
            )

        def build_override_dtype_processor():
            """Create processor that overrides the dtype of the activations."""
            assert self.cfg.override_dtype is not None, (
                "Override dtype must be provided for outputting activations with different dtype"
            )
            return OverrideDtypeProcessor(dtype=self.cfg.override_dtype)

        processors = []
        if self.cfg.batch_size is not None:
            processors.append(build_batchler())
        if self.cfg.override_dtype is not None:
            processors.append(build_override_dtype_processor())

        def process_activations(activations: Iterable[dict[str, Any]], **kwargs: Any):
            """Process aggregated activations through post-processors.

            Args:
                activations: Stream of aggregated activation data
                **kwargs: Additional arguments passed to processors

            Returns:
                Processed activation stream
            """
            for processor in processors:
                activations = processor.process(activations, **kwargs)
            return activations

        return process_activations

    def build_aggregator(self):
        """Build function to aggregate data from multiple sources.

        Returns:
            Callable that aggregates data streams. Currently is a simple weighted random sampler.
        """
        source_sample_weights = np.array([source.sample_weights for source in self.cfg.sources])

        def aggregate(activations: list[Iterable[dict[str, Any]]], **kwargs: Any) -> Iterable[dict[str, Any]]:
            """Aggregate multiple activation streams by sampling based on weights.

            Args:
                activations: List of activation streams from different sources
                **kwargs: Additional arguments (unused)

            Yields:
                Sampled activation data with source info
            """
            ran_out_of_samples = np.zeros(len(self.cfg.sources), dtype=bool)
            activations: list[Iterator[dict[str, Any]]] = [iter(activation) for activation in activations]
            # Mask out sources run out of samples
            weights = source_sample_weights[~ran_out_of_samples]
            weights = weights / weights.sum()

            while not all(ran_out_of_samples):
                sampled_sources = np.random.choice(len(activations), replace=True, p=weights)
                try:
                    result = next(activations[sampled_sources])
                except StopIteration:
                    ran_out_of_samples[sampled_sources] = True
                    continue
                yield result

        return aggregate

    def process(self, **kwargs: Any):
        """Process data through the full pipeline.

        Args:
            **kwargs: Arguments passed to processors (must include required args)

        Returns:
            Iterable of processed activation data
        """
        streams = [processor(**kwargs) for processor in self.pre_aggregation_processors]
        stream = self.aggregator(streams)
        return self.post_aggregation_processor(stream, **kwargs)
