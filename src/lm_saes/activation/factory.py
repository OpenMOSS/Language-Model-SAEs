from typing import Any, Callable, Iterable, Iterator

import numpy as np
from datasets import Dataset
from transformer_lens import HookedTransformer

from lm_saes.activation.processors.activation import (
    ActivationBatchler,
    ActivationGenerator,
    ActivationTransformer,
)
from lm_saes.activation.processors.core import BaseActivationProcessor
from lm_saes.activation.processors.huggingface import HuggingFaceDatasetLoader
from lm_saes.activation.processors.token import (
    PadAndTruncateTokensProcessor,
    RawDatasetTokenProcessor,
)
from lm_saes.config import (
    ActivationFactoryActivationsSource,
    ActivationFactoryConfig,
    ActivationFactoryDatasetSource,
)


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

    def __init__(self, cfg: ActivationFactoryConfig):
        """Initialize the factory with the given configuration.

        Args:
            cfg: Configuration object specifying data sources, processing pipeline and output format
        """
        self.cfg = cfg
        self.pre_aggregation_processors = self.build_pre_aggregation_processors(cfg)
        self.post_aggregation_processor = self.build_post_aggregation_processor(cfg)
        self.aggregator = self.build_aggregator(cfg)

    @staticmethod
    def build_pre_aggregation_processors(cfg: ActivationFactoryConfig):
        """Build processors that run before aggregation for each data source.

        Args:
            cfg: Factory configuration object

        Returns:
            List of callables that process data from each source
        """
        # Split sources by type
        dataset_sources = [source for source in cfg.sources if isinstance(source, ActivationFactoryDatasetSource)]
        activations_sources = [
            source for source in cfg.sources if isinstance(source, ActivationFactoryActivationsSource)
        ]

        pre_aggregation_processors: list[Callable[..., Iterable[dict[str, Any]]]] = []

        # Build processors for dataset sources
        for dataset_source in dataset_sources:
            loader = HuggingFaceDatasetLoader(
                batch_size=1,
                num_workers=4,
                with_info=True,
                show_progress=True,
            )

            # Map target type to number of processing stages needed
            stage_map = {
                "tokens": 1,
                "activations-2d": 3,
                "activations-1d": 4,
                "batched-activations-1d": 4,
            }

            # Define processor pipeline factories
            processor_factories: list[
                Callable[[], BaseActivationProcessor[Iterable[dict[str, Any]], Iterable[dict[str, Any]]]]
            ] = [
                lambda: RawDatasetTokenProcessor(prepend_bos=dataset_source.prepend_bos),
                lambda: PadAndTruncateTokensProcessor(seq_len=cfg.context_size),
                lambda: ActivationGenerator(hook_points=cfg.hook_points),
                lambda: ActivationTransformer(hook_points=cfg.hook_points),
            ]

            # Create processors up to required stage
            processors = [factory() for factory in processor_factories[: stage_map[cfg.target]]]

            def process_dataset(**kwargs: Any):
                """Process a single dataset through the pipeline.

                Args:
                    **kwargs: Must contain 'datasets' dict and 'model' transformer

                Returns:
                    Stream of processed data
                """
                datasets: dict[str, Dataset] | None = kwargs.get("datasets")
                assert datasets is not None, "`datasets` must be provided for dataset sources"
                model: HookedTransformer | None = kwargs.get("model")
                assert model is not None, "`model` must be provided for dataset sources"

                dataset = datasets.get(dataset_source.name)
                assert dataset is not None, f"Dataset {dataset_source.name} not found in `datasets`"

                stream = loader(dataset)

                for processor in processors:
                    stream = processor(stream, model=model)

                return stream

            pre_aggregation_processors.append(process_dataset)

        if len(activations_sources) > 0:
            raise NotImplementedError("Activations sources are not implemented yet")

        return pre_aggregation_processors

    @staticmethod
    def build_post_aggregation_processor(
        cfg: ActivationFactoryConfig,
    ):
        """Build processor that runs after aggregation.

        Args:
            cfg: Factory configuration object

        Returns:
            Callable that processes aggregated data
        """
        stage_map = {
            "tokens": 0,
            "activations-2d": 0,
            "activations-1d": 0,
            "batched-activations-1d": 1,
        }

        def build_batchler():
            """Create batchler for batched-activations-1d target."""
            assert cfg.batch_size is not None, "Batch size must be provided for outputting batched-activations-1d"
            return ActivationBatchler(
                hook_points=cfg.hook_points, batch_size=cfg.batch_size, buffer_size=cfg.buffer_size
            )

        processor_factories: list[
            Callable[[], BaseActivationProcessor[Iterable[dict[str, Any]], Iterable[dict[str, Any]]]]
        ] = [lambda: build_batchler()]

        processors = [factory() for factory in processor_factories[: stage_map[cfg.target]]]

        def process_activations(activations: Iterable[dict[str, Any]], **kwargs: Any):
            """Process aggregated activations through post-processors.

            Args:
                activations: Stream of aggregated activation data
                **kwargs: Additional arguments passed to processors

            Returns:
                Processed activation stream
            """
            for processor in processors:
                activations = processor(activations, **kwargs)
            return activations

        return process_activations

    @staticmethod
    def build_aggregator(
        cfg: ActivationFactoryConfig,
    ):
        """Build function to aggregate data from multiple sources.

        Args:
            cfg: Factory configuration object

        Returns:
            Callable that aggregates data streams. Currently is a simple weighted random sampler.
        """
        source_sample_weights = np.array([source.sample_weights for source in cfg.sources])

        def aggregate(activations: list[Iterable[dict[str, Any]]], **kwargs: Any) -> Iterable[dict[str, Any]]:
            """Aggregate multiple activation streams by sampling based on weights.

            Args:
                activations: List of activation streams from different sources
                **kwargs: Additional arguments (unused)

            Yields:
                Sampled activation data with source info
            """
            ran_out_of_samples = np.zeros(len(cfg.sources), dtype=bool)
            activations: list[Iterator[dict[str, Any]]] = [iter(activation) for activation in activations]
            # Mask out sources run out of samples
            weights = source_sample_weights[~ran_out_of_samples]
            weights = weights / weights.sum()

            while not all(ran_out_of_samples):
                sampled_sources = np.random.choice(len(activations), replace=True, p=weights)
                try:
                    result = next(activations[sampled_sources])
                    if "info" in result:
                        result = result | {
                            "info": {
                                **result["info"],
                                "source": cfg.sources[sampled_sources].name,
                            },
                        }

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
