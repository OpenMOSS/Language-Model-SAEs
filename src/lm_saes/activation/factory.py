from typing import Any, Iterable, Iterator, Optional, Sequence

import numpy as np
import torch.distributed as dist
from datasets import Dataset
from transformer_lens import HookedTransformer

from lm_saes.activation.processors.activation import (
    ActivationBatchler,
    ActivationGenerator,
    ActivationTransformer,
)
from lm_saes.activation.processors.cached_activation import CachedActivationLoader
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
    ActivationFactoryTarget,
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
    def _build_pre_aggregation_dataset_source_processors(
        cfg: ActivationFactoryConfig, dataset_source: ActivationFactoryDatasetSource
    ):
        loader = HuggingFaceDatasetLoader(
            batch_size=1,
            num_workers=4,
            with_info=True,
            show_progress=True,
        )

        processors_optional: Sequence[
            BaseActivationProcessor[Iterable[dict[str, Any]], Iterable[dict[str, Any]]] | None
        ] = [
            RawDatasetTokenProcessor(prepend_bos=dataset_source.prepend_bos)
            if cfg.target >= ActivationFactoryTarget.TOKENS
            else None,
            PadAndTruncateTokensProcessor(seq_len=cfg.context_size)
            if cfg.target >= ActivationFactoryTarget.ACTIVATIONS_2D
            else None,
            ActivationGenerator(hook_points=cfg.hook_points, batch_size=cfg.model_batch_size)
            if cfg.target >= ActivationFactoryTarget.ACTIVATIONS_2D
            else None,
            ActivationTransformer(hook_points=cfg.hook_points)
            if cfg.target >= ActivationFactoryTarget.ACTIVATIONS_1D
            else None,
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
            model: HookedTransformer | None = kwargs.get("model")
            assert model is not None, "`model` must be provided for dataset sources"
            model_name: str | None = kwargs.get("model_name")
            assert model_name is not None, "`model_name` must be provided for dataset sources"

            dataset = datasets.get(dataset_source.name)
            assert dataset is not None, f"Dataset {dataset_source.name} not found in `datasets`"
            dataset, metadata = dataset

            stream = loader.process(dataset, dataset_name=dataset_source.name, metadata=metadata)

            for processor in processors:
                stream = processor.process(
                    stream, model=model, model_name=model_name, ignore_token_ids=cfg.ignore_token_ids
                )

            return stream

        return process_dataset

    @staticmethod
    def _build_pre_aggregation_activations_source_processors(
        cfg: ActivationFactoryConfig, activations_source: ActivationFactoryActivationsSource
    ):
        if cfg.target < ActivationFactoryTarget.ACTIVATIONS_2D:
            raise ValueError("Activations sources are only supported for target >= ACTIVATIONS_2D")

        loader = CachedActivationLoader(
            cache_dir=activations_source.path,
            hook_points=cfg.hook_points,
            device=activations_source.device,
            dtype=activations_source.dtype,
            num_workers=activations_source.num_workers,
            prefetch_factor=activations_source.prefetch,
        )

        processors = (
            [ActivationTransformer(hook_points=cfg.hook_points)]
            if cfg.target >= ActivationFactoryTarget.ACTIVATIONS_1D
            else []
        )

        def process_activations(**kwargs: Any):
            """Process a single activations source through the pipeline.

            Args:
                **kwargs: Could contain 'model' for providing tokenizer info for special tokens filtering

            Returns:
                Stream of processed data
            """
            model: HookedTransformer | None = kwargs.get("model")

            stream = loader.process()
            for processor in processors:
                stream = processor.process(stream, ignore_token_ids=cfg.ignore_token_ids, model=model)

            return stream

        return process_activations

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

        pre_aggregation_processors = [
            ActivationFactory._build_pre_aggregation_dataset_source_processors(cfg, source)
            for source in dataset_sources
        ] + [
            ActivationFactory._build_pre_aggregation_activations_source_processors(cfg, source)
            for source in activations_sources
        ]
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

        def build_batchler():
            """Create batchler for batched-activations-1d target."""
            assert cfg.batch_size is not None, "Batch size must be provided for outputting batched-activations-1d"
            return ActivationBatchler(
                hook_points=cfg.hook_points,
                batch_size=cfg.batch_size,
                buffer_size=cfg.buffer_size,
                buffer_shuffle_config=cfg.buffer_shuffle,
            )

        processors = [build_batchler()] if cfg.target >= ActivationFactoryTarget.BATCHED_ACTIVATIONS_1D else []

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

            # Fixed pattern, e.g., [0, 1, 1] repeats cyclically
            # pattern = [0, 1, 1]  # Can be adjusted as needed
            # pattern = [0,]
            pattern = [0, 1, 1, 1, 1, 1, 1, 1, 1]  # for analyze shared #####
            pattern_length = len(pattern)
            pattern_index = 0  # This will help cycle through the pattern

            while not all(ran_out_of_samples):
                sampled_sources = pattern[pattern_index % pattern_length]
                pattern_index += 1  # Move to the next element in the pattern

                try:
                    result = next(activations[sampled_sources])
                except StopIteration:
                    ran_out_of_samples[sampled_sources] = True
                    continue

                rank = dist.get_rank()

                if rank == 0:
                    with open(
                        "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs/exp/log/analyze-shared-0.txt",
                        "a",
                    ) as f:
                        f.write(f"selected_source: {sampled_sources}, shape[0]: {result['tokens'].shape}\n")

                        # if rank == 1:
                        #     with open('/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs/exp/log/shared-1.txt','a') as f:
                        #         f.write(f"selected_source: {sampled_sources}, shape[0]: {result['tokens'].shape}\n")

                        # if rank == 2:
                        #     with open('/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs/exp/log/shared-2.txt','a') as f:
                        f.write(f"selected_source: {sampled_sources}, shape[0]: {result['tokens'].shape}\n")

                # with open('/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs/exp/log/shared.txt','a') as f:
                # f.write(f"selected_source: {sampled_sources}, shape[0]: {result['tokens'].shape}\n")
                # print(result['meta'])
                # print(f"selected_source: {sampled_sources}, shape[0]: {result['tokens'].shape}\n")

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

        # print(f'{next(streams[0])['blocks.15.hook_resid_post'].shape}   0')
        # print(f'{next(streams[1])['blocks.15.hook_resid_post'].shape}   1')

        stream = self.aggregator(streams)

        # a = next(stream)
        # print(f'{a['meta'][0]['dataset_name']}     {a['blocks.15.hook_resid_post'].shape}')
        # a = next(stream)
        # print(f'{a['meta'][0]['dataset_name']}     {a['blocks.15.hook_resid_post'].shape}')

        return self.post_aggregation_processor(stream, **kwargs)
