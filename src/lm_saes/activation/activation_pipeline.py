import itertools
from typing import Any

from datasets import Dataset
from transformer_lens import HookedTransformer

from lm_saes.activation.processors.activation import (
    ActivationBatchler,
    ActivationGenerator,
    ActivationTransformer,
)
from lm_saes.activation.processors.huggingface import HuggingFaceDatasetLoader
from lm_saes.activation.processors.token import (
    PadAndTruncateTokensProcessor,
    RawDatasetTokenProcessor,
)
from lm_saes.config import (
    ActivationPipelineActivationsSource,
    ActivationPipelineConfig,
    ActivationPipelineDatasetSource,
)


class ActivationPipeline:
    def __init__(self, cfg: ActivationPipelineConfig):
        self.cfg = cfg

    def process(self, **kwargs: Any):
        dataset_sources = [source for source in self.cfg.sources if isinstance(source, ActivationPipelineDatasetSource)]
        activations_sources = [
            source for source in self.cfg.sources if isinstance(source, ActivationPipelineActivationsSource)
        ]

        activation_streams = []

        if len(dataset_sources) > 0:
            datasets: dict[str, Dataset] | None = kwargs.get("datasets")
            assert datasets is not None, "dataset must be provided"
            assert all(
                dataset_source.name in datasets for dataset_source in dataset_sources
            ), "all dataset sources must be provided"

            model: HookedTransformer | None = kwargs.get("model")
            assert model is not None, "model must be provided"

            def process_dataset(dataset_source: ActivationPipelineDatasetSource, dataset: Dataset):
                stream = HuggingFaceDatasetLoader(
                    batch_size=1,
                    num_workers=16,
                    with_info=True,
                    show_progress=True,
                )(dataset)

                stream = RawDatasetTokenProcessor(prepend_bos=dataset_source.prepend_bos)(stream, model=model)

                stream = PadAndTruncateTokensProcessor(seq_len=self.cfg.context_size)(stream)

                stream = ActivationGenerator(hook_points=self.cfg.hook_points)(stream, model=model)

                stream = ActivationTransformer(hook_points=self.cfg.hook_points)(stream, model=model)

                return stream

            for dataset_source in dataset_sources:
                dataset = datasets[dataset_source.name]
                activation_streams.append(process_dataset(dataset_source, dataset))

        if len(activations_sources) > 0:
            raise NotImplementedError("Activations sources are not implemented yet")

        activation_stream = itertools.chain(*activation_streams)

        activation_stream = ActivationBatchler(
            hook_points=self.cfg.hook_points,
            batch_size=self.cfg.batch_size,
            buffer_size=self.cfg.buffer_size,
        )(activation_stream)

        return activation_stream
