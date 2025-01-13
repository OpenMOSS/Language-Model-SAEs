import itertools
from typing import Any, Iterable, Optional, cast

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from lm_saes.activation.processors.core import BaseActivationProcessor


def identity_collate_fn(x):
    """Identity collate function that returns the input as is.

    This should be defined in the global scope so it can be pickled when multiprocessing with `num_workers > 0`.
    """
    return x


class HuggingFaceDatasetLoader(BaseActivationProcessor[Dataset, Iterable[dict[str, Any]]]):
    """A processor that directly loads raw dataset from HuggingFace datasets.

    This processor takes a HuggingFace dataset and converts it into an iterable of raw data records.
    It can optionally add context index information to each data record and show a progress bar.

    Args:
        batch_size (int): Number of samples per batch. The batch is only used for loading the dataset.
            The returned data is always a flattened list of raw data records.
        num_workers (int, optional): Number of workers to use for loading the dataset.
            Defaults to 0. Use a larger `num_workers` if you have a large dataset and want to speed up loading.
        with_info (bool, optional): Whether to include context index information with each batch.
            If True, returns tuples of (batch, info) where info contains context indices. This context
            index is used to identify the original sample in the dataset.
            Defaults to False.
        show_progress (bool, optional): Whether to display a progress bar while loading.
            Defaults to True.

    Returns:
        Iterable: An iterator over batches from the dataset. If with_info=True, yields tuples of
            (batch, info) where info contains context indices for each sample in the batch.
    """

    def __init__(self, batch_size: int, num_workers: int = 0, with_info: bool = False, show_progress: bool = True):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.with_info = with_info
        self.show_progress = show_progress

    def process(
        self,
        data: Dataset,
        *,
        dataset_name: str | None = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Iterable[dict[str, Any]]:
        """Process the input dataset into batches.

        Args:
            data (Dataset): Input HuggingFace dataset to process
            dataset_name (str, optional): Name of the dataset. If provided, it will be added to the info field.
                Defaults to None.
            metadata (dict[str, Any], optional): Metadata to add to each batch. Defaults to None.
            **kwargs: Additional keyword arguments for processing. Not used by this processor.

        Returns:
            Iterable: Iterator over batches, optionally with context info if with_info=True
        """

        dataloader = cast(
            Iterable[list[dict[str, Any]]],
            DataLoader(
                cast(torch.utils.data.Dataset, data),
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                collate_fn=identity_collate_fn,
                num_workers=self.num_workers,
            ),
        )

        if self.show_progress:
            dataloader = tqdm(dataloader, desc="Loading dataset")

        flattened = itertools.chain.from_iterable(dataloader)

        if self.with_info:
            flattened = map(
                lambda x: x[1]
                | {
                    "meta": {
                        "context_idx": x[0],
                        **({"dataset_name": dataset_name} if dataset_name else {}),
                        **(metadata if metadata else {}),
                    }
                },
                enumerate(flattened),
            )

        return flattened
