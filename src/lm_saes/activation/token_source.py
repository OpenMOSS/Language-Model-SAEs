import random
from typing import Any, cast

import datasets
import torch
import torch.distributed as dist
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer

from ..config import TextDatasetConfig


class TokenSource:
    def __init__(
        self,
        dataloader: list[DataLoader],
        model: HookedTransformer,
        is_dataset_tokenized: bool,
        concat_tokens: list[bool],
        seq_len: int,
        sample_probs: list[float],
        prepend_bos: list[bool],
        show_progress: bool = True,
    ):
        self.dataloader = dataloader
        self.model = model
        assert model.tokenizer is not None, "Tokenizer is not set"
        self.tokenizer = model.tokenizer
        self.is_dataset_tokenized = is_dataset_tokenized
        self.concat_tokens = concat_tokens
        self.seq_len = seq_len
        self.device = model.cfg.device

        if show_progress:
            self.data_iter = [
                iter(tqdm(dataloader, desc=f"Loading samples from dataset {i}"))
                for i, dataloader in enumerate(self.dataloader)
            ]
        else:
            self.data_iter = [iter(dataloader) for dataloader in self.dataloader]

        self.token_buffer = torch.empty((0, seq_len), dtype=torch.long, device=self.device)

        self.bos_token_id_tensor = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long, device=self.device)
        self.resid = torch.tensor([], dtype=torch.long, device=self.device)

        self.sample_probs = sample_probs
        self.prepend_bos = prepend_bos

    def fill_with_one_batch(self, batch: dict[str, Any] | list[dict[str, Any]], pack: bool, prepend_bos: bool) -> None:
        if self.is_dataset_tokenized:
            assert isinstance(batch, dict), "Batch must be a dictionary for tokenized dataset"
            if isinstance(batch["input_ids"], torch.Tensor):
                assert not batch["input_ids"].dtype.is_floating_point, "input_ids must be a tensor of integers"
                tokens = batch["input_ids"].to(self.device)
            else:
                assert isinstance(batch["input_ids"], list), "input_ids must be a list or a tensor"
                print("Batch size:", len(batch["input_ids"]), "Type:", type(batch["input_ids"]))
                print("Sequence length:", len(batch["input_ids"][0]), "Type:", type(batch["input_ids"][0]))
                # Check if all sequences in the batch have the same length
                assert all(
                    len(seq) == len(batch["input_ids"][0]) for seq in batch["input_ids"]
                ), "All sequences must have the same length"
                tokens = torch.tensor(batch["input_ids"], dtype=torch.long, device=self.device)
            unaligned_tokens = [
                tokens
            ]  # Unaligned tokens are the tokens that are not aligned to the same sequence length
        else:
            assert isinstance(batch, list), "Batch must be a list for non-tokenized dataset"
            unaligned_tokens = [
                self.model.to_tokens_with_origins(input, tokens_only=True, prepend_bos=prepend_bos) for input in batch
            ]

        for tokens in unaligned_tokens:
            if pack:
                while tokens.size(0) > 0:
                    cur_tokens = tokens[0]
                    cur_tokens = cur_tokens[cur_tokens != self.tokenizer.bos_token_id]
                    cur_tokens = cur_tokens[cur_tokens != self.tokenizer.eos_token_id]
                    cur_tokens = cur_tokens[cur_tokens != self.tokenizer.pad_token_id]

                    self.resid = torch.cat([self.resid, self.bos_token_id_tensor.clone(), cur_tokens], dim=0)
                    while self.resid.size(0) >= self.seq_len:
                        self.token_buffer = torch.cat(
                            [self.token_buffer, self.resid[: self.seq_len].unsqueeze(0)],
                            dim=0,
                        )
                        self.resid = self.resid[self.seq_len :]
                        self.resid = torch.cat([self.bos_token_id_tensor.clone(), self.resid], dim=0)
                    tokens = tokens[1:]
            else:
                tokens = tokens[:, : self.seq_len]

                if tokens.size(1) < self.seq_len:
                    pad_len = self.seq_len - tokens.size(1)
                    pad_token_id = cast(int, self.tokenizer.pad_token_id)
                    if pad_token_id is None:
                        pad_token_id = 0  # Default to 0 if pad token not set
                    tokens = torch.cat(
                        [
                            tokens,
                            torch.full(
                                (tokens.size(0), pad_len),
                                pad_token_id,
                                dtype=torch.long,
                                device=self.device,
                            ),
                        ],
                        dim=1,
                    )
                self.token_buffer = torch.cat([self.token_buffer, tokens], dim=0)

    def reset_iter(self, empty_idx: int):
        self.data_iter = self.data_iter[:empty_idx] + self.data_iter[empty_idx + 1 :]

        self.sample_probs = self.sample_probs[:empty_idx] + self.sample_probs[empty_idx + 1 :]

        self.sample_probs = [prob / sum(self.sample_probs) for prob in self.sample_probs]

    def next(self, batch_size: int) -> torch.Tensor | None:
        while self.token_buffer.size(0) < batch_size:
            dataset_idx_to_fetch = random.choices(range(len(self.dataloader)), weights=self.sample_probs)[0]
            try:
                batch = next(self.data_iter[dataset_idx_to_fetch])
            except StopIteration:
                if len(self.data_iter) > 1:
                    self.reset_iter(dataset_idx_to_fetch)
                    continue
                else:
                    return None

            self.fill_with_one_batch(
                batch, self.concat_tokens[dataset_idx_to_fetch], prepend_bos=self.prepend_bos[dataset_idx_to_fetch]
            )

        ret = self.token_buffer[:batch_size]
        self.token_buffer = self.token_buffer[batch_size:]

        return ret

    @staticmethod
    def _process_dataset(dataset_path: str, cfg: TextDatasetConfig):
        if not cfg.is_dataset_on_disk:
            dataset = load_dataset(dataset_path, split="train", cache_dir=cfg.cache_dir)
        else:
            dataset = load_from_disk(dataset_path)
        dataset = cast(datasets.Dataset, dataset)
        if dist.is_initialized():
            shard_id = dist.get_rank()
            shard = dataset.shard(num_shards=dist.get_world_size(), index=shard_id, contiguous=True)
        else:
            shard = dataset
        shard = shard.with_format("torch")

        dataloader = DataLoader(
            dataset=cast(Dataset[dict[str, Any]], shard),
            batch_size=cfg.store_batch_size,
            pin_memory=True,
            collate_fn=lambda x: x if not cfg.is_dataset_tokenized else None,
        )
        return dataloader

    @staticmethod
    def from_config(model: HookedTransformer, cfg: TextDatasetConfig):
        dataloader = [TokenSource._process_dataset(dataset_path, cfg) for dataset_path in cfg.dataset_path]

        return TokenSource(
            dataloader=dataloader,
            model=model,
            is_dataset_tokenized=cfg.is_dataset_tokenized,
            concat_tokens=cfg.concat_tokens,
            seq_len=cfg.context_size,
            sample_probs=cfg.sample_probs,
            prepend_bos=cfg.prepend_bos,
        )
