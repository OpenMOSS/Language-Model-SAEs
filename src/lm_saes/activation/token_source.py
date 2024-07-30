import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, load_from_disk, Dataset

from transformer_lens import HookedTransformer
import torch.distributed as dist
from lm_saes.config import TextDatasetConfig
import random


class TokenSource:
    def __init__(
        self,
        dataloader: list[DataLoader],
        model: HookedTransformer,
        is_dataset_tokenized: bool,
        concat_tokens: list[bool],
        seq_len: int,
        sample_probs: list[float],
        prepend_bos: list[bool]
    ):
        self.dataloader = dataloader
        self.model = model
        self.is_dataset_tokenized = is_dataset_tokenized
        self.concat_tokens = concat_tokens
        self.seq_len = seq_len
        self.device = model.cfg.device

        self.data_iter = [iter(dataloader) for dataloader in self.dataloader]

        self.token_buffer = torch.empty(
            (0, seq_len), dtype=torch.long, device=self.device
        )

        self.bos_token_id_tensor = torch.tensor(
            [self.model.tokenizer.bos_token_id], dtype=torch.long, device=self.device
        )
        self.resid = torch.tensor([], dtype=torch.long, device=self.device)

        self.sample_probs = sample_probs
        self.prepend_bos = prepend_bos


    def fill_with_one_batch(self, batch, pack: bool, prepend_bos: bool) -> None:
        if self.is_dataset_tokenized:
            tokens: torch.Tensor = batch["tokens"].to(self.device)
        else:
            tokens = self.model.to_tokens(batch["text"], prepend_bos=prepend_bos).to(self.device)
        if pack:
            while tokens.size(0) > 0:
                cur_tokens = tokens[0]
                cur_tokens = cur_tokens[cur_tokens != self.model.tokenizer.bos_token_id]
                cur_tokens = cur_tokens[cur_tokens != self.model.tokenizer.eos_token_id]
                cur_tokens = cur_tokens[cur_tokens != self.model.tokenizer.pad_token_id]

                self.resid = torch.cat(
                    [self.resid, self.bos_token_id_tensor.clone(), cur_tokens], dim=0
                )
                while self.resid.size(0) >= self.seq_len:
                    self.token_buffer = torch.cat(
                        [self.token_buffer, self.resid[: self.seq_len].unsqueeze(0)],
                        dim=0,
                    )
                    self.resid = self.resid[self.seq_len :]
                    self.resid = torch.cat(
                        [self.bos_token_id_tensor.clone(), self.resid], dim=0
                    )
                tokens = tokens[1:]
        else:
            tokens = tokens[:, : self.seq_len]

            if tokens.size(1) < self.seq_len:
                pad_len = self.seq_len - tokens.size(1)
                tokens = torch.cat([tokens, torch.full((tokens.size(0), pad_len), self.model.tokenizer.pad_token_id, dtype=torch.long, device=self.device)], dim=1)
            self.token_buffer = torch.cat([self.token_buffer, tokens], dim=0)

    def reset_iter(self, empty_idx: int):
        self.data_iter = self.data_iter[:empty_idx] + self.data_iter[empty_idx + 1 :]

        self.sample_probs = (
            self.sample_probs[:empty_idx] + self.sample_probs[empty_idx + 1 :]
        )

        self.sample_probs = [
            prob / sum(self.sample_probs) for prob in self.sample_probs
        ]

    def next(self, batch_size: int) -> torch.Tensor | None:
        while self.token_buffer.size(0) < batch_size:
            dataset_idx_to_fetch = random.choices(
                range(len(self.dataloader)), weights=self.sample_probs
            )[0]
            try:
                batch = next(self.data_iter[dataset_idx_to_fetch])
            except StopIteration:
                if len(self.data_iter) > 1:
                    self.reset_iter(dataset_idx_to_fetch)
                    continue
                else:
                    return None

            self.fill_with_one_batch(batch, self.concat_tokens[dataset_idx_to_fetch], prepend_bos=self.prepend_bos[dataset_idx_to_fetch])

        ret = self.token_buffer[:batch_size]
        self.token_buffer = self.token_buffer[batch_size:]

        return ret

    @staticmethod
    def _process_dataset(dataset_path: str, cfg: TextDatasetConfig):
        if not cfg.is_dataset_on_disk:
            dataset = load_dataset(dataset_path, split="train", cache_dir=cfg.cache_dir)
        else:
            dataset = load_from_disk(dataset_path)
        if dist.is_initialized():
            shard_id = dist.get_rank()
            shard = dataset.shard(
                num_shards=dist.get_world_size(), index=shard_id, contiguous=True
            )
        else:
            shard = dataset


        dataloader = DataLoader(shard, batch_size=cfg.store_batch_size, num_workers=4, prefetch_factor=4, pin_memory=True)
        return dataloader

    @staticmethod
    def from_config(model: HookedTransformer, cfg: TextDatasetConfig):
        dataloader = [
            TokenSource._process_dataset(dataset_path, cfg)
            for dataset_path in cfg.dataset_path
        ]

        return TokenSource(
            dataloader=dataloader,
            model=model,
            is_dataset_tokenized=cfg.is_dataset_tokenized,
            concat_tokens=cfg.concat_tokens,
            seq_len=cfg.context_size,
            sample_probs=cfg.sample_probs,
            prepend_bos=cfg.prepend_bos
        )