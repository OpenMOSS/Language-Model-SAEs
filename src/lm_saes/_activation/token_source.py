import random
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import datasets
import torch
import torch.distributed as dist
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer

from ..config import TextDatasetConfig


class TokenSourceInfo(NamedTuple):
    dataset_idx: int
    context_idx: int


@dataclass
class TokenBatch:
    tokens: torch.Tensor
    source_info: list[TokenSourceInfo]

    def __len__(self):
        return len(self.tokens)


class BaseTokenSource:
    def __init__(
        self,
        dataloader: list[DataLoader],
        model: HookedTransformer,
        is_dataset_tokenized: bool,
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
        self.seq_len = seq_len
        self.device = model.cfg.device
        self.sample_probs = sample_probs
        self.prepend_bos = prepend_bos

        # TODO: Refactor to record the context_idx elsewhere
        if show_progress:
            self.data_iter = [
                iter((enumerate(tqdm(dataloader, desc=f"Loading samples from dataset {i}"))))
                for i, dataloader in enumerate(self.dataloader)
            ]
        else:
            self.data_iter = [iter(enumerate(dataloader)) for dataloader in self.dataloader]

    def reset_iter(self, empty_idx: int):
        self.data_iter = self.data_iter[:empty_idx] + self.data_iter[empty_idx + 1 :]
        self.sample_probs = self.sample_probs[:empty_idx] + self.sample_probs[empty_idx + 1 :]
        self.sample_probs = [prob / sum(self.sample_probs) for prob in self.sample_probs]

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


class MappedTokenSource(BaseTokenSource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_buffer = TokenBatch(
            tokens=torch.empty((0, self.seq_len), dtype=torch.long, device=self.device), source_info=[]
        )

    def fill_with_one_batch(
        self, batch: dict[str, Any] | list[dict[str, Any]], dataset_idx: int, batch_idx: int
    ) -> None:
        if self.is_dataset_tokenized:
            assert isinstance(batch, dict)
            tokens = get_tokens_from_tokenized_batch(batch, self.device)
        else:
            assert isinstance(batch, list)
            tokens = [
                t[:, : self.seq_len]
                for t in get_tokens_from_raw_batch(batch, self.model, self.prepend_bos[dataset_idx])
            ]

        tokens = pad_tokens(tokens, self.seq_len, self.tokenizer, self.device)
        source_info = [TokenSourceInfo(dataset_idx, batch_idx * len(tokens) + i) for i in range(len(tokens))]

        self.token_buffer = TokenBatch(
            tokens=torch.cat([self.token_buffer.tokens, tokens], dim=0),
            source_info=self.token_buffer.source_info + source_info,
        )

    def next(self, batch_size: int) -> TokenBatch | None:
        while self.token_buffer.tokens.size(0) < batch_size:
            dataset_idx = random.choices(range(len(self.dataloader)), weights=self.sample_probs)[0]
            try:
                batch_idx, batch = next(self.data_iter[dataset_idx])
            except StopIteration:
                if len(self.data_iter) > 1:
                    self.reset_iter(dataset_idx)
                    continue
                else:
                    return None

            self.fill_with_one_batch(batch, dataset_idx, batch_idx)

        ret = TokenBatch(
            tokens=self.token_buffer.tokens[:batch_size], source_info=self.token_buffer.source_info[:batch_size]
        )
        self.token_buffer = TokenBatch(
            tokens=self.token_buffer.tokens[batch_size:], source_info=self.token_buffer.source_info[batch_size:]
        )
        return ret

    @staticmethod
    def from_config(model: HookedTransformer, cfg: TextDatasetConfig):
        dataloader = [MappedTokenSource._process_dataset(dataset_path, cfg) for dataset_path in cfg.dataset_path]

        return MappedTokenSource(
            dataloader=dataloader,
            model=model,
            is_dataset_tokenized=cfg.is_dataset_tokenized,
            seq_len=cfg.context_size,
            sample_probs=cfg.sample_probs,
            prepend_bos=cfg.prepend_bos,
        )


class TokenSource(BaseTokenSource):
    def __init__(self, concat_tokens: list[bool], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_buffer = torch.empty((0, self.seq_len), dtype=torch.long, device=self.device)
        self.bos_token_id_tensor = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long, device=self.device)
        self.resid = torch.tensor([], dtype=torch.long, device=self.device)
        self.concat_tokens = concat_tokens

    def fill_with_one_batch(self, batch: dict[str, Any] | list[dict[str, Any]], pack: bool, prepend_bos: bool) -> None:
        self.token_buffer, self.resid = process_batch(
            batch=batch,
            pack=pack,
            prepend_bos=prepend_bos,
            is_dataset_tokenized=self.is_dataset_tokenized,
            model=self.model,
            device=self.device,
            seq_len=self.seq_len,
            token_buffer=self.token_buffer,
            resid=self.resid,
            bos_token_id_tensor=self.bos_token_id_tensor,
        )

    def next(self, batch_size: int) -> torch.Tensor | None:
        while self.token_buffer.size(0) < batch_size:
            dataset_idx = random.choices(range(len(self.dataloader)), weights=self.sample_probs)[0]
            try:
                _, batch = next(self.data_iter[dataset_idx])
            except StopIteration:
                if len(self.data_iter) > 1:
                    self.reset_iter(dataset_idx)
                    continue
                else:
                    return None

            self.fill_with_one_batch(
                batch, pack=self.concat_tokens[dataset_idx], prepend_bos=self.prepend_bos[dataset_idx]
            )

        ret = self.token_buffer[:batch_size]
        self.token_buffer = self.token_buffer[batch_size:]
        return ret

    @staticmethod
    def from_config(model: HookedTransformer, cfg: TextDatasetConfig):
        dataloader = [TokenSource._process_dataset(dataset_path, cfg) for dataset_path in cfg.dataset_path]

        return TokenSource(
            dataloader=dataloader,
            model=model,
            is_dataset_tokenized=cfg.is_dataset_tokenized,
            seq_len=cfg.context_size,
            sample_probs=cfg.sample_probs,
            prepend_bos=cfg.prepend_bos,
            concat_tokens=cfg.concat_tokens,
        )


def get_tokens_from_tokenized_batch(
    batch: dict[str, Any],
    device: str | None,
) -> torch.Tensor:
    """Extract tokens from a pre-tokenized batch."""
    if isinstance(batch["input_ids"], torch.Tensor):
        assert not batch["input_ids"].dtype.is_floating_point, "input_ids must be a tensor of integers"
        return batch["input_ids"].to(device)

    assert isinstance(batch["input_ids"], list), "input_ids must be a list or a tensor"
    assert all(
        len(seq) == len(batch["input_ids"][0]) for seq in batch["input_ids"]
    ), "All sequences must have the same length"
    return torch.tensor(batch["input_ids"], dtype=torch.long, device=device)


def get_tokens_from_raw_batch(
    batch: list[dict[str, Any]],
    model: HookedTransformer,
    prepend_bos: bool,
) -> list[torch.Tensor]:
    """Extract tokens from a raw (non-tokenized) batch."""
    return [model.to_tokens_with_origins(input, tokens_only=True, prepend_bos=prepend_bos) for input in batch]


def process_packed_tokens(
    tokens: torch.Tensor,
    tokenizer,
    seq_len: int,
    token_buffer: torch.Tensor,
    resid: torch.Tensor,
    bos_token_id_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process tokens in packed mode."""
    new_token_buffer = token_buffer
    new_resid = resid

    token_idx = 0
    while token_idx < tokens.size(0):
        cur_tokens = tokens[token_idx]
        # Filter special tokens
        cur_tokens = cur_tokens[cur_tokens != tokenizer.bos_token_id]
        cur_tokens = cur_tokens[cur_tokens != tokenizer.eos_token_id]
        cur_tokens = cur_tokens[cur_tokens != tokenizer.pad_token_id]

        # Add to residual
        new_resid = torch.cat([new_resid, bos_token_id_tensor.clone(), cur_tokens], dim=0)

        # Process complete sequences
        while new_resid.size(0) >= seq_len:
            new_token_buffer = torch.cat(
                [new_token_buffer, new_resid[:seq_len].unsqueeze(0)],
                dim=0,
            )
            new_resid = new_resid[seq_len:]
            new_resid = torch.cat([bos_token_id_tensor.clone(), new_resid], dim=0)
        token_idx += 1

    return new_token_buffer, new_resid


def process_unpacked_tokens(
    tokens: torch.Tensor,
    tokenizer,
    seq_len: int,
    device: str | None,
    token_buffer: torch.Tensor,
) -> torch.Tensor:
    """Process tokens in unpacked mode."""
    tokens = tokens[:, :seq_len]

    if tokens.size(1) < seq_len:
        tokens = pad_tokens(tokens, seq_len, tokenizer, device)

    return torch.cat([token_buffer, tokens], dim=0)


def pad_tokens(
    tokens: torch.Tensor | list[torch.Tensor],
    seq_len: int,
    tokenizer,
    device: str | None,
) -> torch.Tensor:
    """Pad tokens to desired sequence length."""
    if isinstance(tokens, list):
        return torch.cat([pad_tokens(t, seq_len, tokenizer, device) for t in tokens])
    pad_len = seq_len - tokens.size(1)
    pad_token_id = cast(int, tokenizer.pad_token_id)
    if pad_token_id is None:
        pad_token_id = 0  # Default to 0 if pad token not set

    padding = torch.full(
        (tokens.size(0), pad_len),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    return torch.cat([tokens, padding], dim=1)


def process_batch(
    batch: dict[str, Any] | list[dict[str, Any]],
    pack: bool,
    prepend_bos: bool,
    is_dataset_tokenized: bool,
    model: HookedTransformer,
    device: str | None,
    seq_len: int,
    token_buffer: torch.Tensor,
    resid: torch.Tensor,
    bos_token_id_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process a batch of tokens and return new token buffer and residual."""
    tokenizer = model.tokenizer
    assert tokenizer is not None, "Tokenizer is not set"

    # Get tokens based on dataset type
    if is_dataset_tokenized:
        assert isinstance(batch, dict), "Batch must be a dictionary for tokenized dataset"
        unaligned_tokens = [get_tokens_from_tokenized_batch(batch, device)]
    else:
        assert isinstance(batch, list), "Batch must be a list for non-tokenized dataset"
        unaligned_tokens = get_tokens_from_raw_batch(batch, model, prepend_bos)

    # Process tokens based on packing mode
    for tokens in unaligned_tokens:
        if pack:
            token_buffer, resid = process_packed_tokens(
                tokens,
                tokenizer,
                seq_len,
                token_buffer,
                resid,
                bos_token_id_tensor,
            )
        else:
            token_buffer = process_unpacked_tokens(
                tokens,
                tokenizer,
                seq_len,
                device,
                token_buffer,
            )
    return token_buffer, resid
