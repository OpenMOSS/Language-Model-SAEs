import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, load_from_disk

from transformer_lens import HookedTransformer

from core.config import TextDatasetConfig

class TokenSource:
    def __init__(
        self,
        dataloader: DataLoader,
        model: HookedTransformer,
        is_dataset_tokenized: bool,
        concat_tokens: bool,
        seq_len: int,
        device: str
    ):
        self.dataloader = dataloader
        self.model = model
        self.is_dataset_tokenized = is_dataset_tokenized
        self.concat_tokens = concat_tokens
        self.seq_len = seq_len
        self.device = device

        self.data_iter = iter(self.dataloader)
        self.token_buffer = torch.empty((0, seq_len), dtype=torch.int64, device=self.device)
        if self.concat_tokens:
            self.bos_token_id_tensor = torch.tensor([self.model.tokenizer.bos_token_id], dtype=torch.int64, device=self.device)
            self.resid = self.bos_token_id_tensor.clone()
    
    def next(self, batch_size: int) -> torch.Tensor | None:
        while self.token_buffer.size(0) < batch_size:
            try:
                batch = next(self.data_iter)
            except StopIteration:
                return None
            if self.is_dataset_tokenized:
                tokens: torch.Tensor = batch["tokens"].to(self.device)
            else:
                tokens = self.model.to_tokens(batch["text"]).to(self.device)
            if self.concat_tokens:
                while tokens.size(0) > 0:
                    cur_tokens = tokens[0]
                    cur_tokens = cur_tokens[torch.logical_and(cur_tokens != self.model.tokenizer.pad_token_id, cur_tokens != self.model.tokenizer.bos_token_id)]
                    self.resid = torch.cat([self.resid, self.bos_token_id_tensor.clone(), cur_tokens], dim=0)
                    while self.resid.size(0) >= self.seq_len:
                        self.token_buffer = torch.cat([self.token_buffer, self.resid[:self.seq_len].unsqueeze(0)], dim=0)
                        self.resid = self.resid[self.seq_len:]
                        self.resid = torch.cat([self.bos_token_id_tensor.clone(), self.resid], dim=0)
                    tokens = tokens[1:]
            else:
                tokens = tokens[:, torch.logical_and(tokens[:, self.seq_len - 1] != self.model.tokenizer.pad_token_id, tokens[:, self.seq_len - 1] != self.model.tokenizer.eos_token_id)]
                self.token_buffer = torch.cat([self.token_buffer, tokens], dim=0)

        ret = self.token_buffer[:batch_size]
        self.token_buffer = self.token_buffer[batch_size:]

        return ret
    
    @staticmethod
    def from_config(model: HookedTransformer, cfg: TextDatasetConfig):
        if not cfg.is_dataset_on_disk:
            dataset = load_dataset(cfg.dataset_path, split="train", cache_dir=cfg.cache_dir)
        else:
            dataset = load_from_disk(cfg.dataset_path)
        if cfg.use_ddp:
            shard_id = cfg.rank
            shard = dataset.shard(num_shards=cfg.world_size, index=shard_id)
        else:
            shard = dataset
        
        if cfg.use_ddp:
            shard_id = cfg.rank
            shard = dataset.shard(num_shards=cfg.world_size, index=shard_id)
        else:
            shard = dataset
            
        dataloader = DataLoader(shard, batch_size=cfg.store_batch_size)
        return TokenSource(
            dataloader=dataloader,
            model=model,
            is_dataset_tokenized=cfg.is_dataset_tokenized,
            concat_tokens=cfg.concat_tokens,
            seq_len=cfg.context_size,
            device=cfg.device,
        )