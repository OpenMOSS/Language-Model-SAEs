from typing import Callable, Dict, Generator
import torch
import torch.distributed as dist
import torch.utils.data

from transformer_lens import HookedTransformer

from datasets import load_dataset, load_from_disk

from core.activation_dataset import prepare_realtime_activation_dataset
from core.config import RunnerConfig

class ActivationStore:
    def __init__(self, 
        activation_generator: Callable[..., Generator[Dict[str, torch.Tensor], None, None]],
        gen_kwargs=None, 
        n_tokens_in_buffer=500000,
        device="cuda",
        use_ddp=False
    ):
        self.activation_generator = activation_generator
        self.gen_kwargs = gen_kwargs
        self.buffer_size = n_tokens_in_buffer
        self.device = device
        self.use_ddp = use_ddp
        self._store: Dict[str, torch.Tensor] = {}
        
    def initialize(self):
        self.create_generator()
        self.refill()
        self.shuffle()

    def shuffle(self):
        for k in self._store:
            self._store[k] = self._store[k][torch.randperm(len(self._store[k]))]

    def refill(self):
        while self.__len__() < self.buffer_size:
            new_act = next(self._generator, None)
            if new_act is None:
                self.create_generator()
                continue
            for k, v in new_act.items():
                v = v.to(self.device)
                if k not in self._store:
                    self._store[k] = v
                else:
                    self._store[k] = torch.cat([self._store[k], v], dim=0)
            # Check if all activations have the same size
            assert len(set(v.size(0) for v in self._store.values())) == 1

    def create_generator(self):
        if self.gen_kwargs is not None:
            self._generator = self.activation_generator(**self.gen_kwargs)
        else:
            self._generator = self.activation_generator()

    def __len__(self):
        if len(self._store) == 0:
            return 0
        return next(iter(self._store.values())).size(0)

    def next(self, batch_size) -> Dict[str, torch.Tensor] | None:
        # Check if the activation store needs to be refilled.
        need_refill = torch.tensor([self.__len__() < self.buffer_size // 2], device=self.device, dtype=torch.int)
        if self.use_ddp: # When using DDP, we do refills in a synchronized manner to save time
            dist.all_reduce(need_refill, op=dist.ReduceOp.MAX)
        if need_refill.item() > 0:
            self.refill()
            self.shuffle()
        if self.use_ddp: # Wait for all processes to refill the store
            dist.barrier()

        ret = {k: v[:batch_size] for k, v in self._store.items()}
        for k in self._store:
            self._store[k] = self._store[k][batch_size:]
        return ret if len(ret) > 0 else None
    
    @staticmethod
    def from_config(model: HookedTransformer, cfg: RunnerConfig):
        if cfg.use_cached_activations:
            # TODO: Implement this
            raise NotImplementedError
        else:           
            if not cfg.is_dataset_on_disk:
                dataset = load_dataset(cfg.dataset_path)['train']
            else:
                dataset = load_from_disk(cfg.dataset_path)
            if cfg.use_ddp:
                shard_id = cfg.rank
                shard = dataset.shard(num_shards=cfg.world_size, index=shard_id)
            else:
                shard = dataset

            pad_token_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.bos_token_id
            
            if cfg.is_dataset_tokenized:
                assert 'tokens' in shard.column_names
                assert len(shard[0]['tokens']) == cfg.context_size
                def collate_fn(examples):
                    tokens = torch.LongTensor([example['tokens'] for example in examples])
                    return {'tokens': tokens}
            else:
                def collate_fn(examples):
                    tokens = model.to_tokens([example['text'] for example in examples])[:, :cfg.context_size]
                    if tokens.shape[1] < cfg.context_size:
                        tokens = torch.cat([tokens, torch.tensor([[pad_token_id] * (cfg.context_size - tokens.shape[1])])], dim=1)
                    return {'tokens': tokens}
                
            dataloader = torch.utils.data.DataLoader(shard, batch_size=cfg.store_batch_size, collate_fn=collate_fn)

            generator = prepare_realtime_activation_dataset(
                dataloader=dataloader,
                model=model,
                act_name=cfg.hook_point,
                length=cfg.context_size,
                pad_token_id=pad_token_id,
                device=cfg.device,
            )
            
            return ActivationStore(
                activation_generator=generator,
                n_tokens_in_buffer=cfg.n_tokens_in_buffer,
                device=cfg.device,
                use_ddp=cfg.use_ddp,
            )