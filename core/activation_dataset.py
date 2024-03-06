import numpy as np
from transformer_lens import HookedTransformer
import torch
from tqdm.auto import tqdm
import os
import json
import datasets
import argparse
from torch.utils.data import DataLoader
from einops import rearrange, repeat
import copy

from core.utils import compute_attention_mask

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--length', type=int, default=1024)
    parser.add_argument('--pad_token_id', type=int, default=50256)
    parser.add_argument('--act_size', type=int, default=768)
    parser.add_argument('--chunk_size', type=int, default=0.5 * 2 ** 30)
    parser.add_argument('--world_size', type=int, default=8)
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()

    return args



def make_activation_dataset(
	dataloader: DataLoader,
	model: HookedTransformer,
	save_path: str,
    act_size: int,
	act_names: list[str],
    length: int,
	batch_size: int,
    chunk_size: int,
    pad_token_id: int,
	device: torch.device,
):
    os.makedirs(save_path, exist_ok=False)

    with torch.no_grad():
        token_act_size = 4 * act_size # 4 for float32
        max_tokens_per_chunk = chunk_size // token_act_size
        print(f"Making activation dataset with approximately {max_tokens_per_chunk} tokens per chunk")

        data_iter = iter(dataloader)

        n_tokens = 0
        pbar = tqdm(total=len(dataloader))
        chunk_idx = 0
        end = False

        while not end:
            act_dict = {act_name: torch.empty((0, act_size), dtype=torch.float32, device=device) for act_name in act_names}
            context = torch.empty((0, length), dtype=torch.long, device=device)
            position = torch.empty((0,), dtype=torch.long, device=device)
            context_ids = torch.empty((0,), dtype=torch.long, device=device)

            n_tokens_in_chunk = 0

            for batch_idx, batch in enumerate(data_iter):
                input_ids = batch["tokens"].to(device)
                _, cache = model.run_with_cache(input_ids)
                attention_mask = rearrange(compute_attention_mask(input_ids, pad_token_id), "b l -> (b l)")
                for act_name in act_names:
                    act = cache[act_name]
                    act_dict[act_name] = torch.cat([act_dict[act_name], rearrange(act.float(), "b l d -> (b l) d")[attention_mask]], dim=0)
                context = torch.cat([context, input_ids], dim=0)
                position = torch.cat([position, repeat(torch.arange(length, device=device), 'l -> (b l)', b=batch_size)[attention_mask]], dim=0)
                context_ids = torch.cat([context_ids, repeat(torch.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size, device=device), 'b -> (b l)', l=length)[attention_mask]], dim=0)
                n_tokens += attention_mask.sum().item()
                n_tokens_in_chunk += attention_mask.sum().item()

                pbar.update(1)
                pbar.set_postfix({"n_tokens": n_tokens})

                if n_tokens_in_chunk >= max_tokens_per_chunk:
                    break
            else:
                end = True
                
            chunk_folder = os.path.join(save_path, f"chunk_{chunk_idx}")
            os.makedirs(chunk_folder, exist_ok=False)
            for act_name, act in act_dict.items():
                torch.save(act.cpu(), os.path.join(chunk_folder, f"{act_name}.pt"))
            torch.save(position.cpu(), os.path.join(chunk_folder, "position.pt"))
            torch.save(context.cpu(), os.path.join(chunk_folder, "context.pt"))
            torch.save(context_ids.cpu(), os.path.join(chunk_folder, "context_ids.pt"))
            file_size = sum(os.path.getsize(os.path.join(chunk_folder, f"{act_name}.pt")) for act_name in act_dict) + os.path.getsize(os.path.join(chunk_folder, "position.pt")) + os.path.getsize(os.path.join(chunk_folder, "context.pt")) + os.path.getsize(os.path.join(chunk_folder, "context_ids.pt"))

            with open(os.path.join(save_path, "metadata.jsonl"), "a") as f:
                metadata = {
                    "chunk_idx": chunk_idx,
                    "chunk_path": chunk_folder,
                    "start_token": n_tokens - n_tokens_in_chunk,
                    "end_token": n_tokens,
                    "file_size": file_size,
                    "n_tokens": n_tokens_in_chunk,
                }
                f.write(json.dumps(metadata) + "\n")

            print(f"Successfully saved chunk {chunk_idx} with size {file_size / 2 ** 30:.2f} GB")

            chunk_idx += 1

    pbar.close()

def prepare_realtime_activation_dataset(
	dataloader: DataLoader,
	model: HookedTransformer,
	act_name: str,
    length: int,
    pad_token_id: int,
	device: torch.device,
):
    def generator():
        for batch in dataloader:
            with torch.no_grad():
                input_ids: torch.Tensor = batch["tokens"].to(device)
                bs = input_ids.size(0)
                _, cache = model.run_with_cache(input_ids, names_filter=[act_name])
                attention_mask = rearrange(compute_attention_mask(input_ids, pad_token_id), "b l -> (b l)")

                result = {}
                result["activation"] = rearrange(cache[act_name], "b l d -> (b l) d")[attention_mask].detach()
                result["context"] = repeat(input_ids.to(dtype=torch.int32), 'b l -> (b repeat) l', repeat=length)[attention_mask].detach()
                result["position"] = repeat(torch.arange(length, device=device), 'l -> (b l)', b=bs)[attention_mask].detach()
            yield result
    return generator

def prepare_activation_dataset(
    save_path: str,
    act_name: str,
):
    print("Loading activation dataset from", save_path)
    metadata_path = os.path.join(save_path, "metadata.jsonl")
    metadata = [json.loads(line) for line in open(metadata_path)]
    def generator(chunk_ids):
        for chunk_id in chunk_ids:
            chunk_folder = metadata[chunk_id]["chunk_path"]

            activation = torch.load(os.path.join(chunk_folder, f"{act_name}.pt")).cuda()
            position = torch.load(os.path.join(chunk_folder, "position.pt")).cuda()
            context = torch.load(os.path.join(chunk_folder, "context.pt")).cuda()
            context_ids = torch.load(os.path.join(chunk_folder, "context_ids.pt")).cuda()
            yield {
                "activation": activation,
                "position": position,
                "context": context[context_ids],
            }
    return copy.deepcopy(metadata), generator

if __name__ == '__main__':
    args = get_args()

    model = HookedTransformer.from_pretrained('gpt2')
    model = model.cuda()
    model.eval()

    # dataset = datasets.load_dataset("Skylion007/openwebtext", cache_dir=args.cache_dir)
    # dataset.save_to_disk(os.path.join(args.cache_dir, "openwebtext"))
    dataset = datasets.load_from_disk(os.path.join(args.cache_dir, "openwebtext"))
    def tokenize(batch):
        tokens = model.to_tokens(batch['text'])[:, :args.length]
        if tokens.shape[1] < args.length:
            tokens = torch.cat([tokens, torch.tensor([[args.pad_token_id] * (args.length - tokens.shape[1])])], dim=1)
        # tokens = tokens.cpu()
        return {'tokens': tokens}
    shard = dataset['train'].shard(num_shards=64, index=args.rank).map(tokenize, batched=True)

    def collate_fn(examples):
        tokens = torch.LongTensor([example['tokens'] for example in examples])
        return {'tokens': tokens}

    dataloader = DataLoader(shard, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    act_names = ['hook_pos_embed'] + [f'blocks.{i}.hook_attn_out' for i in range(12)] + [f'blocks.{i}.hook_mlp_out' for i in range(12)]
    
    make_activation_dataset(
        dataloader,
        model,
        os.path.join(args.save_path, f"shard_{args.rank}"),
        args.act_size,
        act_names,
        args.length,
        args.batch_size,
        args.chunk_size,
        args.pad_token_id,
        torch.device("cuda"),
    )