# based on https://github.com/EleutherAI/gpt-neox/blob/main/megatron/tokenizer/tokenizer.py
import json
import pathlib
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import torch
import tqdm

class HFAutoTokenizer:
    def __init__(self, vocab_file):
        try:
            from tokenizers import Tokenizer
        except ImportError:
            print("tokenizers not found, unable to use HFAutoTokenizer")
            Tokenizer = None

        self.tokenizer = Tokenizer.from_file(vocab_file)
        self.eos = "</s>"
        self.bos = "<s>"
        self.eos_id = self.tokenize(self.eos)
        self.bos_id = self.tokenize(self.bos)
        self.vsize = 32000

    def encode_to_list(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def tokenize_file(self, input_file, output_file, verbose=False):
        if verbose:
            print(f"Tokenizing file: {input_file}")

        if pathlib.Path(output_file).exists():
            print(f"Output file {output_file} already exists, skipping")
            return
        with open(input_file, "r") as fin, open(output_file, "w") as fout:
            for line in tqdm.tqdm(fin):
                if verbose:
                    print(f"Tokenizing line: {line[-200:]}")
                data = json.loads(line.strip())
                if "text" not in data.keys():
                    break
                tokenized_data = self.tokenize(data["text"])
                fout.write(json.dumps({"tokens": tokenized_data}) + "\n")

    def tokenize(self, text: str, *args, **kwargs):
        ids = self.tokenizer.encode(text)
        if type(ids) == list:
            return torch.tensor(ids)
        else:
            return torch.tensor(ids.ids)

    def tokenize_batch(self, text_batch):
        return self.tokenizer.encode_batch(text_batch)

    def detokenize(self, token_ids, skip_special_tokens=False):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def detokenize_batch(self, token_ids_batch, skip_special_tokens=False):
        out = []
        for token_ids in token_ids_batch:
            out.append(
                self.detokenize(
                    [t.item() for t in token_ids],
                    skip_special_tokens=skip_special_tokens,
                )
            )
        return out

    @property
    def eod(self):
        return self.eod_id

    @property
    def vocab_size(self):
        return 32000


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError("detokenizer is not implemented for {} " "tokenizer".format(self.name))

    @property
    def cls(self):
        raise NotImplementedError("CLS is not provided for {} " "tokenizer".format(self.name))

    @property
    def sep(self):
        raise NotImplementedError("SEP is not provided for {} " "tokenizer".format(self.name))

    @property
    def pad(self):
        raise NotImplementedError("PAD is not provided for {} " "tokenizer".format(self.name))

    @property
    def eod(self):
        raise NotImplementedError("EOD is not provided for {} " "tokenizer".format(self.name))

    @property
    def mask(self):
        raise NotImplementedError("MASK is not provided for {} " "tokenizer".format(self.name))


class CharLevelTokenizer(AbstractTokenizer):
    """Character Level Tokenizer"""

    def __init__(self, vocab_size):
        name = "CharLevelTokenizer"
        super().__init__(name)
        self._vocab_size = vocab_size
        self.eod_id = 0
        self.eos_id = 0
        self.pad_id = 1

    def clamp(self, n):
        return max(32, min(n, self.vocab_size))

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def vocab(self):
        raise NotImplementedError

    @property
    def inv_vocab(self):
        raise NotImplementedError

    def decode_token(self, token: int):
        return str(chr(self.clamp(token)))

    def tokenize(self, text: str):
        return list(np.frombuffer(text.encode('utf-8'), dtype=np.uint8))

    def tokenize_batch(self, text_batch: Union[List[str], str]):
        if isinstance(text_batch, list):
            return [self.tokenize(s) for s in text_batch]
        else:
            return self.tokenize(text_batch)

    def detokenize(self, token_ids):
        return "".join(list(map(self.decode_token, token_ids)))

    def detokenize_batch(self, token_ids: Union[List[str], str]):
        if isinstance(token_ids, list):
            return [self.detokenize(s) for s in token_ids]
        # elif if tensor, convert to list first
        elif isinstance(token_ids, torch.Tensor):
            return [self.detokenize(s) for s in token_ids.tolist()]
        else:
            return self.detokenize(token_ids)

    @property
    def eod(self):
        return self.eod_id

    # duplicate to suppose both names, eos and eod
    @property
    def eos(self):
        return self.eod_id


class HookedEvo2Tokenizer:
    """Lightweight tokenizer adapter for Evo2 inside HookedTransformer."""

    def __init__(self, vocab_size: int, padding_side: str = "right"):
        self.base_tokenizer = CharLevelTokenizer(vocab_size)
        self.name_or_path = "arcinstitute/evo2_7b"
        self.padding_side = padding_side
        self.pad_token_id = self.base_tokenizer.pad_id
        self.eos_token_id = self.base_tokenizer.eod
        self.bos_token_id = self.base_tokenizer.eod
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"

    def tokenize(self, text: str):
        return self.base_tokenizer.tokenize(text)

    def tokenize_batch(self, text_batch: Union[List[str], str]):
        return self.base_tokenizer.tokenize_batch(text_batch)

    def detokenize(self, token_ids):
        return self.base_tokenizer.detokenize(token_ids)

    def detokenize_batch(self, token_ids):
        return self.base_tokenizer.detokenize_batch(token_ids)

    def encode(self, text: str):
        return self.tokenize(text)

    def decode(self, token_ids, clean_up_tokenization_spaces: bool = False):
        del clean_up_tokenization_spaces
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.detokenize(token_ids)

    def batch_decode(self, token_ids_batch, clean_up_tokenization_spaces: bool = False):
        del clean_up_tokenization_spaces
        if isinstance(token_ids_batch, torch.Tensor):
            token_ids_batch = token_ids_batch.tolist()
        return self.detokenize_batch(token_ids_batch)

    def decode_token(self, token_id: int):
        return self.base_tokenizer.decode_token(token_id)

    def __call__(
        self,
        input: Union[str, List[str]],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int | None = None,
    ):
        if isinstance(input, str):
            text_batch = [input]
        else:
            text_batch = input

        token_lists = [self.tokenize(text) for text in text_batch]
        if truncation and max_length is not None:
            token_lists = [tokens[:max_length] for tokens in token_lists]

        max_len = max((len(tokens) for tokens in token_lists), default=0)
        padded_tokens = []
        attention_masks = []
        for tokens in token_lists:
            pad_len = max_len - len(tokens) if padding else 0
            if self.padding_side == "left":
                padded = [self.pad_token_id] * pad_len + tokens
                mask = [0] * pad_len + [1] * len(tokens)
            else:
                padded = tokens + [self.pad_token_id] * pad_len
                mask = [1] * len(tokens) + [0] * pad_len
            padded_tokens.append(padded)
            attention_masks.append(mask)

        if return_tensors != "pt":
            raise ValueError(f"Unsupported return_tensors={return_tensors!r} for HookedEvo2Tokenizer")

        return {
            "input_ids": torch.tensor(padded_tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }
