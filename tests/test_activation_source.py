from core.activation.activation_source import TokenActivationSource
from core.activation.token_source import TokenSource

from datasets import load_dataset
from transformer_lens import HookedTransformer

import torch

import pytest

@pytest.fixture
def dataset():
    return load_dataset("Skylion007/openwebtext", split="train")

@pytest.fixture
def dataloader(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=32)

@pytest.fixture
def model():
    return HookedTransformer.from_pretrained('gpt2')

def test_token_source(dataloader, model):
    token_source = TokenSource(
        dataloader=dataloader,
        model=model,
        is_dataset_tokenized=False,
        seq_len=128,
        device="cuda",
    )
    tokens = token_source.next(4)
    print(tokens.detach().cpu().numpy())

def test_token_activation_source(dataloader, model):
    token_source = TokenSource(
        dataloader=dataloader,
        model=model,
        is_dataset_tokenized=False,
        seq_len=128,
        device="cuda",
    )
    act_source = TokenActivationSource(
        token_source=token_source,
        model=model,
        token_batch_size=32,
        act_name="activation",
        seq_len=128,
        d_model=768,
        device="cuda",
        dtype=torch.float32,
    )
    act = act_source.next(4)
    print(act["activation"].detach().cpu().numpy())

