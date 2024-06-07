from transformer_lens import HookedTransformer
import torch
from torch._subclasses.fake_tensor import FakeTensor

MODEL = "solu-2l"

model = HookedTransformer.from_pretrained(MODEL, device="cuda")

def test_fake_params_after():
    allocated_before = torch.cuda.memory_allocated(0)
    model.fake_params_after("blocks.0.hook_resid_post", torch.tensor([[0.]], device="cuda"))
    allocated_after = torch.cuda.memory_allocated(0)
    assert allocated_after < allocated_before * 0.55