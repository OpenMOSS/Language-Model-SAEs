from transformer_lens.hook_points import HookedRootModule, HookPoint
import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensor

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.subblock1 = nn.Linear(10, 10)
        self.subblock2 = nn.Linear(10, 10)
        self.activation = nn.ReLU()
        self.hook_mid = HookPoint()

    def forward(self, x):
        return self.subblock2(self.hook_mid(self.activation(self.subblock1(x))))

class TestModule(HookedRootModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = nn.ModuleList([Block() for _ in range(3)])
        self.embed = nn.Linear(1, 10)
        self.unembed = nn.Linear(10, 1)
        self.setup()

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.unembed(x)


def test_fake_params_after():
    model = TestModule()
    _, cache_before = model.run_with_cache(torch.tensor([0.]))

    model.fake_params_after("blocks.1.hook_mid", torch.tensor([0.]))
    assert not isinstance(model.blocks[0].subblock1.weight, FakeTensor)
    assert not isinstance(model.blocks[1].subblock1.weight, FakeTensor)
    assert isinstance(model.blocks[2].subblock1.weight, FakeTensor)
    assert isinstance(model.unembed.weight, FakeTensor)

    out, cache_after = model.run_with_cache(torch.tensor([0.]))
    assert torch.allclose(cache_before["blocks.0.hook_mid"], cache_after["blocks.0.hook_mid"])
    assert torch.allclose(cache_before["blocks.1.hook_mid"], cache_after["blocks.1.hook_mid"])
    assert isinstance(cache_after["blocks.2.hook_mid"], FakeTensor)
    assert len(out.shape) == 1 and out.shape[0] == 1