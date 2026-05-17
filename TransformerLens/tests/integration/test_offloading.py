from transformer_lens.hook_points import HookedRootModule, HookPoint
import torch
import torch.nn as nn

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
    __test__ = False
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
    
def test_run_with_cache_until():
    model = TestModule()
    _, cache_before = model.run_with_cache(torch.tensor([1.]), names_filter=["blocks.0.hook_mid", "blocks.1.hook_mid"])
    out, cache_after = model.run_with_cache_until(torch.tensor([1.]), names_filter=["blocks.0.hook_mid", "blocks.1.hook_mid"])

    assert torch.allclose(cache_before["blocks.0.hook_mid"], cache_after["blocks.0.hook_mid"])
    assert torch.allclose(cache_before["blocks.1.hook_mid"], cache_after["blocks.1.hook_mid"])
    assert torch.allclose(cache_before["blocks.1.hook_mid"], out)

def test_offload_params_after():
    model = TestModule()
    _, cache_before = model.run_with_cache(torch.tensor([1.]))

    model.offload_params_after("blocks.1.hook_mid", torch.tensor([1.]))
    assert model.blocks[0].subblock1.weight is not None
    assert model.blocks[1].subblock1.weight is not None
    assert model.blocks[2].subblock1.weight is None
    assert model.unembed.weight is None

    _, cache_after = model.run_with_cache_until(torch.tensor([1.]), names_filter=["blocks.0.hook_mid", "blocks.1.hook_mid"])
    assert torch.allclose(cache_before["blocks.0.hook_mid"], cache_after["blocks.0.hook_mid"])
    assert torch.allclose(cache_before["blocks.1.hook_mid"], cache_after["blocks.1.hook_mid"])
