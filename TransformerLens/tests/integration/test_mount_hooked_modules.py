from transformer_lens.hook_points import HookedRootModule, HookPoint
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.subblock1 = nn.Linear(10, 10)
        self.subblock2 = nn.Linear(10, 10)
        self.activation = nn.ReLU()
        self.hook_pre = HookPoint()
        self.hook_mid = HookPoint()

    def forward(self, x):
        return self.subblock2(self.hook_mid(self.activation(self.subblock1(self.hook_pre(x)))))

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
    
class TestMountModule(HookedRootModule):
    __test__ = False
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hook_mid = HookPoint()
        self.setup()

    def forward(self, x):
        return self.hook_mid(x) * 2
    
def test_apply_hooked_modules():
    model = TestModule()
    model_to_mount = TestMountModule()
    with model.mount_hooked_modules([("blocks.0.hook_mid", "m", model_to_mount)]):
        assert model.blocks[0].hook_mid.m == model_to_mount
        assert model_to_mount.hook_mid.name == "blocks.0.hook_mid.m.hook_mid"
        assert "blocks.0.hook_mid.m.hook_mid" in model.hook_dict
    assert not hasattr(model.blocks[0].hook_mid, "m")
    assert model_to_mount.hook_mid.name == "hook_mid"