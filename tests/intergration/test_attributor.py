from math import isclose

import torch
import torch.nn as nn
from transformer_lens.hook_points import HookedRootModule, HookPoint

from lm_saes.circuit.attributors import DirectAttributor, HierachicalAttributor, Node


class TestModule(HookedRootModule):
    __test__ = False

    def __init__(self):
        super().__init__()
        self.W_1 = nn.Parameter(torch.tensor([[1.0, 2.0]]))
        self.W_2 = nn.Parameter(torch.tensor([[1.0], [1.0]]))
        self.W_3 = nn.Parameter(torch.tensor([[1.0, 1.0]]))
        self.W_4 = nn.Parameter(torch.tensor([[2.0], [1.0]]))
        self.hook_mid_1 = HookPoint()
        self.hook_mid_2 = HookPoint()
        self.setup()

    def forward(self, input):
        input = input + self.hook_mid_1(input @ self.W_1) @ self.W_2
        input = input + self.hook_mid_2(input @ self.W_3) @ self.W_4
        return input


def test_direct_attributor():
    model = TestModule()
    attributor = DirectAttributor(model)
    input = torch.tensor([1.0])
    input = input.requires_grad_()
    circuit = attributor.attribute(input, Node(None, "0"), [Node("hook_mid_1"), Node("hook_mid_2")])
    assert len(circuit.nodes) == 5
    assert isclose(circuit.nodes[Node("hook_mid_1", "0")]["attribution"], 1.0)
    assert isclose(circuit.nodes[Node("hook_mid_1", "1")]["attribution"], 2.0)
    assert isclose(circuit.nodes[Node("hook_mid_2", "0")]["attribution"], 8.0)
    assert isclose(circuit.nodes[Node("hook_mid_2", "1")]["attribution"], 4.0)


def test_hierachical_attributor():
    model = TestModule()
    attributor = HierachicalAttributor(model)
    input = torch.tensor([1.0])
    input = input.requires_grad_()
    circuit = attributor.attribute(input, Node(None, "0"), [Node("hook_mid_1"), Node("hook_mid_2")])
    assert len(circuit.nodes) == 5
    assert isclose(circuit.nodes[Node("hook_mid_1", "0")]["attribution"], 4.0)
    assert isclose(circuit.nodes[Node("hook_mid_1", "1")]["attribution"], 8.0)
    assert isclose(circuit.nodes[Node("hook_mid_2", "0")]["attribution"], 8.0)
    assert isclose(circuit.nodes[Node("hook_mid_2", "1")]["attribution"], 4.0)


def test_hierachical_attributor_with_threshold():
    model = TestModule()
    attributor = HierachicalAttributor(model)
    input = torch.tensor([1.0])
    input = input.requires_grad_()
    circuit = attributor.attribute(input, Node(None, "0"), [Node("hook_mid_1"), Node("hook_mid_2")], threshold=5.0)
    assert len(circuit.nodes) == 3
    assert isclose(circuit.nodes[Node("hook_mid_1", "1")]["attribution"], 6.0)
    assert isclose(circuit.nodes[Node("hook_mid_2", "0")]["attribution"], 8.0)
