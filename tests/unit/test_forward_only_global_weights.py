from types import SimpleNamespace

import pytest
import torch

from llamascopium.circuits.global_weights import Atlas, ConnectedFeatures, ForwardOnlyReplacementModel
from llamascopium.circuits.global_weights.weights import GlobalConnection
from llamascopium.circuits.utils.batched_features import BatchedFeatures


def test_atlas_expands_upstream_and_downstream_with_signed_influence():
    seed_batch = BatchedFeatures.from_str_nodes(["L1M#0"], device="cpu")
    atlas = Atlas(seed_batch)
    atlas.update(
        seed_batch,
        [ConnectedFeatures(
            upstream_features=BatchedFeatures.from_str_nodes(["L0M#2"], device="cpu"),
            downstream_features=BatchedFeatures.from_str_nodes(["L2A#1"], device="cpu"),
            upstream_values=torch.tensor([2.0]),
            downstream_values=torch.tensor([-1.0]),
        )],
    )
    assert atlas.select_top_k_nodes_to_visit(1).to_str_nodes() == ["L0M#2"]
    assert atlas.nodes["L2A#1"].influence == -1.0
    assert len(atlas.export_to_json()["links"]) == 2


def test_forward_only_model_exposes_both_existing_scans(monkeypatch):
    import llamascopium.circuits.global_weights.forward_only_replacement_model as module

    target = ("blocks.1.hook_mlp_out", 0)
    source = ("blocks.0.hook_mlp_out", 1)
    edge = GlobalConnection(source[0], source[1], target[0], target[1], 2.0)
    global_result = SimpleNamespace(topk=lambda *args: [edge], downstream=lambda *args: [])
    inhibitory_result = SimpleNamespace(topk=lambda *args: ["inhibitory edge"])
    calls = []

    def compute_global(model, dictionaries, inputs, targets):
        calls.append(("global", list(inputs), targets))
        return global_result

    def compute_inhibitory(model, dictionaries, inputs, targets):
        calls.append(("inhibitory", list(inputs), targets))
        return inhibitory_result

    monkeypatch.setattr(module, "compute_global_weights", compute_global)
    monkeypatch.setattr(module, "compute_inhibitory_global_weights", compute_inhibitory)
    runner = ForwardOnlyReplacementModel(object(), [])
    runner.prepare_virtual_weights([target])
    assert runner.record_activation_matrix(iter(["first"])) is global_result
    connected = runner.parse_global_weight_results()[0]
    assert connected.upstream_features.to_str_nodes() == ["L0M#1"]
    assert connected.upstream_values.tolist() == [2.0]
    assert runner.record_activation_matrix(iter(["second"]), mode="inhibitory") is inhibitory_result
    assert runner.parse_inhibitory_results() == [["inhibitory edge"]]
    assert calls == [("global", ["first"], [target]), ("inhibitory", ["second"], [target])]
    with pytest.raises(ValueError, match="Unknown"):
        runner.record_activation_matrix([], mode="invalid")
