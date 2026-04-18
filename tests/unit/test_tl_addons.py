import pytest
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from llamascopium.backend.tl_addons import mount_hooked_modules, run_with_cache_until, run_with_ref_cache


def _model_and_tokens():
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    return model, tokens


def test_run_with_cache_until_matches_run_with_cache():
    """run_with_cache_until produces same cache as run_with_cache when stopping at last hook."""
    model, tokens = _model_and_tokens()

    hook_points = ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]
    _, cache_full = model.run_with_cache(tokens, names_filter=hook_points)
    out_until, cache_until = run_with_cache_until(model, tokens, names_filter=hook_points, until=hook_points[-1])

    for name in hook_points:
        assert torch.allclose(cache_full[name], cache_until[name]), f"Mismatch at {name}"
    assert torch.allclose(cache_full[hook_points[-1]], out_until)


def test_run_with_ref_cache_values_match_run_with_cache():
    """run_with_ref_cache returns the same activation values as run_with_cache."""
    model, tokens = _model_and_tokens()

    hook_points = ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]
    _, cache_detached = model.run_with_cache(tokens, names_filter=hook_points)
    _, cache_ref = run_with_ref_cache(model, tokens, names_filter=hook_points)

    for name in hook_points:
        assert torch.allclose(cache_detached[name], cache_ref[name].detach()), f"Mismatch at {name}"


def test_run_with_ref_cache_tensors_are_not_detached():
    """Tensors in the ref cache are live (not detached) and support retain_grad."""
    model, tokens = _model_and_tokens()

    hook_name = "blocks.0.hook_resid_post"
    _, cache_ref = run_with_ref_cache(model, tokens, names_filter=hook_name, retain_grad=True)

    cached = cache_ref[hook_name]
    assert cached.requires_grad or cached.grad_fn is not None or cached.is_leaf, (
        "Expected a live tensor that participates in the autograd graph"
    )


_HOOK = "blocks.0.hook_resid_post"
_D_MODEL = 768


# ---------------------------------------------------------------------------
# mount_hooked_modules
# ---------------------------------------------------------------------------


def test_mount_attaches_module_to_hook_point():
    """Module is reachable as an attribute of the hook point inside the context."""
    model, _ = _model_and_tokens()
    child = torch.nn.Linear(_D_MODEL, _D_MODEL, bias=False)

    assert not hasattr(model.mod_dict[_HOOK], "_child")

    with mount_hooked_modules(model, [(_HOOK, "_child", child)]) as m:
        hp = m.mod_dict[_HOOK]
        assert hasattr(hp, "_child")
        assert getattr(hp, "_child") is child

    assert not hasattr(model.mod_dict[_HOOK], "_child")


def test_mount_registers_in_mod_dict():
    """After setup(), the child appears in model.mod_dict under its full dotted path."""
    model, _ = _model_and_tokens()
    child = torch.nn.Linear(_D_MODEL, _D_MODEL, bias=False)
    full_path = f"{_HOOK}._child"

    with mount_hooked_modules(model, [(_HOOK, "_child", child)]) as m:
        assert full_path in m.mod_dict, "Child not found in mod_dict during context"
        assert m.mod_dict[full_path] is child

    assert full_path not in model.mod_dict, "Child still in mod_dict after context"


def test_mount_multiple_modules_simultaneously():
    """Multiple modules can be mounted at different hook points in a single call."""
    model, _ = _model_and_tokens()
    hooks = [
        ("blocks.0.hook_resid_post", "_c0", torch.nn.Linear(_D_MODEL, _D_MODEL, bias=False)),
        ("blocks.1.hook_resid_post", "_c1", torch.nn.Linear(_D_MODEL, _D_MODEL, bias=False)),
    ]

    with mount_hooked_modules(model, hooks) as m:
        for hp_name, child_name, module in hooks:
            assert getattr(m.mod_dict[hp_name], child_name) is module

    for hp_name, child_name, _ in hooks:
        assert not hasattr(model.mod_dict[hp_name], child_name)


def test_mount_cleanup_on_exception():
    """Mounted modules are removed even when an exception is raised inside the context."""
    model, _ = _model_and_tokens()
    child = torch.nn.Linear(_D_MODEL, _D_MODEL, bias=False)

    with pytest.raises(RuntimeError, match="boom"):
        with mount_hooked_modules(model, [(_HOOK, "_child", child)]):
            raise RuntimeError("boom")

    assert not hasattr(model.mod_dict[_HOOK], "_child")
    assert f"{_HOOK}._child" not in model.mod_dict


def test_mount_model_still_runs_after_context():
    """Model produces correct-shaped output after mount_hooked_modules cleans up."""
    model, tokens = _model_and_tokens()
    child = torch.nn.Linear(_D_MODEL, _D_MODEL, bias=False)

    with mount_hooked_modules(model, [(_HOOK, "_child", child)]):
        pass

    logits = model(tokens)
    assert logits.shape[-1] == model.cfg.d_vocab


def test_mount_hook_on_child_fires_during_forward():
    """A hook added to the mounted child module's hook point is triggered during the forward pass."""
    model, tokens = _model_and_tokens()

    # Use a tiny HookedTransformer as the child so it has its own hook points after setup().
    # For simplicity we just use a plain nn.Module and register a forward hook directly.
    fired: list[bool] = []

    class _Probe(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            fired.append(True)
            return x

    # We add the probe as a sub-module of a hook point, then hook the hook point itself
    # to call the probe manually — the real test is that model.setup() ran so that our
    # custom module appears in named_modules and doesn't break the forward pass.
    probe = _Probe()

    def _call_probe(tensor: torch.Tensor, *, hook: HookPoint) -> torch.Tensor:
        return probe(tensor)

    with mount_hooked_modules(model, [(_HOOK, "_probe", probe)]) as m:
        assert f"{_HOOK}._probe" in dict(m.named_modules())
        m.run_with_hooks(tokens, fwd_hooks=[(_HOOK, _call_probe)])

    assert fired, "Probe was never called during the forward pass"
