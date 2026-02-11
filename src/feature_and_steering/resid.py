import torch
from transformer_lens import HookedTransformer

def resid_patching(
    model: HookedTransformer,
    fen: str,
    fen_corrupted: str,
    hook_point: str,
    position: int,
    *,
    original_out: object | None = None,
    original_cache: object | None = None,
    cache_corrupted: object | None = None,
) -> dict[str, object]:
    def _reset() -> None:
        if hasattr(model, "reset_hooks"):
            try:
                model.reset_hooks(including_permanent=True)
            except TypeError:
                model.reset_hooks()

    if original_out is None or original_cache is None:
        with torch.no_grad():
            original_out, original_cache = model.run_with_cache(fen, prepend_bos=False)
        _reset()

    if cache_corrupted is None:
        with torch.no_grad():
            _, cache_corrupted = model.run_with_cache(fen_corrupted, prepend_bos=False)
        _reset()

    assert cache_corrupted is not None
    corrupted_act = cache_corrupted[hook_point]

    def _patch(act: torch.Tensor, hook):
        out = act.clone()
        if out.dim() == 3:
            # out[:, position, :] += corrupted_act[:, position, :].to(out.device)
            out[:, position, :] = corrupted_act[:, position, :].to(out.device)
        elif out.dim() == 2:
            # out[position, :] += corrupted_act[position, :].to(out.device)
            out[position, :] = corrupted_act[position, :].to(out.device)
        else:
            raise ValueError(f"unexpected activation shape at {hook_point}: {tuple(out.shape)}")
        return out

    with model.hooks(fwd_hooks=[(hook_point, _patch)]), torch.no_grad():
        patched_out, patched_cache = model.run_with_cache(fen, prepend_bos=False)
    _reset()

    return {
        "original_out": original_out,
        "original_cache": original_cache,
        "patched_out": patched_out,
        "patched_cache": patched_cache,
    }
    
    
def resid_patching_multi_pos(
    model: HookedTransformer,
    fen: str,
    fen_corrupted: str,
    hook_point: str,
    positions: list[int],
    *,
    original_out=None,
    original_cache=None,
    cache_corrupted=None,
):
    def _reset():
        if hasattr(model, "reset_hooks"):
            try:
                model.reset_hooks(including_permanent=True)
            except TypeError:
                model.reset_hooks()

    if original_out is None or original_cache is None:
        with torch.no_grad():
            original_out, original_cache = model.run_with_cache(fen, prepend_bos=False)
        _reset()

    if cache_corrupted is None:
        with torch.no_grad():
            _, cache_corrupted = model.run_with_cache(fen_corrupted, prepend_bos=False)
        _reset()

    corrupted_act = cache_corrupted[hook_point]

    def _patch(act, hook):
        out = act.clone()
        for pos in positions:
            if out.dim() == 3:
                out[:, pos, :] = corrupted_act[:, pos, :].to(out.device)
            elif out.dim() == 2:
                out[pos, :] = corrupted_act[pos, :].to(out.device)
            else:
                raise ValueError(f"unexpected shape {out.shape}")
        return out

    with model.hooks(fwd_hooks=[(hook_point, _patch)]), torch.no_grad():
        patched_out, patched_cache = model.run_with_cache(fen, prepend_bos=False)

    _reset()

    return {
        "original_out": original_out,
        "patched_out": patched_out,
    }