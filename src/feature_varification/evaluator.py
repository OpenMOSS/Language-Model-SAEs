from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from tqdm.auto import tqdm

from .rules import RULE_REGISTRY, VerificationRule
from .thresholds import resolve_thresholds
from .types import (
    FeatureSpec,
    FeatureVerificationResult,
    ThresholdSpec,
    VerificationCase,
    VerificationCounts,
)


def _normalize_cases(
    cases: Sequence[VerificationCase | str],
    move_uci_list: Sequence[str | None] | None = None,
) -> list[VerificationCase]:
    """Convert lightweight user input into ``VerificationCase`` objects.

    Supporting both ``list[str]`` and ``list[VerificationCase]`` keeps notebook
    migration simple:

    - old notebook code usually has ``fen_list``
    - move-based rules additionally have ``move_uci_list``
    - new code can pass richer ``VerificationCase`` objects directly
    """

    if move_uci_list is not None and len(move_uci_list) != len(cases):
        raise ValueError("move_uci_list must have the same length as cases")

    normalized: list[VerificationCase] = []
    for idx, case in enumerate(cases):
        if isinstance(case, VerificationCase):
            normalized.append(case)
            continue
        move_uci = None if move_uci_list is None else move_uci_list[idx]
        normalized.append(VerificationCase(fen=str(case), move_uci=move_uci, label=f"case_{idx:04d}"))
    return normalized


def _resolve_rule(rule: str | VerificationRule) -> VerificationRule:
    """Look up a rule by name or return the rule instance directly."""

    if isinstance(rule, str):
        if rule not in RULE_REGISTRY:
            available = ", ".join(sorted(RULE_REGISTRY))
            raise KeyError(f"Unknown verification rule '{rule}'. Available rules: {available}")
        return RULE_REGISTRY[rule]
    return rule


def _get_sae_and_hook_name(
    lorsas: Sequence[Any],
    transcoders: dict[int, Any] | dict[str, Any],
    feature: FeatureSpec,
) -> tuple[Any, str]:
    """Resolve which SAE module and which model hook are used by a feature.

    The taxonomy notebooks use two different residual spaces:

    - transcoder features are encoded from ``resid_mid_after_ln``
    - lorsa features are encoded from ``hook_attn_in``

    Keeping this mapping in one helper avoids repeating the same fragile
    ``if feature_type == ...`` block everywhere.
    """

    if feature.feature_type == "transcoder":
        sae = transcoders.get(feature.layer)
        if sae is None:
            sae = transcoders.get(str(feature.layer))
        if sae is None:
            raise KeyError(f"Could not find transcoder for layer {feature.layer}")
        return sae, f"blocks.{feature.layer}.resid_mid_after_ln"

    if feature.feature_type == "lorsa":
        if not (0 <= feature.layer < len(lorsas)):
            raise KeyError(f"Could not find lorsa for layer {feature.layer}")
        return lorsas[feature.layer], f"blocks.{feature.layer}.hook_attn_in"

    raise ValueError(f"Unsupported feature type: {feature.feature_type}")


@torch.no_grad()
def extract_feature_activations(
    *,
    model: Any,
    lorsas: Sequence[Any],
    transcoders: dict[int, Any] | dict[str, Any],
    feature: FeatureSpec,
    fen: str,
    prepend_bos: bool = False,
) -> torch.Tensor:
    """Return one feature's activation vector for a single FEN.

    Returns
    -------
    torch.Tensor
        Shape ``[64]``. Each entry corresponds to one BT4 board position.
    """

    sae, hook_name = _get_sae_and_hook_name(lorsas, transcoders, feature)
    _, cache = model.run_with_cache(fen, prepend_bos=prepend_bos)
    encoded = sae.encode(cache[hook_name])

    # Some models return [1, 64, d_sae], others return [64, d_sae].
    if encoded.dim() == 3:
        acts = encoded[0, :, feature.feature_id]
    elif encoded.dim() == 2:
        acts = encoded[:, feature.feature_id]
    else:
        raise ValueError(f"Unexpected encoded activation shape: {tuple(encoded.shape)}")

    return acts.detach().to(torch.float32).cpu()


@torch.no_grad()
def collect_feature_activations(
    *,
    model: Any,
    lorsas: Sequence[Any],
    transcoders: dict[int, Any] | dict[str, Any],
    feature: FeatureSpec,
    cases: Sequence[VerificationCase | str],
    move_uci_list: Sequence[str | None] | None = None,
    prepend_bos: bool = False,
    show_progress: bool = True,
) -> tuple[list[VerificationCase], list[torch.Tensor]]:
    """Pre-compute activations for an evaluation set.

    This is useful when the same cached activations will be re-used under
    several threshold choices. The function returns both normalized cases and
    their matching activation tensors so callers do not need to keep track of
    index alignment manually.
    """

    normalized_cases = _normalize_cases(cases, move_uci_list=move_uci_list)
    activations: list[torch.Tensor] = []
    iterator = normalized_cases
    if show_progress:
        iterator = tqdm(normalized_cases, desc=f"Collecting {feature.feature_type} L{feature.layer} F{feature.feature_id}")

    for case in iterator:
        activations.append(
            extract_feature_activations(
                model=model,
                lorsas=lorsas,
                transcoders=transcoders,
                feature=feature,
                fen=case.fen,
                prepend_bos=prepend_bos,
            )
        )
    return normalized_cases, activations


def _safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _safe_max(values: list[float]) -> float | None:
    return max(values) if values else None


def _top_positions(values: torch.Tensor, *, k: int = 5) -> list[dict[str, float]]:
    """Return a short summary of the strongest positions for debugging output."""

    if values.numel() == 0:
        return []
    k = min(k, values.numel())
    top_values, top_indices = torch.topk(values, k=k)
    return [
        {"position": int(position.item()), "activation": float(value.item())}
        for value, position in zip(top_values, top_indices)
    ]


def evaluate_feature_rule(
    *,
    model: Any,
    lorsas: Sequence[Any],
    transcoders: dict[int, Any] | dict[str, Any],
    feature: FeatureSpec,
    rule: str | VerificationRule,
    cases: Sequence[VerificationCase | str],
    move_uci_list: Sequence[str | None] | None = None,
    threshold: ThresholdSpec | None = None,
    prepend_bos: bool = False,
    show_progress: bool = True,
    max_examples: int = 20,
) -> FeatureVerificationResult:
    """Evaluate whether a feature matches a rule-defined spatial pattern.

    The evaluation is position-level. For each FEN:

    1. run the model and extract one feature activation value per board square
    2. convert activations into a binary active/inactive mask via ``threshold``
    3. ask the rule for a binary supervision mask on the same 64 squares
    4. accumulate TP / FP / TN / FN over all positions

    ``threshold`` supports both of the use cases you explicitly asked for:

    - fixed activation threshold:
      ``ThresholdSpec(mode="absolute", value=0.8)``
    - threshold as a fraction of maximum activation:
      ``ThresholdSpec(mode="ratio_to_max", value=0.7, scope="sample")``
    """

    normalized_cases, activations_per_case = collect_feature_activations(
        model=model,
        lorsas=lorsas,
        transcoders=transcoders,
        feature=feature,
        cases=cases,
        move_uci_list=move_uci_list,
        prepend_bos=prepend_bos,
        show_progress=show_progress,
    )
    resolved_rule = _resolve_rule(rule)
    threshold = threshold or ThresholdSpec(mode="absolute", value=0.0, scope="dataset")
    thresholds = resolve_thresholds(activations_per_case, threshold)

    counts = VerificationCounts()
    active_positions = 0
    total_positions = 0
    on_target_values: list[float] = []
    off_target_values: list[float] = []
    example_rows: list[dict[str, Any]] = []

    for case, activations, resolved_threshold in zip(normalized_cases, activations_per_case, thresholds):
        rule_result = resolved_rule.evaluate(case.fen, case.move_uci)
        if len(rule_result.mask) != int(activations.shape[0]):
            raise ValueError(
                f"Rule '{resolved_rule.name}' produced mask length {len(rule_result.mask)} "
                f"for activations of length {int(activations.shape[0])}"
            )

        target_mask = torch.tensor(rule_result.mask, dtype=torch.bool)
        active_mask = activations > float(resolved_threshold)

        tp_mask = active_mask & target_mask
        fp_mask = active_mask & ~target_mask
        tn_mask = ~active_mask & ~target_mask
        fn_mask = ~active_mask & target_mask

        counts.tp += int(tp_mask.sum().item())
        counts.fp += int(fp_mask.sum().item())
        counts.tn += int(tn_mask.sum().item())
        counts.fn += int(fn_mask.sum().item())

        active_positions += int(active_mask.sum().item())
        total_positions += int(active_mask.numel())

        on_target_values.extend(activations[target_mask].tolist())
        off_target_values.extend(activations[~target_mask].tolist())

        # Store a small number of representative rows. These are invaluable
        # when debugging a feature that looks good numerically but actually
        # fires on the wrong subset of squares.
        if len(example_rows) < max_examples:
            has_error = bool(fp_mask.any().item() or fn_mask.any().item())
            has_signal = bool(active_mask.any().item() or target_mask.any().item())
            if has_error or has_signal:
                example_rows.append(
                    {
                        "label": case.label,
                        "fen": case.fen,
                        "move_uci": case.move_uci,
                        "metadata": case.metadata,
                        "rule_metadata": rule_result.metadata,
                        "threshold": float(resolved_threshold),
                        "n_positive_positions": int(target_mask.sum().item()),
                        "n_active_positions": int(active_mask.sum().item()),
                        "tp_positions": torch.nonzero(tp_mask, as_tuple=False).reshape(-1).tolist(),
                        "fp_positions": torch.nonzero(fp_mask, as_tuple=False).reshape(-1).tolist(),
                        "fn_positions": torch.nonzero(fn_mask, as_tuple=False).reshape(-1).tolist(),
                        "top_activations": _top_positions(activations),
                    }
                )

    return FeatureVerificationResult(
        feature=feature,
        rule_name=resolved_rule.name,
        threshold=threshold,
        counts=counts,
        n_fens_evaluated=len(normalized_cases),
        active_positions=active_positions,
        total_positions=total_positions,
        max_activation_on_target=_safe_max(on_target_values),
        max_activation_off_target=_safe_max(off_target_values),
        mean_activation_on_target=_safe_mean(on_target_values),
        mean_activation_off_target=_safe_mean(off_target_values),
        example_rows=example_rows,
    )
