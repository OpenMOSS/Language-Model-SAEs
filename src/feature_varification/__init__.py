"""Reusable utilities for verifying whether a chess SAE feature matches a rule.

This package is meant to replace the ad-hoc taxonomy notebook pattern:

1. manually define one rule inside a notebook
2. manually choose an activation threshold
3. manually run the model and compute accuracy / recall / F1

The new package keeps those concerns separate:

- ``rules`` defines *what squares should be positive*
- ``thresholds`` defines *when a real-valued feature counts as active*
- ``evaluator`` defines *how activations and labels are compared*

Typical usage
-------------
```python
from feature_varification import (
    FeatureSpec,
    ThresholdSpec,
    VerificationCase,
    PieceFrontSpanRule,
    evaluate_feature_rule,
)

feature = FeatureSpec(feature_type="transcoder", layer=10, feature_id=1958)
rule = PieceFrontSpanRule(piece_type="own k", include_adjacent_files=True)
threshold = ThresholdSpec(mode="ratio_to_max", value=0.7, scope="sample")

result = evaluate_feature_rule(
    model=model,
    lorsas=lorsas,
    transcoders=transcoders,
    feature=feature,
    rule=rule,
    cases=[VerificationCase(fen=fen, move_uci=move) for fen, move in rows],
    threshold=threshold,
)
```
"""

from .evaluator import collect_feature_activations, evaluate_feature_rule, extract_feature_activations
from .rules import (
    AllOfRule,
    AnyOfRule,
    FunctionalRule,
    KingNeighborhoodRule,
    MoveEndSquareRule,
    MoveStartSquareRule,
    PieceDestinationRule,
    PieceFrontSpanRule,
    PieceNeighborhoodRule,
    PieceRayRule,
    PieceTypeRule,
    QueenCheckAroundOpponentKingRule,
    RelativeOffsetRule,
    RULE_REGISTRY,
    VerificationRule,
    front_cone_rule,
    front_file_rule,
    register_rule,
    same_diagonal_rule,
    same_file_rule,
    same_rank_rule,
)
from .thresholds import resolve_thresholds
from .types import (
    FeatureSpec,
    FeatureVerificationResult,
    FeatureType,
    RuleEvaluation,
    ThresholdSpec,
    VerificationCase,
    VerificationCounts,
)

__all__ = [
    "AllOfRule",
    "AnyOfRule",
    "FeatureSpec",
    "FeatureType",
    "FeatureVerificationResult",
    "FunctionalRule",
    "KingNeighborhoodRule",
    "MoveEndSquareRule",
    "MoveStartSquareRule",
    "PieceDestinationRule",
    "PieceFrontSpanRule",
    "PieceNeighborhoodRule",
    "PieceRayRule",
    "PieceTypeRule",
    "QueenCheckAroundOpponentKingRule",
    "RULE_REGISTRY",
    "RelativeOffsetRule",
    "RuleEvaluation",
    "ThresholdSpec",
    "VerificationCase",
    "VerificationCounts",
    "VerificationRule",
    "collect_feature_activations",
    "evaluate_feature_rule",
    "extract_feature_activations",
    "front_cone_rule",
    "front_file_rule",
    "register_rule",
    "resolve_thresholds",
    "same_diagonal_rule",
    "same_file_rule",
    "same_rank_rule",
]
