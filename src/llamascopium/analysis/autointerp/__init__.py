"""Prompt builders for auto-interpretation of SAE features.

This package contains modules for generating prompts used in the auto-interpretation
process, organized by purpose:
- explanation_prompts: Prompts for generating feature explanations
- evaluation_prompts: Prompts for evaluating feature explanations
"""

from lm_saes.analysis.samples import Segment, TokenizedSample, process_token

from .autointerp_base import (
    AutoInterpConfig,
    ExplainerType,
    ScorerType,
)
from .evaluation_prompts import (
    generate_detection_prompt,
    generate_fuzzing_prompt,
)
from .explanation_prompts import (
    generate_explanation_prompt,
    generate_explanation_prompt_neuronpedia,
)
from .feature_interpreter import (
    FeatureInterpreter,
)

__all__ = [
    "generate_explanation_prompt",
    "generate_explanation_prompt_neuronpedia",
    "generate_detection_prompt",
    "generate_fuzzing_prompt",
    "FeatureInterpreter",
    "AutoInterpConfig",
    "ExplainerType",
    "ScorerType",
    "Segment",
    "TokenizedSample",
    "process_token",
    "FeatureInterpreter",
]
