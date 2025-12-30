"""Prompt builders for evaluating feature explanations.

This module contains functions for generating prompts used to evaluate SAE feature
explanations, including detection and fuzzing evaluation methods.
"""

from typing import Any

from lm_saes.analysis.autointerp.autointerp_base import AutoInterpConfig
from lm_saes.analysis.samples import TokenizedSample


def generate_detection_prompt(
    cfg: AutoInterpConfig,
    explanation: dict[str, Any],
    examples: list[TokenizedSample],
) -> tuple[str, str]:
    """Generate a prompt for detection evaluation.

    Args:
        cfg: Auto-interpretation configuration
        explanation: The explanation to evaluate
        examples: List of examples (mix of activating and non-activating)

    Returns:
        Tuple of (system_prompt, user_prompt) strings
    """
    system_prompt = f"""We're studying features in a neural network. Each feature activates on some particular word/words/substring/concept in a short document. You will be given a short explanation of what this feature activates for, and then be shown {len(examples)} example sequences in random order. You will have to return a boolean list of the examples where you think the feature should activate at least once, on ANY of the words or substrings in the document, true if it does, false if it doesn't. Try not to be overly specific in your interpretation of the explanation."""
    system_prompt += """
Your output should be a JSON object that has the following fields: `steps`, `evaluation_results`. `steps` should be an array of strings, each representing a step in the chain-of-thought process within 50 words. `evaluation_results` should be an array of booleans, each representing whether the feature should activate on the corresponding example.
"""
    user_prompt = f"Here is the explanation:\n\n{explanation['final_explanation']}\n\nHere are the examples:\n\n"

    for i, example in enumerate(examples, 1):
        user_prompt += f"Example {i}: {example.display_plain()}\n"

    return system_prompt, user_prompt


def generate_fuzzing_prompt(
    cfg: AutoInterpConfig,
    explanation: dict[str, Any],
    examples: list[tuple[TokenizedSample, bool]],  # (sample, is_correctly_marked)
) -> tuple[str, str]:
    """Generate a prompt for fuzzing evaluation.

    Args:
        cfg: Auto-interpretation configuration
        explanation: The explanation to evaluate
        examples: List of tuples (example, is_correctly_marked)

    Returns:
        Tuple of (system_prompt, user_prompt) strings
    """
    system_prompt = f"""We're studying features in a neural network. Each feature activates on some particular word/words/substring/concept in a short document. You will be given a short explanation of what this feature activates for, and then be shown {len(examples)} example sequences in random order. In each example, text segments highlighted with << >> are presented as activating the feature as described in the explanation. You will have to return a boolean list of the examples where you think the highlighted parts CORRECTLY correspond to the explanation, true if they do, false if they don't. Try not to be overly specific in your interpretation of the explanation."""
    system_prompt += """
Your output should be a JSON object that has the following fields: `steps`, `evaluation_results`. `steps` should be an array of strings, each representing a step in the chain-of-thought process within 50 words. `evaluation_results` should be an array of booleans, each representing whether the feature should activate on the corresponding example.
"""
    user_prompt = f"Here is the explanation:\n\n{explanation['final_explanation']}\n\nHere are the examples:\n\n"

    for i, (example, _) in enumerate(examples, 1):
        highlighted = example.display_highlighted(cfg.activation_threshold)
        user_prompt += f"Example {i}: {highlighted}\n"

    return system_prompt, user_prompt
