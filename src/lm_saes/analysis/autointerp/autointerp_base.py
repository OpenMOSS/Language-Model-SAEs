"""Utility classes and functions for auto-interpretation of SAE features.

This module contains shared utilities used across the auto-interpretation system,
including configuration, data structures, and helper functions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch
from pydantic import Field

from lm_saes.config import BaseConfig
from lm_saes.utils.logging import get_logger

logger = get_logger("analysis.autointerp_utils")


def process_token(token: str) -> str:
    """Process a token string by replacing special characters.

    Args:
        token: The token string to process

    Returns:
        Processed token string with special characters replaced
    """
    return token.replace("\n", "⏎").replace("\t", "→").replace("\r", "↵")


class ExplainerType(str, Enum):
    """Types of LLM explainers supported."""

    OPENAI = "openai"
    NEURONPEDIA = "neuronpedia"


class ScorerType(str, Enum):
    """Types of explanation scoring methods."""

    DETECTION = "detection"
    FUZZING = "fuzzing"
    GENERATION = "generation"
    SIMULATION = "simulation"


class AutoInterpConfig(BaseConfig):
    """Configuration for automatic interpretation of SAE features."""

    # LLM settings
    explainer_type: ExplainerType = ExplainerType.OPENAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_base_url: Optional[str] = None
    openai_proxy: Optional[str] = None

    # Activation retrieval settings
    n_activating_examples: int = 7
    n_non_activating_examples: int = 20
    activation_threshold: float = 0.7  # Threshold relative to max activation for highlighting tokens
    max_length: int = 50

    # Scoring settings
    scorer_type: list[ScorerType] = Field(default_factory=lambda: [ScorerType.DETECTION, ScorerType.FUZZING])

    # Detection settings
    detection_n_examples: int = 5  # Number of examples to show for detection

    # Fuzzing settings
    fuzzing_n_examples: int = 5  # Number of examples to use for fuzzing
    fuzzing_decile_correct: int = 5  # Number of correctly marked examples per decile
    fuzzing_decile_incorrect: int = 2  # Number of incorrectly marked examples per decile

    # Prompting settings
    include_cot: bool = True  # Whether to use chain-of-thought prompting
    overwrite_existing: bool = False  # Whether to overwrite existing interpretations


@dataclass
class Segment:
    """A segment of text with its activation value."""

    text: str
    """The text of the segment."""

    activation: float
    """The activation value of the segment."""

    def display(self, abs_threshold: float) -> str:
        """Display the segment as a string with whether it's highlighted."""
        if self.activation > abs_threshold:
            return f"<<{self.text}>>"
        else:
            return self.text

    def display_max(self, abs_threshold: float) -> str:
        """Display the segment text if it exceeds the threshold."""
        if self.activation > abs_threshold:
            return f"{self.text}\n"
        else:
            return ""

@dataclass
class ZPatternSegment:
    """Data for a z pattern of a single token."""

    contributing_indices: list[int]
    """The indices of the contributing tokens in the sequence."""
    contributions: list[float]
    """The contributions of the contributing tokens to the activation of the token."""
    max_contribution: float
    """The maximum contribution of the contributing tokens to the activation of the token."""

@dataclass
class TokenizedSample:
    """A tokenized sample with its activation pattern organized into segments."""

    segments: list[Segment]
    """List of segments, each containing start/end positions and activation values."""

    max_activation: float
    """Global maximum activation value."""

    z_pattern_data: dict[int, ZPatternSegment] | None = None

    def display_highlighted(self, threshold: float = 0.7) -> str:
        """Get the text with activating segments highlighted with << >> delimiters.

        Args:
            threshold: Threshold relative to max activation for highlighting

        Returns:
            Text with activating segments highlighted
        """
        highlighted_text = "".join([seg.display(threshold * self.max_activation) for seg in self.segments])
        return highlighted_text

    def display_plain(self) -> str:
        """Get the text with all segments displayed."""
        return "".join([seg.text for seg in self.segments])

    def display_max(self, threshold: float = 0.7) -> str:
        """Get the text with max activating tokens and their context."""
        max_activation_text = ""
        hash_ = {}
        for i, seg in enumerate(self.segments):
            if seg.activation > threshold * self.max_activation:
                text = seg.text
                if text != "" and hash_.get(text, None) is None:
                    hash_[text] = 1
                    prev_text = "".join([self.segments[idx].text for idx in range(max(0, i - 3), i)])
                    if self.z_pattern_data is not None and i in self.z_pattern_data:
                        z_pattern_segment = self.z_pattern_data[i]
                        k_prev_tokens = [f"({process_token(''.join([self.segments[idx].text for idx in range(max(0, j - 3), j)]))}) {process_token(self.segments[j].text)}" 
                                         for j, contribution in zip(z_pattern_segment.contributing_indices, z_pattern_segment.contributions)
                                         if contribution > threshold * z_pattern_segment.max_contribution]
                        contributing_text = f"[{'; '.join(k_prev_tokens)}] => "
                        max_activation_text += contributing_text
                    max_activation_text += f"({process_token(prev_text)}) {process_token(text)}\n"
        return max_activation_text

    def display_next(self, threshold: float = 0.7) -> str:
        """Get the token immediately after the max activating token."""
        next_activation_text = ""
        hash_ = {}
        Flag = False
        for seg in self.segments:
            if Flag:
                text = seg.text
                if text != "" and hash_.get(text, None) is None:
                    hash_[text] = 1
                    next_activation_text = process_token(text) + "\n"
            if seg.activation > threshold * self.max_activation:
                Flag = True
            else:
                Flag = False
        return next_activation_text
    
    def add_z_pattern_data(
        self,
        z_pattern_indices: torch.Tensor,
        z_pattern_values: torch.Tensor,
        origins: list[dict[str, Any]]
    ):
        self.z_pattern_data = {}
        activating_indices = z_pattern_indices[0].unique_consecutive()
        for i in activating_indices:
            if origins[i] is not None:
                contributing_indices_mask = z_pattern_indices[0] == i
                self.z_pattern_data[i.item()] = ZPatternSegment(
                    contributing_indices=z_pattern_indices[1, contributing_indices_mask].tolist(),
                    contributions=z_pattern_values[contributing_indices_mask].tolist(),
                    max_contribution=z_pattern_values[contributing_indices_mask].max().item(),
                )
    
    def has_z_pattern_data(self):
        return self.z_pattern_data is not None

    @staticmethod
    def construct(
        text: str,
        activations: torch.Tensor,
        origins: list[dict[str, Any]],
        max_activation: float,
    ) -> "TokenizedSample":
        """Construct a TokenizedSample from text, activations, and origins.

        Args:
            text: The full text string
            activations: Tensor of activation values
            origins: List of origin dictionaries with position information
            max_activation: Maximum activation value

        Returns:
            A TokenizedSample instance
        """
        positions: set[int] = set()
        for origin in origins:
            if origin and origin["key"] == "text":
                assert "range" in origin, f"Origin {origin} does not have a range"
                positions.add(origin["range"][0])
                positions.add(origin["range"][1])

        sorted_positions = sorted(positions)

        segments = []
        for i in range(len(sorted_positions) - 1):
            start, end = sorted_positions[i], sorted_positions[i + 1]
            try:
                segment_activation = max(
                    act
                    for origin, act in zip(origins, activations)
                    if origin and origin["key"] == "text" and origin["range"][0] >= start and origin["range"][1] <= end
                )
            except Exception as e:
                logger.error(f"Error processing segment:\nstart={start}, end={end}, segment={text[start:end]}\n\n. Error: {e}")
                continue
            segments.append(Segment(text[start:end], segment_activation.item()))

        return TokenizedSample(segments, max_activation)

