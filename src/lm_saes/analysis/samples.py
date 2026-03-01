"""Sample processing utilities for feature activation visualization and analysis.

This module provides data structures and utilities for working with tokenized samples
and their activation patterns, used across analysis and visualization tools.
"""

import bisect
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lm_saes.utils.logging import get_logger

logger = get_logger("analysis.samples")


def process_token(token: str) -> str:
    """Process a token string by replacing special characters.

    Args:
        token: The token string to process

    Returns:
        Processed token string with special characters replaced
    """
    return token.replace("\n", "⏎").replace("\t", "→").replace("\r", "↵")


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

    suffix_text: str | None = None
    """Optional suffix text with no activation (e.g. side_to_move, wdl, fen for chess). Appended when displaying."""

    @property
    def sample_max_activation(self) -> float:
        """The maximum activation value of the sample."""
        return max([seg.activation for seg in self.segments])

    def to_max_activation_surrounding(self, visible_range: int) -> "TokenizedSample":
        """Convert to a TokenizedSample with only the max activation surrounding tokens."""
        assert self.z_pattern_data is None, "Z pattern data is not supported for max activation surrounding tokens"
        max_activation_index = np.argmax([seg.activation for seg in self.segments])
        range = (max_activation_index - visible_range, max_activation_index + visible_range)
        return TokenizedSample(
            segments=self.segments[range[0] : range[1]],
            max_activation=self.max_activation,
            suffix_text=None,
        )

    def display_highlighted(self, threshold: float = 0.7) -> str:
        """Get the text with activating segments highlighted with << >> delimiters.

        Args:
            threshold: Threshold relative to max activation for highlighting

        Returns:
            Text with activating segments highlighted
        """
        highlighted_text = "".join([seg.display(threshold * self.max_activation) for seg in self.segments])
        if self.suffix_text is not None:
            highlighted_text += self.suffix_text
        return highlighted_text

    def display_plain(self) -> str:
        """Get the text with all segments displayed."""
        plain = "".join([seg.text for seg in self.segments])
        if self.suffix_text is not None:
            plain += self.suffix_text
        return plain

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
                        k_prev_tokens = [
                            f"({process_token(''.join([self.segments[idx].text for idx in range(max(0, j - 3), j)]))}) {process_token(self.segments[j].text)}"
                            for j, contribution in zip(
                                z_pattern_segment.contributing_indices, z_pattern_segment.contributions
                            )
                            if contribution > threshold * z_pattern_segment.max_contribution
                        ]
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
        self, z_pattern_indices: torch.Tensor, z_pattern_values: torch.Tensor, origins: list[dict[str, Any]]
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
            max_activation: Global maximum activation value

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

        # Pre-process origins for efficient range queries
        origin_activations = [
            (origin["range"][0], origin["range"][1], act.item())
            for origin, act in zip(origins, activations)
            if origin and origin["key"] == "text" and "range" in origin
        ]
        origin_activations.sort(key=lambda x: x[0])
        origin_starts = [x[0] for x in origin_activations]

        segments = []
        for i in range(len(sorted_positions) - 1):
            start, end = sorted_positions[i], sorted_positions[i + 1]
            # Use binary search to find the first origin with start >= segment_start
            first_idx = bisect.bisect_left(origin_starts, start)
            # Iterate only through candidates, breaking early when start exceeds segment_end
            max_act = 0.0
            for orig_start, orig_end, act in origin_activations[first_idx:]:
                if orig_start > end:
                    break
                if orig_end <= end:
                    max_act = max(max_act, act)
            segments.append(Segment(text[start:end], max_act))

        return TokenizedSample(segments, max_activation)


def _fen_board_to_64chars(fen: str, side_to_move: str = "?") -> str:
    """Expand FEN board part to 64-character string (one char per square, '.' for empty).

    FEN board order is rank8, rank7, ..., rank1 (row 0 = a8-h8, row 7 = a1-h1).
    Reverse row order only when white to move (columns a-h unchanged): then index 0-7 = a1-h1.
    When black to move or unknown, keep FEN order: index 0-7 = a8-h8.
    """
    parts = fen.split()
    board_fen = parts[0]
    # Build 8 rows of 8 chars each (FEN order: rank8, rank7, ..., rank1)
    rows: list[str] = []
    current_row = ""
    for char in board_fen:
        if char == "/":
            rows.append(current_row)
            current_row = ""
        elif char.isdigit():
            current_row += "." * int(char)
        else:
            current_row += char
    rows.append(current_row)
    if len(rows) != 8:
        raise ValueError(f"FEN board must have 8 rows, got {len(rows)}")
    # White: reverse rows so index 0-7 = a1-h1. Black: keep FEN order so index 0-7 = a8-h8.
    if side_to_move.lower() == "w":
        rows = rows[::-1]
    board_str = "".join(rows)
    return board_str


def sae_pos_to_display_pos(sae_pos: int, side_to_move: str) -> int:
    """Map SAE context position (0-63, FEN order a8-h8..a1-h1) to display segment index.

    Display order matches build_chess_tokenized_sample / _fen_board_to_64chars:
    - White: a1-h1, ..., a8-h8 so display_pos = (7 - row)*8 + col.
    - Black: a8-h8, ..., a1-h1 (FEN order), so display_pos = sae_pos.
    """
    if side_to_move.lower() == "w":
        row, col = sae_pos // 8, sae_pos % 8
        return (7 - row) * 8 + col
    return sae_pos


def longfen_index_to_square(i: int, side_to_move: str = "b") -> str:
    """Map longfen index 0-63 to algebraic square.

    When side_to_move is 'w', segment order is a1-h1, ..., a8-h8 (rank = i//8 + 1).
    When side_to_move is 'b' or unknown, segment order is a8-h8, ..., a1-h1 (rank = 8 - i//8).
    """
    col = i % 8
    file_ = chr(ord("a") + col)
    row = i // 8
    if side_to_move.lower() == "w":
        rank = row + 1  # row 0 = rank 1 (a1-h1), ..., row 7 = rank 8 (a8-h8)
    else:
        rank = 8 - row  # row 0 = rank 8 (a8-h8), ..., row 7 = rank 1 (a1-h1)
    return f"{file_}{rank}"


def build_chess_tokenized_sample(
    fen: str,
    feature_acts: list[float] | np.ndarray | torch.Tensor,
    max_activation: float,
    side_to_move: str = "?",
    wdl: tuple[float, float, float] | None = None,
    top_moves: list[tuple[str, float]] | None = None,
    z_pattern_data: dict[int, ZPatternSegment] | None = None,
) -> TokenizedSample:
    """Build a TokenizedSample for chess: 64 segments (square+piece with activations) and suffix_text (metadata).

    Args:
        fen: FEN string (full or board-only).
        feature_acts: Activation per position; length must be 64 (one per square).
        max_activation: Global max activation for the feature.
        side_to_move: 'w' or 'b' (who is to move); use '?' if unknown.
        wdl: (win, draw, loss) for the side to move; use None if unknown.
        top_moves: Optional list of (uci, prob) for model's top predicted moves (e.g. top 3).
        z_pattern_data: Optional z-pattern data keyed by segment index (0-63).

    Returns:
        TokenizedSample with 64 board segments and suffix_text for side_to_move, wdl, top_moves, fen (no activation).
    """
    if isinstance(feature_acts, torch.Tensor):
        feature_acts = feature_acts.flatten().tolist()
    else:
        feature_acts = list(feature_acts)
    board_str = _fen_board_to_64chars(fen, side_to_move)
    if len(board_str) != 64:
        raise ValueError(f"FEN board must expand to 64 chars, got {len(board_str)}")
    # Pad or trim activations to 64
    if len(feature_acts) < 64:
        feature_acts = feature_acts + [0.0] * (64 - len(feature_acts))
    else:
        feature_acts = feature_acts[:64]

    segments: list[Segment] = []
    for i in range(64):
        square = longfen_index_to_square(i, side_to_move)
        piece = board_str[i] if i < len(board_str) else "."
        text = f"{square}{piece}"
        act = float(feature_acts[i]) if i < len(feature_acts) else 0.0
        segments.append(Segment(text=text, activation=act))

    # Metadata with no activation: pass as optional suffix_text instead of Segment.
    # Use newlines so autointerp can read side_to_move, wdl, top_moves, fen as separate lines.
    top_moves_str = (
        ",".join(f"{uci} {prob:.3f}" for uci, prob in top_moves)
        if top_moves
        else "?"
    )
    suffix_parts: list[str] = [
        f"side_to_move:{side_to_move}",
        f"wdl:{wdl[0]:.2f},{wdl[1]:.2f},{wdl[2]:.2f}" if wdl is not None else "wdl:?,?,?",
        f"top_moves:{top_moves_str}",
        f"fen:{fen.strip()}",
    ]
    suffix_text = "\n" + "\n".join(suffix_parts)

    return TokenizedSample(
        segments=segments,
        max_activation=max_activation,
        z_pattern_data=z_pattern_data,
        suffix_text=suffix_text,
    )