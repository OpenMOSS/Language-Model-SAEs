from abc import ABC, abstractmethod
from typing import Any

import torch


class LanguageModel(ABC):
    @abstractmethod
    def to_tokens(self, raw: dict[str, Any]) -> torch.Tensor:
        """Convert raw data to tokens.

        Args:
            raw (dict[str, Any]): The raw data to convert to tokens. May contain keys like "text", "images", "videos", etc.

        Returns:
            torch.Tensor: The tokens. Shape: (batch_size, n_tokens)
        """
        pass

    @abstractmethod
    def trace(self, raw: dict[str, Any]) -> list[list[Any]]:
        """Trace how raw data is eventually aligned with tokens.

        Args:
            raw (dict[str, Any]): The raw data to trace.

        Returns:
            list[list[Any]]: The origins of the tokens in the raw data. Shape: (batch_size, n_tokens)
        """
        pass

    @abstractmethod
    def to_activations(self, raw: dict[str, Any], hook_points: list[str]) -> dict[str, torch.Tensor]:
        """Convert raw data to activations.

        Args:
            raw (dict[str, Any]): The raw data to convert to activations.
            hook_points (list[str]): The hook points to use for activations.

        Returns:
            dict[str, torch.Tensor]: The activations. Shape: (batch_size, n_tokens, n_activations)
        """
        pass

    @abstractmethod
    def to_activations_from_tokens(self, tokens: torch.Tensor, hook_points: list[str]) -> dict[str, torch.Tensor]:
        """Convert tokens to activations.

        Args:
            tokens (torch.Tensor): The tokens. Shape: (batch_size, n_tokens)
        """
        pass
