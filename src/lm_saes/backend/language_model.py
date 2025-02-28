import re
from abc import ABC, abstractmethod
from itertools import accumulate
from typing import Any, Optional, cast

import torch
from transformers import AutoProcessor, BatchFeature, Qwen2_5_VLForConditionalGeneration

from lm_saes.config import LanguageModelConfig


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


class HuggingFaceLanguageModel(LanguageModel):
    def __init__(self, cfg: LanguageModelConfig):
        self.device = (
            torch.device(f"cuda:{torch.cuda.current_device()}") if cfg.device == "cuda" else torch.device(cfg.device)
        )
        self.tokenizer = None
        self.processor = None
        if "Qwen2.5-VL" in cfg.model_name:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                cfg.model_name,
                cache_dir=cfg.cache_dir,
                local_files_only=cfg.local_files_only,
                torch_dtype=cfg.dtype,
                attn_implementation="flash_attention_2" if cfg.use_flash_attn else None,
            ).to(self.device)  # type: ignore
            self.processor = AutoProcessor.from_pretrained(
                cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path,
                cache_dir=cfg.cache_dir,
                local_files_only=cfg.local_files_only,
                padding_side="left",
            )
            self.tokenizer = self.processor.tokenizer

        self.model.eval()

    def to_tokens(self, raw: dict[str, Any]) -> torch.Tensor:  # type: ignore
        pass

    def to_activations_from_tokens(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:  # type: ignore
        pass

    def trace(self, raw: dict[str, Any]) -> list[list[Any]]:
        assert self.tokenizer is not None, "tokenizer must be initialized"
        assert self.processor is not None, "processor must be initialized"
        inputs, processed_raw = self.process_raw_data(raw)
        input_ids = inputs["input_ids"]
        # print(inputs["image_grid_thw"].shape)
        resized_shape = inputs["image_grid_thw"][:, 1:] * 14  # torch.Size([batch_size, 2])
        batch_str_tokens = [
            self.tokenizer.batch_decode(input_id, clean_up_tokenization_spaces=False) for input_id in input_ids
        ]

        # print("input_ids: ", input_ids)
        # print("batch_str_tokens: ", batch_str_tokens)
        # print("resized_shape: ", resized_shape)
        def _match_str_tokens_to_input(text: str, str_tokens: list[str]) -> list[Optional[tuple[int, int]]]:
            """Match the tokens to the input text, returning a list of tuples of the form (start_idx, end_idx) for each token."""
            # Initialize list to store token positions
            token_positions = []

            # Keep track of current position in text
            curr_pos = 0

            # For each token, try to find its position in the input text
            for token in str_tokens:
                # Search for token in remaining text
                pos = text.find(token, curr_pos)

                if pos != -1:
                    # Found a match, store position and update curr_pos
                    token_positions.append((pos, pos + len(token)))
                    curr_pos = pos + len(token)
                else:
                    # No match found. This is only allowed if the token is a special token
                    # that doesn't appear in the input text, or if the token is a subword token
                    # which cannot be decoded separately.
                    # TODO: Deal with subword tokens properly
                    if not ((token.startswith("<") and token.endswith(">")) or "�" in token):
                        raise ValueError(f"Token {token} not found in input text")
                    token_positions.append(None)

            return token_positions

        def split_number(n: int, m: float) -> list[int]:
            # split n into m parts, the parts are as even as possible
            assert m.is_integer()
            quotient = n // int(m)
            remainder = n % int(m)
            return [quotient + 1 if i < remainder else quotient for i in range(int(m))]

        batch_token_positions: list[list[Any]] = [
            _match_str_tokens_to_input(text, str_tokens) for (text, str_tokens) in zip(raw["text"], batch_str_tokens)
        ]

        if "images" in raw:
            assert "pixel_values" in inputs
            if "Qwen2.5-VL" in self.model.name_or_path:
                start_id_list = []
                end_id_list = []
                for i, (str_tokens, images, resized_shape) in enumerate(
                    zip(batch_str_tokens, raw["images"], resized_shape)
                ):
                    # str_tokens: list[str], tokens for each text in the batch
                    # images: tensor, [1, 3, height, width]
                    # resized_shape: [batch_size, 2]
                    # token_positions: list[Any], positions of the tokens in the input text

                    # find the start and end of the image tokens of this text
                    start_id_list = [id for id, str_token in enumerate(str_tokens) if str_token == "<|vision_start|>"]
                    end_id_list = [id for id, str_token in enumerate(str_tokens) if str_token == "<|vision_end|>"]

                    assert len(start_id_list) == len(end_id_list)
                    assert len(start_id_list) == 1, "only one image is supported"
                    assert images.shape[0] == 1, "only one image is supported"

                    start_id = start_id_list[0]
                    end_id = end_id_list[0]
                    resized_height, resized_width = int(resized_shape[0]), int(resized_shape[1])
                    original_height, original_width = images.shape[2], images.shape[3]
                    image_token_num = end_id - start_id - 1
                    assert image_token_num == resized_shape[0] * resized_shape[1] / 14 / 14 / 4

                    split_height = split_number(original_height, resized_height / 28)
                    split_width = split_number(original_width, resized_width / 28)
                    prefix_sum_height = [0] + list(accumulate(split_height))
                    prefix_sum_width = [0] + list(accumulate(split_width))
                    # print("original_height: ", original_height)
                    # print("original_width: ", original_width)
                    # print("resized_height: ", resized_height)
                    # print("resized_width: ", resized_width)
                    # print("split_height: ", split_height)
                    # print("split_width: ", split_width)
                    # print("prefix_sum_height: ", prefix_sum_height)
                    # print("prefix_sum_width: ", prefix_sum_width)
                    grid_coords = [
                        (
                            id // (resized_width // 28),
                            id % (resized_width // 28),
                        )
                        for id in range(image_token_num)
                    ]
                    # print("grid_coords: ", grid_coords)
                    original_coords = [
                        (
                            prefix_sum_height[grid_coords[id][0]],
                            prefix_sum_width[grid_coords[id][1]],
                            prefix_sum_height[grid_coords[id][0] + 1],
                            prefix_sum_width[grid_coords[id][1] + 1],
                        )
                        for id in range(image_token_num)
                    ]
                    batch_token_positions[i][start_id + 1 : end_id] = original_coords

        return batch_token_positions

    def to_activations(self, raw: dict[str, Any], hook_points: list[str]) -> dict[str, torch.Tensor]:
        residual_pattern = r"^blocks\.(\d+)\.hook_resid_post$"
        matches = [re.match(residual_pattern, hook_point) for hook_point in hook_points]
        assert all(match is not None for match in matches), "hook_points must be residual stream hook points"
        layer_indices = [int(cast(re.Match[str], match).group(1)) for match in matches]
        inputs = self.process_raw_data(raw)[0].to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        activations = {
            hook_points[i]: outputs.hidden_states[layer_index + 1] for i, layer_index in enumerate(layer_indices)
        }
        return activations

    def process_raw_data(self, raw: dict[str, Any]) -> tuple[BatchFeature, dict[str, Any]]:
        # raw is a dict with keys "text", "images", "videos", etc.
        IMAGE_PAD_TOKEN: str = raw.get("image_pad_token", "<image>")
        inputs = cast(BatchFeature, {})
        processed_raw = {}
        if "Qwen2.5-VL" in self.model.name_or_path:
            # use process_vision_info to resize images
            assert self.processor is not None, "processor must be initialized"
            if "images" in raw:
                images = raw["images"] if isinstance(raw["images"], list) else list(raw["images"])
            else:
                images = None
            processed_raw["images"] = images
            processed_raw["text"] = [
                text.replace(IMAGE_PAD_TOKEN, f"<|vision_start|>{self.processor.image_token}<|vision_end|>")
                for text in raw["text"]
            ]
            inputs: BatchFeature = self.processor(
                text=processed_raw["text"],
                images=images,
                return_tensors="pt",
                padding=True,
            )

        return inputs, processed_raw
