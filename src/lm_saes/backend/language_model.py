import re
import warnings
from abc import ABC, abstractmethod
from itertools import accumulate
from typing import Any, Optional, cast

import torch
from transformer_lens import HookedTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BatchFeature,
    Qwen2_5_VLForConditionalGeneration,
)

from lm_saes.config import LanguageModelConfig, LLaDAConfig
from lm_saes.utils.misc import pad_and_truncate_tokens
from lm_saes.utils.timer import timer


def get_input_with_manually_prepended_bos(tokenizer, input):
    """
    Manually prepends the bos token to the input.

    Args:
        tokenizer (AutoTokenizer): The tokenizer to use for prepending the bos token.
        input (Union[str, List[str]]): The input to prepend the bos token to.

    Returns:
        Union[str, List[str]]: The input with the bos token manually prepended.
    """
    if isinstance(input, str):
        input = tokenizer.bos_token + input
    else:
        input = [tokenizer.bos_token + string for string in input]
    return input


def to_tokens(tokenizer, text, max_length, device="cpu"):
    tokenizer_prepends_bos = len(tokenizer.encode("")) > 0
    text = text if tokenizer_prepends_bos else get_input_with_manually_prepended_bos(tokenizer, text)
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    return tokens.to(device)


def set_tokens(tokenizer):
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    return tokenizer


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
            token_positions.append({"key": "text", "range": (pos, pos + len(token))})
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


def _get_layer_indices_from_hook_points(hook_points: list[str]) -> list[int]:
    residual_pattern = r"^blocks\.(\d+)\.hook_resid_post$"
    matches = [re.match(residual_pattern, hook_point) for hook_point in hook_points]
    assert all(match is not None for match in matches), "hook_points must be residual stream hook points"
    layer_indices = [int(cast(re.Match[str], match).group(1)) for match in matches]
    return layer_indices


class LanguageModel(ABC):
    @abstractmethod
    def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
        """Trace how raw data is eventually aligned with tokens.

        Args:
            raw (dict[str, Any]): The raw data to trace.

        Returns:
            list[list[Any]]: The origins of the tokens in the raw data. Shape: (batch_size, n_tokens)
        """
        pass

    @abstractmethod
    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]] | None]:
        """Convert raw data to activations.

        Args:
            raw (dict[str, Any]): The raw data to convert to activations.
            hook_points (list[str]): The hook points to use for activations.

        Returns:
            dict[str, torch.Tensor]: The activations. Shape: (batch_size, n_tokens, n_activations)
        """
        pass

    @abstractmethod
    def preprocess_raw_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Preprocess the raw data.

        Args:
            raw (dict[str, Any]): The raw data to preprocess.

        Returns:
            dict[str, Any]: The preprocessed raw data.
        """
        pass

    @property
    @abstractmethod
    def eos_token_id(self) -> int | None:
        """The ID of the end-of-sequence token."""
        pass

    @property
    @abstractmethod
    def bos_token_id(self) -> int | None:
        """The ID of the beginning-of-sequence token."""
        pass

    @property
    @abstractmethod
    def pad_token_id(self) -> int | None:
        """The ID of the padding token."""
        pass


class TransformerLensLanguageModel(LanguageModel):
    def __init__(self, cfg: LanguageModelConfig):
        self.cfg = cfg
        if cfg.device == "cuda":
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        elif cfg.device == "npu":
            self.device = torch.device(f"npu:{torch.npu.current_device()}")  # type: ignore[reportAttributeAccessIssue]
        else:
            self.device = torch.device(cfg.device)

        hf_model = (
            AutoModelForCausalLM.from_pretrained(
                (cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path),
                cache_dir=cfg.cache_dir,
                local_files_only=cfg.local_files_only,
                torch_dtype=cfg.dtype,
                trust_remote_code=True,
            )
            if cfg.load_ckpt
            else None
        )
        hf_tokenizer = AutoTokenizer.from_pretrained(
            (cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path),
            trust_remote_code=True,
            use_fast=True,
            add_bos_token=True,
            local_files_only=cfg.local_files_only,
        )
        self.tokenizer = set_tokens(hf_tokenizer)
        self.model = (
            HookedTransformer.from_pretrained_no_processing(
                cfg.model_name,
                use_flash_attn=cfg.use_flash_attn,
                device=self.device,
                cache_dir=cfg.cache_dir,
                hf_model=hf_model,
                hf_config=hf_model.config,
                tokenizer=hf_tokenizer,
                dtype=cfg.dtype,  # type: ignore ; issue with transformer_lens
            )
            if hf_model
            else None
        )

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        return self.tokenizer.bos_token_id

    @property
    def pad_token_id(self) -> int | None:
        return self.tokenizer.pad_token_id

    def preprocess_raw_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw

    def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
        if any(key in ["images", "videos"] for key in raw):
            warnings.warn(
                "Tracing with modalities other than text is not implemented for TransformerLensLanguageModel. Only text fields will be used."
            )
        tokens = to_tokens(self.tokenizer, raw["text"], max_length=self.cfg.max_length, device=self.cfg.device)
        if n_context is not None:
            assert self.pad_token_id is not None, (
                "Pad token ID must be set for TransformerLensLanguageModel when n_context is provided"
            )
            tokens = pad_and_truncate_tokens(tokens, n_context, pad_token_id=self.pad_token_id)
        batch_str_tokens = [self.tokenizer.batch_decode(token, clean_up_tokenization_spaces=False) for token in tokens]
        return [
            _match_str_tokens_to_input(text, str_tokens) for (text, str_tokens) in zip(raw["text"], batch_str_tokens)
        ]

    @timer.time("to_activations")
    @torch.no_grad()
    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]] | None]:
        assert self.model is not None
        if any(key in ["images", "videos"] for key in raw):
            warnings.warn(
                "Activations with modalities other than text is not implemented for TransformerLensLanguageModel. Only text fields will be used."
            )
        with timer.time("to_tokens"):
            tokens = self.model.to_tokens(raw["text"], prepend_bos=self.cfg.prepend_bos)
        if n_context is not None:
            assert self.pad_token_id is not None, (
                "Pad token ID must be set for TransformerLensLanguageModel when n_context is provided"
            )
            tokens = pad_and_truncate_tokens(tokens, n_context, pad_token_id=self.pad_token_id)
        with timer.time("run_with_cache_until"):
            _, activations = self.model.run_with_cache_until(tokens, names_filter=hook_points)
        for key, value in activations.items():
            if key.endswith("hook_z"):
                activations[key] = activations[key].reshape(*activations[key].shape[:-2], -1)
        return {hook_point: activations[hook_point] for hook_point in hook_points} | {"tokens": tokens}, None


class HuggingFaceLanguageModel(LanguageModel):
    def __init__(self, cfg: LanguageModelConfig):
        self.cfg = cfg
        self.device = (
            torch.device(f"cuda:{torch.cuda.current_device()}") if cfg.device == "cuda" else torch.device(cfg.device)
        )

    def preprocess_raw_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw


class LLaDALanguageModel(TransformerLensLanguageModel):
    cfg: LLaDAConfig  # Explicitly specify the type to avoid linter errors

    @property
    def mdm_mask_token_id(self) -> int:
        return self.cfg.mdm_mask_token_id

    @torch.no_grad()
    def _get_masked_tokens(self, tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply random masking to non-pad tokens based on mask_ratio. The random seed should be determined by the raw data.
        Args:
            tokens (torch.Tensor): The tokens to mask.
            pad_mask (torch.Tensor): The mask of the padding tokens.

        Returns:
            torch.Tensor: The tokens after masking. Pad tokens are included.
        """
        if self.cfg.mask_ratio <= 0:
            return tokens

        # Calculate random seeds based on non-pad tokens
        random_seeds = (tokens * ~pad_mask).sum(dim=1).tolist()
        masked_tokens = tokens.clone()

        # For each sequence in the batch
        for i, seed in enumerate(random_seeds):
            # Create random generator with deterministic seed
            random_generator = torch.Generator(device=tokens.device)
            random_generator.manual_seed(seed)

            # Generate random values for all positions
            random_values = torch.rand(tokens[i].shape, device=tokens[i].device, generator=random_generator)

            # Count non-pad tokens
            non_pad_mask = ~pad_mask[i]
            non_pad_count = non_pad_mask.sum()

            # Calculate number of tokens to mask
            num_to_mask = int(non_pad_count.item() * self.cfg.mask_ratio)
            if num_to_mask == 0:
                continue

            # Set random values for pad positions to infinity so they won't be selected
            random_values = torch.where(non_pad_mask, random_values, torch.full_like(random_values, float("inf")))

            # Get indices of positions to mask (lowest random values)
            mask_indices = torch.topk(random_values, k=num_to_mask, largest=False).indices

            # Apply masking
            masked_tokens[i][mask_indices] = self.mdm_mask_token_id

        return masked_tokens

    @torch.no_grad()
    def preprocess_raw_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        assert self.model is not None
        tokens = self.model.to_tokens(raw["text"], prepend_bos=self.cfg.prepend_bos)
        pad_mask = tokens == self.pad_token_id
        masked_tokens = self._get_masked_tokens(tokens, pad_mask)
        original_text = raw["text"]
        raw["text"] = self.tokenizer.batch_decode(masked_tokens)
        pad_token = self.tokenizer.pad_token
        raw["text"] = [text.replace(pad_token, "") for text in raw["text"]]
        raw_meta = raw.get("meta", [])
        raw["meta"] = [
            (raw_meta[i] if i < len(raw_meta) else {})
            | {"original_text": original_text[i]}
            | {"mask_ratio": self.cfg.mask_ratio}
            for i in range(len(raw["text"]))
        ]
        return raw

    @torch.no_grad()
    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]] | None]:
        assert self.model is not None
        tokens = self.model.to_tokens(raw["text"], prepend_bos=self.cfg.prepend_bos)
        if n_context is not None:
            assert self.pad_token_id is not None, (
                "Pad token ID must be set for LLaDALanguageModel when n_context is provided"
            )
            tokens = pad_and_truncate_tokens(tokens, n_context, pad_token_id=self.pad_token_id)
        pad_mask = tokens == self.pad_token_id
        predicted_tokens = None
        if self.cfg.calculate_logits:
            logits, activations = self.model.run_with_cache(tokens, names_filter=hook_points)
            assert isinstance(logits, torch.Tensor)
            predicted_token_ids = logits.argmax(dim=-1)
            predicted_tokens = torch.where(pad_mask, -1, predicted_token_ids)
        else:
            _, activations = self.model.run_with_cache_until(tokens, names_filter=hook_points)
        meta_info = [{"predicted_tokens": predicted_tokens[i]} for i in range(len(predicted_tokens))] if predicted_tokens is not None else None
        return {hook_point: activations[hook_point] for hook_point in hook_points} | {"tokens": tokens} , meta_info


class QwenVLLanguageModel(HuggingFaceLanguageModel):
    def __init__(self, cfg: LanguageModelConfig):
        super().__init__(cfg)
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
            max_pixels=1800 * 28 * 28,
        )
        self.tokenizer = self.processor.tokenizer
        self.model.eval()

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        return self.tokenizer.bos_token_id  # should be None

    @property
    def pad_token_id(self) -> int | None:
        return self.tokenizer.pad_token_id

    def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
        assert self.tokenizer is not None, "tokenizer must be initialized"
        assert self.processor is not None, "processor must be initialized"
        inputs, _ = self.process_raw_data(raw)
        input_ids = inputs["input_ids"]
        if n_context is not None:
            assert self.pad_token_id is not None, (
                "Pad token ID must be set for QwenVLLanguageModel when n_context is provided"
            )
            input_ids = pad_and_truncate_tokens(input_ids, n_context, pad_token_id=self.pad_token_id)
        batch_str_tokens: list[list[str]] = [
            self.tokenizer.batch_decode(input_id, clean_up_tokenization_spaces=False) for input_id in input_ids
        ]

        def split_number(n: int, m: float) -> list[int]:
            # split n into m parts, the parts are as even as possible
            assert m.is_integer()
            quotient = n // int(m)
            remainder = n % int(m)
            return [quotient + 1 if i < remainder else quotient for i in range(int(m))]

        batch_token_positions: list[list[Any]] = [
            _match_str_tokens_to_input(text, str_tokens) for (text, str_tokens) in zip(raw["text"], batch_str_tokens)
        ]

        if "images" in raw and raw["images"] is not None:
            assert "image_grid_thw" in inputs
            assert "pixel_values" in inputs
            resized_shape_list = (inputs["image_grid_thw"][:, 1:] * 14).tolist()
            for str_tokens, images in zip(batch_str_tokens, raw["images"]):
                # str_tokens: list[str], tokens for each text in the batch
                # images: list[torch.Tensor], images for each text in the batch
                # resized_shape: [total_image_number, 2]
                # token_positions: list[Any], positions of the tokens in the input text
                start_id_list = [id for id, str_token in enumerate(str_tokens) if str_token == "<|vision_start|>"]
                end_id_list = [id for id, str_token in enumerate(str_tokens) if str_token == "<|vision_end|>"]
                assert len(start_id_list) == len(end_id_list)
                images_num = len(start_id_list)  # number of images in this text
                resized_shapes = resized_shape_list[:images_num]  # get the resized shapes for images in this text
                resized_shape_list = resized_shape_list[images_num:]

                for i, (start_id, end_id, image, resized_shape) in enumerate(
                    zip(start_id_list, end_id_list, images, resized_shapes)
                ):
                    resized_height, resized_width = int(resized_shape[0]), int(resized_shape[1])
                    original_height, original_width = image.shape[-2], image.shape[-1]
                    image_token_num = end_id - start_id - 1
                    assert image_token_num == resized_height * resized_width / 14 / 14 / 4

                    split_height = split_number(original_height, resized_height / 28)
                    split_width = split_number(original_width, resized_width / 28)

                    prefix_sum_height = [0] + list(accumulate(split_height))
                    prefix_sum_width = [0] + list(accumulate(split_width))

                    prefix_sum_height = [i / original_height for i in prefix_sum_height]
                    prefix_sum_width = [i / original_width for i in prefix_sum_width]
                    grid_coords = [
                        (
                            id // (resized_width // 28),
                            id % (resized_width // 28),
                        )
                        for id in range(image_token_num)
                    ]
                    original_coords = [
                        (
                            prefix_sum_width[grid_coords[id][1]],
                            prefix_sum_height[grid_coords[id][0]],
                            prefix_sum_width[grid_coords[id][1] + 1],
                            prefix_sum_height[grid_coords[id][0] + 1],
                        )
                        for id in range(image_token_num)
                    ]

                    batch_token_positions[i][start_id + 1 : end_id] = original_coords
                    for j in range(len(original_coords)):
                        batch_token_positions[i][start_id + 1 + j] = {
                            "key": "image",
                            "rect": original_coords[j],
                            "image_index": i,
                        }
        return batch_token_positions

    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]] | None]:
        layer_indices = _get_layer_indices_from_hook_points(hook_points)
        inputs = self.process_raw_data(raw)[0].to(self.device)
        if n_context is not None:
            assert self.pad_token_id is not None, (
                "Pad token ID must be set for QwenVLLanguageModel when n_context is provided"
            )
            inputs["input_ids"] = pad_and_truncate_tokens(
                inputs["input_ids"], n_context, pad_token_id=self.pad_token_id
            )
        outputs = self.model(**inputs, output_hidden_states=True)
        activations = {
            hook_points[i]: outputs.hidden_states[layer_index + 1] for i, layer_index in enumerate(layer_indices)
        }
        activations["tokens"] = inputs["input_ids"]
        return activations, None

    def process_raw_data(
        self, raw: dict[str, Any], padding: str | bool = "max_length"
    ) -> tuple[BatchFeature, dict[str, Any]]:
        # raw is a dict with keys "text", "images", "videos", etc.
        IMAGE_PAD_TOKEN: str = raw.get("image_pad_token", "<image>")
        inputs = cast(BatchFeature, {})
        processed_raw = {}

        # use process_vision_info to resize images
        assert self.processor is not None, "processor must be initialized"
        if "images" in raw:
            processed_raw["images"] = raw["images"]

        processed_raw["text"] = [
            text.replace(IMAGE_PAD_TOKEN, f"<|vision_start|>{self.processor.image_token}<|vision_end|>")
            for text in raw["text"]
        ]
        inputs: BatchFeature = self.processor(
            text=processed_raw["text"],
            images=raw.get("images", None),
            return_tensors="pt",
            max_length=self.cfg.max_length,
            padding=padding,
            truncation=True,
        )

        return inputs, processed_raw


class QwenLanguageModel(HuggingFaceLanguageModel):
    def __init__(self, cfg: LanguageModelConfig):
        super().__init__(cfg)
        # hidden_size is 3584 for Qwen2.5-7B
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
            torch_dtype=cfg.dtype,
        ).to(self.device)  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
            padding_side="left",
        )
        self.model.eval()

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        return self.tokenizer.bos_token_id  # should be None

    @property
    def pad_token_id(self) -> int | None:
        return self.tokenizer.pad_token_id

    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]] | None]:
        layer_indices = _get_layer_indices_from_hook_points(hook_points)
        inputs = self.tokenizer(
            raw["text"],
            return_tensors="pt",
            padding="max_length",
            max_length=self.cfg.max_length,
            truncation=True,
        ).to(self.device)
        if n_context is not None:
            assert self.pad_token_id is not None, (
                "Pad token ID must be set for QwenLanguageModel when n_context is provided"
            )
            inputs["input_ids"] = pad_and_truncate_tokens(
                inputs["input_ids"], n_context, pad_token_id=self.pad_token_id
            )
        outputs = self.model(**inputs, output_hidden_states=True)
        activations = {
            hook_points[i]: outputs.hidden_states[layer_index + 1] for i, layer_index in enumerate(layer_indices)
        }
        activations["tokens"] = inputs["input_ids"]
        return activations, None

    def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
        inputs = self.tokenizer(
            raw["text"], return_tensors="pt", padding="max_length", max_length=self.cfg.max_length, truncation=True
        )
        input_ids = inputs["input_ids"]
        if n_context is not None:
            assert self.pad_token_id is not None, (
                "Pad token ID must be set for QwenLanguageModel when n_context is provided"
            )
            input_ids = pad_and_truncate_tokens(input_ids, n_context, pad_token_id=self.pad_token_id)
        batch_str_tokens = [
            self.tokenizer.batch_decode(input_id, clean_up_tokenization_spaces=False) for input_id in input_ids
        ]
        return [
            _match_str_tokens_to_input(text, str_tokens) for (text, str_tokens) in zip(raw["text"], batch_str_tokens)
        ]