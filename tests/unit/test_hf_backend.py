from typing import Any

import pytest
import torch

from lm_saes.backend.language_model import Qwen_LanguageModel, Qwen_VL_LanguageModel
from lm_saes.config import LanguageModelConfig

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)


class Test_Qwen_VL_LanguageModel:
    @pytest.fixture
    def language_model_config(self) -> LanguageModelConfig:
        return LanguageModelConfig(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            device="cuda",
            dtype=torch.bfloat16,
            use_flash_attn=True,
            d_model=3584,
            local_files_only=False,
        )

    @pytest.fixture
    def multi_image_input(self) -> dict[str, Any]:
        return {
            "text": [f"<image><image>{i}<image>!" for i in range(10)],
            "images": [torch.randint(0, 256, (3, 3, 28 + 28 * i, 28)) for i in range(10)],
        }

    @pytest.fixture
    def single_image_input(self) -> dict[str, Any]:
        return {
            "text": ["<image>test!" for _ in range(10)],
            "images": [torch.randint(0, 256, (1, 3, 28 + 28 * i, 28)) for i in range(10)],
        }

    def test_to_activations(self, language_model_config: LanguageModelConfig, multi_image_input: dict[str, Any]):
        hf_language_model = Qwen_VL_LanguageModel(language_model_config)
        hf_language_model.model.eval()
        hook_points = [f"blocks.{i}.hook_resid_post" for i in range(28)]
        activations = hf_language_model.to_activations(multi_image_input, hook_points)
        assert len(activations) == 28

        model = hf_language_model.model
        processor = hf_language_model.processor
        for i in range(len(multi_image_input["text"])):
            multi_image_input["text"][i] = multi_image_input["text"][i].replace(
                "<image>", "<|vision_start|><|image_pad|><|vision_end|>"
            )
        inputs = processor(
            text=multi_image_input["text"],
            images=list(multi_image_input["images"]),
            return_tensors="pt",
            padding=True,
        ).to(language_model_config.device)
        outputs = model(**inputs, output_hidden_states=True)
        activations_from_model = {hook_points[i]: outputs.hidden_states[i + 1] for i in range(28)}
        for key in activations:
            assert torch.allclose(input=activations[key], other=activations_from_model[key], atol=1e-3)

    def test_trace(self, language_model_config: LanguageModelConfig, single_image_input: dict[str, Any]):
        hf_language_model = Qwen_VL_LanguageModel(language_model_config)
        hf_language_model.model.eval()
        trace = hf_language_model.trace(single_image_input)
        assert trace[2][6] == (21, 14, 42, 28)


class Test_Qwen_LanguageModel:
    @pytest.fixture
    def language_model_config(self) -> LanguageModelConfig:
        return LanguageModelConfig(
            model_name="Qwen/Qwen2.5-7B",
            device="cuda",
            dtype=torch.bfloat16,
            d_model=3584,
        )

    @pytest.fixture
    def raw_input(self) -> dict[str, Any]:
        return {
            "text": ["Hello, world!" * i for i in range(10)],
        }

    def test_to_activations(self, language_model_config: LanguageModelConfig, raw_input: dict[str, Any]):
        hook_points = [f"blocks.{i}.hook_resid_post" for i in range(28)]
        hf_language_model = Qwen_LanguageModel(language_model_config)
        hf_language_model.model.eval()
        activations = hf_language_model.to_activations(raw_input, hook_points)
        assert len(activations) == 28

    def test_trace(self, language_model_config: LanguageModelConfig, raw_input: dict[str, Any]):
        hf_language_model = Qwen_LanguageModel(language_model_config)
        hf_language_model.model.eval()
        trace = hf_language_model.trace(raw_input)
        print(trace)
