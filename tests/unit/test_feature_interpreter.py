import pytest
from pytest_mock import MockerFixture
import openai
from lm_saes.analysis.auto_interp import (
    AutoInterpConfig,
    AutoInterpExplanation,
    AutoInterpEvaluation,
    FeatureInterpreter,
    Segment,
    TokenizedSample,
    generate_activating_examples,
    generate_non_activating_examples,
)
import torch
from typing import Any, Callable, Dict, List, Optional
from lm_saes.backend.language_model import LanguageModel
from lm_saes.database import MongoClient, FeatureAnalysis, FeatureAnalysisSampling, FeatureRecord
from lm_saes.abstract_sae import AbstractSparseAutoEncoder

# @pytest.mark.skip()
class TestTokenizedSample:
    @pytest.fixture
    def sample_text(self):
        return "This is a sample text for testing."

    @pytest.fixture
    def sample_activations(self):
        # Length should match potential tokenization of sample_text
        return [0.1, 0.2, 0.9, 0.8, 0.1, 0.05, 0.7, 0.1]

    @pytest.fixture
    def sample_origins(self):
        # Simplified origins matching sample_activations length
        # In reality, these ranges would be precise character indices
        return [
            {"key": "text", "range": (0, 4)},  # "This"
            {"key": "text", "range": (4, 7)},  # " is"
            {"key": "text", "range": (7, 9)},  # " a"
            {"key": "text", "range": (9, 16)},  # " sample"
            {"key": "text", "range": (16, 21)},  # " text"
            {"key": "text", "range": (21, 25)},  # " for"
            {"key": "text", "range": (25, 33)},  # " testing"
            {"key": "text", "range": (33, 34)},  # "."
        ]

    @pytest.fixture
    def sample_tokenized_sample(self):
        """Creates a sample TokenizedSample instance."""
        # A more realistic construct simulation:
        segments = [
            Segment("This", 0.1),
            Segment(" is", 0.2),
            Segment(" a", 0.9),
            Segment(" sample", 0.8),
            Segment(" text", 0.1),
            Segment(" for", 0.05),
            Segment(" testing", 0.7),
            Segment(".", 0.1),
        ]
        max_activation = 0.9
        return TokenizedSample(segments=segments, max_activation=max_activation)

    def test_construct(self, sample_text, sample_activations, sample_origins, sample_tokenized_sample):
        tokenized_sample = TokenizedSample.construct(
            text=sample_text, activations=sample_activations, origins=sample_origins, max_activation=0.9
        )
        assert tokenized_sample.segments == sample_tokenized_sample.segments
        assert tokenized_sample.max_activation == 0.9

    def test_display_highlighted(self, sample_tokenized_sample):
        highlighted_text = sample_tokenized_sample.display_highlighted(threshold=0.7)
        assert highlighted_text == "This is<< a>><< sample>> text for<< testing>>."

    def test_display_plain(self, sample_tokenized_sample):
        plain_text = sample_tokenized_sample.display_plain()
        assert plain_text == "This is a sample text for testing."


class TestFeatureInterpreter:
    def setup_method(self):
        self.cfg = AutoInterpConfig(
            openai_api_key="test",
            openai_model="test",
            openai_base_url="test",
            n_activating_examples=3,
            n_non_activating_examples=3,
            include_cot=True,
            detection_n_examples=3,
            fuzzing_n_examples=3,
        )
        self.feature_interpreter = FeatureInterpreter(self.cfg)


    @pytest.fixture
    def sample_tokenized_sample(self):
        """Creates a sample TokenizedSample instance."""
        # A more realistic construct simulation:
        segments = [
            Segment("This", 0.1),
            Segment(" is", 0.2),
            Segment(" a", 0.9),
            Segment(" sample", 0.8),
            Segment(" text", 0.1),
            Segment(" for", 0.05),
            Segment(" testing", 0.7),
            Segment(".", 0.1),
        ]
        max_activation = 0.9
        return TokenizedSample(segments=segments, max_activation=max_activation)

    @pytest.fixture
    def non_activating_samples(self, mocker):
        mock_1 = mocker.MagicMock(spec=TokenizedSample)
        mock_1.display_plain.return_value = "It was a remarkable breakthrough campaign that saw him named the Jimmy Murphy Academy Player of the Year, an award"

        mock_2 = mocker.MagicMock(spec=TokenizedSample)
        mock_2.display_plain.return_value = "about one-third of Americans hold both liberal and conservative views, depending on the specific issue. Another Pew"

        mock_3 = mocker.MagicMock(spec=TokenizedSample)
        mock_3.display_plain.return_value = "Wish You Were Listed. Patanjali has pitchforked itself into the top"

        mocks = [mock_1, mock_2, mock_3]
        return mocks

    @pytest.fixture
    def activating_samples(self, mocker):

        mock_1 = mocker.MagicMock(spec=TokenizedSample)
        mock_1.display_plain.return_value = "to pay in cash to avoid bank fees from credit card machines. He says he plans to donate a portion"
        mock_1.display_highlighted.return_value = "to pay in cash to avoid<< bank>> fees from credit<< card>><< machines>>. He says he plans to donate a portion"

        mock_2 = mocker.MagicMock(spec=TokenizedSample)
        mock_2.display_plain.return_value = "the original amount is released back to your credit card, but some banks take upwards of 10 working days"
        mock_2.display_highlighted.return_value = "the original amount is released back to your credit<< card>> , but some banks take upwards of 10 working days"

        mock_3 = mocker.MagicMock(spec=TokenizedSample)
        mock_3.display_plain.return_value = "pay many pounds extra to use a debit or credit card"
        mock_3.display_highlighted.return_value = "pay many pounds extra to use a<< debit>> or<< credit>><< card>>"

        mocks = [mock_1, mock_2, mock_3]
        return mocks

    @pytest.fixture
    def mock_explanation(self, mocker):
        mock = mocker.MagicMock(spec=AutoInterpExplanation)
        mock.final_explanation = "This feature activates on the token '!'."
        return {"response": mock}

    @pytest.fixture
    def mock_evaluation(self, mocker):
        mock = mocker.MagicMock(spec=AutoInterpEvaluation)
        mock.evaluation_results = [True, False, True, False, True, False]
        return mock


    def test_generate_explanation(self, activating_samples, mock_explanation):
        explanation = (
            self.feature_interpreter.generate_explanation(activating_samples)
            if self.cfg.openai_api_key != "test"
            else mock_explanation
        )
        assert explanation is not None
        assert isinstance(explanation, dict)
        print("----generated explanation----\n\n")
        for k, v in explanation.items():
            print(f"{k}:\n{v}\n\n")

    # @pytest.mark.skip()
    def test_evaluate_explanation_detection(
        self, activating_samples, non_activating_samples, mock_explanation, mock_evaluation, mocker
    ):
        explanation: AutoInterpExplanation = (
            self.feature_interpreter.generate_explanation(activating_samples)["response"]
            if self.cfg.openai_api_key != "test"
            else mock_explanation["response"]
        )
        if self.cfg.openai_api_key == "test":
            mock_client = mocker.MagicMock()
            mock_response = mocker.MagicMock()
            mock_response.choices[0].message.parsed = mock_evaluation
            mock_client.configure_mock(
                **{
                    "beta.chat.completions.parse.return_value": mock_response
                }
            )
            self.feature_interpreter.explainer_client = mock_client
        result = self.feature_interpreter.evaluate_explanation_detection(
            explanation, activating_samples, non_activating_samples
        )
        assert result is not None
        assert isinstance(result, dict)
        print("----evaluation result----\n\n")
        for k, v in result.items():
            print(f"{k}:\n{v}\n\n")

    # @pytest.mark.skip()
    def test_create_incorrectly_marked_example(self, sample_tokenized_sample: TokenizedSample):
        threshold = self.feature_interpreter.cfg.activation_threshold
        incorrectly_marked_example = self.feature_interpreter._create_incorrectly_marked_example(sample_tokenized_sample)
        assert all([origin.activation < threshold * sample_tokenized_sample.max_activation for origin, marked in zip(sample_tokenized_sample.segments, incorrectly_marked_example.segments) if marked.activation > 0])

    # @pytest.mark.skip()
    def test_evaluate_explanation_fuzzing(self, activating_samples, mock_explanation, mock_evaluation, mocker):
        explanation: AutoInterpExplanation = (
            self.feature_interpreter.generate_explanation(activating_samples)["response"]
            if self.cfg.openai_api_key != "test"
            else mock_explanation["response"]
        )
        if self.cfg.openai_api_key == "test":
            mock_client = mocker.MagicMock()
            mock_response = mocker.MagicMock()
            mock_response.choices[0].message.parsed = mock_evaluation
            mock_client.configure_mock(
                **{
                    "beta.chat.completions.parse.return_value": mock_response
                }
            )
            self.feature_interpreter.explainer_client = mock_client

        result = self.feature_interpreter.evaluate_explanation_fuzzing(explanation, activating_samples)
        assert result is not None
        assert isinstance(result, dict)
        print("----evaluation result----\n\n")
        for k, v in result.items():
            print(f"{k}:\n{v}\n\n")

# @pytest.mark.skip()
class TestGenerateExamples:
    
    @pytest.fixture
    def model(self, mocker):
        mock = mocker.MagicMock(spec=LanguageModel)
        mock.trace.return_value = [[
            {"key": "text", "range": (0, 4)},  # "This"
            {"key": "text", "range": (4, 7)},  # " is"
            {"key": "text", "range": (7, 9)},  # " a"
            {"key": "text", "range": (9, 16)},  # " sample"
            {"key": "text", "range": (16, 21)},  # " text"
            {"key": "text", "range": (21, 25)},  # " for"
            {"key": "text", "range": (25, 33)},  # " testing"
            {"key": "text", "range": (33, 34)},  # "."
        ]]
        mock.to_activations = lambda _, hook_points: {hook_point: torch.randn(1, 8, 10) for hook_point in hook_points}
        return mock

    @pytest.fixture
    def feature_analysis(self, mocker):
        mock = mocker.MagicMock(spec=FeatureAnalysis)
        mock.name = "test_analysis"
        mock.max_feature_acts = 0.7
        mock.samplings = [FeatureAnalysisSampling(
            name="top_activations",
            dataset_name=["test_dataset"],
            context_idx=[0],
            model_name=["test_model"],
            feature_acts=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
        ), FeatureAnalysisSampling(
            name="non_activating",
            dataset_name=["test_dataset"],
            context_idx=[0],
            model_name=["test_model"],
            feature_acts=[[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]],
        )]
        return mock

    @pytest.fixture
    def datasets(self, mocker):
        mock = mocker.MagicMock(spec=Callable[[str, int, int], Any])
        mock.return_value = [
            {
                "text": "This is a sample text for testing.",
            }
        ]
        return mock

    @pytest.fixture
    def mongo_client(self, mocker, feature_analysis):
        mock = mocker.MagicMock(spec=MongoClient)
        mock.get_feature.return_value = FeatureRecord(
            sae_name="test",
            sae_series="test",
            index=0,
            analyses=[feature_analysis],
        )
        return mock


    def test_generate_activating_examples(self, model, datasets, mongo_client, feature_analysis):
        activating_examples = generate_activating_examples(
            feature_index=0,
            model=model,
            datasets=datasets,
            mongo_client=mongo_client,
            sae_name="test",
            sae_series="test",
            analysis_name="test_analysis",
            n=1,
        )
        assert len(activating_examples) == 1
        assert isinstance(activating_examples[0], TokenizedSample)


    def test_generate_non_activating_examples(self, model, datasets, mongo_client, feature_analysis):
        non_activating_examples = generate_non_activating_examples(
            feature_index=0,
            model=model,
            datasets=datasets,
            mongo_client=mongo_client,
            sae_name="test",
            sae_series="test",
            analysis_name="test_analysis",
            n=1,
        )
        assert len(non_activating_examples) == 1
        assert isinstance(non_activating_examples[0], TokenizedSample)