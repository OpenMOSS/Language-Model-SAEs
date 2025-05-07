"""Auto-interpretation functionality for SAE features.

This module provides tools for automatically interpreting and evaluating sparse autoencoder features
based on the EleutherAI auto-interp approach (https://blog.eleuther.ai/autointerp/).

It includes:
1. Methods for prompting LLMs to generate explanations for features
2. Methods for evaluating explanations via different techniques:
   - Detection: Having LLMs identify if examples contain a feature
   - Fuzzing: Having LLMs identify correctly marked activating tokens
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Literal, Optional

import torch
from datasets import Dataset
from pydantic import BaseModel, Field

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.backend.language_model import LanguageModel
from lm_saes.config import BaseConfig, BaseSAEConfig
from lm_saes.database import MongoClient


class ExplainerType(str, Enum):
    """Types of LLM explainers supported."""

    OPENAI = "openai"


class ScorerType(str, Enum):
    """Types of explanation scoring methods."""

    DETECTION = "detection"
    FUZZING = "fuzzing"
    GENERATION = "generation"
    SIMULATION = "simulation"


class Step(BaseModel):
    """A step in the chain-of-thought process."""

    thought: str
    """The thought of the step."""

    output: str
    """The output of the step."""


class AutoInterpExplanation(BaseModel):
    """The result of an auto-interpretation of a SAE feature."""

    steps: list[Step]
    """The steps of the chain-of-thought process."""

    final_explanation: str
    """The explanation of the feature."""

    activation_consistency: Literal[1, 2, 3, 4, 5]
    """The consistency of the feature."""

    complexity: Literal[1, 2, 3, 4, 5]
    """The complexity of the feature."""


class AutoInterpEvaluation(BaseModel):
    """The result of an auto-interpretation of a SAE feature."""

    steps: list[Step]
    """The steps of the chain-of-thought process."""

    evaluation_results: list[bool]
    """The evaluation results for each example. Should be a list of YES/NO values."""


class AutoInterpConfig(BaseConfig):
    """Configuration for automatic interpretation of SAE features."""

    # LLM settings
    explainer_type: ExplainerType = ExplainerType.OPENAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_base_url: Optional[str] = None

    # Activation retrieval settings
    n_activating_examples: int = 7
    n_non_activating_examples: int = 20
    activation_threshold: float = 0.7  # Threshold relative to max activation for highlighting tokens
    max_length: int = 50

    # Scoring settings
    scorer_type: List[ScorerType] = Field(default_factory=lambda: [ScorerType.DETECTION, ScorerType.FUZZING])

    # Detection settings
    detection_n_examples: int = 5  # Number of examples to show for detection

    # Fuzzing settings
    fuzzing_n_examples: int = 5  # Number of examples to use for fuzzing
    fuzzing_decile_correct: int = 5  # Number of correctly marked examples per decile
    fuzzing_decile_incorrect: int = 2  # Number of incorrectly marked examples per decile

    # Prompting settings
    include_cot: bool = True  # Whether to use chain-of-thought prompting


@dataclass
class Segment:
    """A segment of text with its activation value."""

    text: str
    """The text of the segment."""

    activation: float
    """The activation value of the segment."""

    def display(self, abs_threshold: float) -> str:
        """Display the segment as a string with whether it's highlighted."""
        if self.activation >= abs_threshold:
            return f"<<{self.text}>>"
        else:
            return self.text


@dataclass
class TokenizedSample:
    """A tokenized sample with its activation pattern organized into segments."""

    segments: list[Segment]
    """List of segments, each containing start/end positions and activation values."""

    max_activation: float
    """Global maximum activation value."""

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

    @staticmethod
    def construct(
        text: str,
        activations: list[float],
        origins: list[dict[str, Any]],
        max_activation: float,
    ) -> "TokenizedSample":
        """Construct a TokenizedSample from text, activations, and origins."""
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
            segment_activation = max(
                act
                for origin, act in zip(origins, activations)
                if origin and origin["key"] == "text" and origin["range"][0] >= start and origin["range"][1] <= end
            )
            segments.append(Segment(text[start:end], segment_activation))

        return TokenizedSample(segments, max_activation)


def generate_activating_examples(
    feature_index: int,
    model: LanguageModel,
    datasets: Callable[[str, int, int], Dataset],
    mongo_client: MongoClient,
    sae_name: str,
    sae_series: str | None,
    analysis_name: str = "default",
    n: int = 10,
    max_length: int = 50,
) -> list[TokenizedSample]:
    """Generate examples where a feature strongly activates using database records.

    Args:
        feature_index: Index of the feature to analyze
        model: Language model to use
        sae: SAE model to use
        mongo_client: MongoDB client to fetch examples
        sae_name: Name of the SAE
        sae_series: Series of the SAE
        analysis_name: Name of the analysis
        n: Maximum number of examples to generate

    Returns:
        List of TokenizedExample with high activation for the feature
    """
    samples: list[TokenizedSample] = []

    # Get feature information from MongoDB
    feature = mongo_client.get_feature(sae_name, sae_series, feature_index)
    if not feature:
        raise ValueError(f"Feature {feature_index} not found for SAE {sae_name}/{sae_series}")

    # Find the analysis by name
    analysis = next((a for a in feature.analyses if a.name == analysis_name), None)
    if not analysis:
        raise ValueError(f"Analysis {analysis_name} not found for feature {feature_index}")

    # Get examples from each sampling
    sampling = analysis.samplings[0]
    for i, (dataset_name, shard_idx, n_shards, context_idx, feature_acts) in enumerate(
        zip(
            sampling.dataset_name,
            sampling.shard_idx if sampling.shard_idx else [0] * len(sampling.dataset_name),
            sampling.n_shards if sampling.n_shards else [1] * len(sampling.dataset_name),
            sampling.context_idx,
            sampling.feature_acts,
        )
    ):
        dataset = datasets(dataset_name, shard_idx, n_shards)
        data = dataset[context_idx]

        # Process the sample using model's trace method
        origins = model.trace({k: [v] for k, v in data.items()})[0]

        max_act_pos = torch.argmax(torch.tensor(feature_acts)).item()

        left_end = max(0, max_act_pos - max_length // 2)
        right_end = min(len(origins), max_act_pos + max_length // 2)

        # Create TokenizedExample using the trace information
        sample = TokenizedSample.construct(
            text=data["text"],
            activations=feature_acts[left_end:right_end],
            origins=origins[left_end:right_end],
            max_activation=analysis.max_feature_acts,
        )

        samples.append(sample)

        if len(samples) >= n:
            break

    return samples


def generate_non_activating_examples(
    feature_index: int,
    model: LanguageModel,
    datasets: Callable[[str, int, int], Dataset],
    mongo_client: MongoClient,
    sae_name: str,
    sae_series: str | None,
    analysis_name: str = "default",
    n: int = 10,
    max_length: int = 50,
) -> list[TokenizedSample]:
    """Generate examples where a feature doesn't activate much.

    Args:
        feature_index: Index of the feature to analyze
        model: Language model to use
        sae: SAE model to use
        mongo_client: MongoDB client to fetch examples
        sae_name: Name of the SAE
        sae_series: Series of the SAE
        analysis_name: Name of the analysis
        n: Maximum number of examples to generate

    Returns:
        List of TokenizedExample with low activation for the feature
    """

    samples: list[TokenizedSample] = []

    # Get feature information from MongoDB
    feature = mongo_client.get_feature(sae_name, sae_series, feature_index)
    if not feature:
        raise ValueError(f"Feature {feature_index} not found for SAE {sae_name}/{sae_series}")

    # Find the analysis by name
    analysis = next((a for a in feature.analyses if a.name == analysis_name), None)
    if not analysis:
        raise ValueError(f"Analysis {analysis_name} not found for feature {feature_index}")

    # Get examples from each sampling
    sampling = analysis.samplings[-1]
    assert sampling.name == "non_activating", f"Sampling {sampling.name} is not non_activating"
    for i, (dataset_name, shard_idx, n_shards, context_idx, feature_acts) in enumerate(
        zip(
            sampling.dataset_name,
            sampling.shard_idx if sampling.shard_idx else [0] * len(sampling.dataset_name),
            sampling.n_shards if sampling.n_shards else [1] * len(sampling.dataset_name),
            sampling.context_idx,
            sampling.feature_acts,
        )
    ):
        try:
            dataset = datasets(dataset_name, shard_idx, n_shards)
            data = dataset[context_idx]

            # Process the sample using model's trace method
            origins = model.trace({k: [v] for k, v in data.items()})[0]

            # Create TokenizedExample using the trace information
            sample = TokenizedSample.construct(
                text=data["text"],
                activations=feature_acts[:max_length],
                origins=origins[:max_length],
                max_activation=analysis.max_feature_acts,
            )

            samples.append(sample)

        except Exception as e:
            print(f"Error processing example {i} from sampling non-activating: {e}")
            continue

        if len(samples) >= n:
            break

    return samples


class FeatureInterpreter:
    """A class for generating and evaluating explanations for SAE features."""

    def __init__(self, cfg: AutoInterpConfig, mongo_client: Optional[MongoClient] = None):
        """Initialize the feature interpreter.

        Args:
            cfg: Configuration for interpreter
            mongo_client: Optional MongoDB client for retrieving data
        """
        self.cfg = cfg
        self.mongo_client = mongo_client

        # Set up LLM client for explanation generation
        self._setup_llm_clients()

    def _setup_llm_clients(self):
        """Set up OpenAI client for explanation generation and evaluation."""
        try:
            import openai
            self.explainer_client = openai.Client(base_url=self.cfg.openai_base_url, api_key=self.cfg.openai_api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Please install it with `uv add openai`.")

    def get_feature_examples(
        self,
        feature_index: int,
        model: LanguageModel,
        datasets: Callable[[str, int, int], Dataset],
        mongo_client: MongoClient,
        sae_name: str,
        sae_series: str | None,
        analysis_name: str = "default",
        max_length: int = 50,
    ) -> tuple[list[TokenizedSample], list[TokenizedSample]]:
        """Get activating and non-activating examples for a feature."""
        activating_examples = generate_activating_examples(
            feature_index=feature_index,
            model=model,
            datasets=datasets,
            mongo_client=mongo_client,
            sae_name=sae_name,
            sae_series=sae_series,
            analysis_name=analysis_name,
            max_length=max_length,
        )
        non_activating_examples = generate_non_activating_examples(
            feature_index=feature_index,
            model=model,
            datasets=datasets,
            mongo_client=mongo_client,
            sae_name=sae_name,
            sae_series=sae_series,
            analysis_name=analysis_name,
            max_length=max_length,
        )
        return activating_examples, non_activating_examples

    def _generate_explanation_prompt(self, activating_examples: list[TokenizedSample]) -> tuple[str, str]:
        """Generate a prompt for explanation generation.

        Args:
            activating_examples: List of activating examples

        Returns:
            Prompt string for the LLM
        """
        cot_prompt = ""
        if self.cfg.include_cot:
            cot_prompt += "\n\nTo explain this feature, please follow these steps:\n"
            cot_prompt += "Step 1: List a couple activating and contextual tokens you find interesting. "
            cot_prompt += "Search for patterns in these tokens, if there are any. Don't list more than 5 tokens.\n"
            cot_prompt += "Step 2: Write down general shared features of the text examples.\n"
            cot_prompt += "Step 3: Write a concise explanation of what this feature detects.\n"

        examples_prompt = """Some examples:

The feature activates on the word 'knows' in rhetorical questions.
Activation Consistency: 5
Complexity: 4

The feature activates on verbs related to decision-making and preferences.
Activation Consistency: 4
Complexity: 4

The feature activates on the substring 'Ent' at the start of words
Activation Consistency: 5
Complexity: 1

The feature activates on text about government economic policy
Activation Consistency: 3
Complexity: 5
"""
        system_prompt = f"""We're studying features in a neural network. Each feature activates on some particular word/words/substring/concept in a short document. The activating words in each document are indicated with << ... >>. We will give you a list of documents on which the feature activates, in order from most strongly activating to least strongly activating.

Your task is to:

First, Summarize the Activation: Look at the parts of the document the feature activates for and summarize in a single sentence what the feature is activating on. Try not to be overly specific in your explanation. Note that some features will activate only on specific words or substrings, but others will activate on most/all words in a sentence provided that sentence contains some particular concept. Your explanation should cover most or all activating words (for example, don't give an explanation which is specific to a single word if all words in a sentence cause the feature to activate). Pay attention to things like the capitalization and punctuation of the activating words or concepts, if that seems relevant. Keep the explanation as short and simple as possible, limited to 20 words or less. Omit punctuation and formatting. You should avoid giving long lists of words.{cot_prompt}

Second, Assess Activation Consistency: Based on your summary and the provided examples, evaluate the consistency of the feature's activation. Return your assessment as a single integer from the following scale:

5: Clear pattern with no deviating examples
4: Clear pattern with one or two deviating examples
3: Clear overall pattern but quite a few examples not fitting that pattern
2: Broad consistent theme but lacking structure
1: No discernible pattern

Third, Assess Feature Complexity: Based on your summary and the nature of the activation, evaluate the complexity of the feature. Return your assessment as a single integer from the following scale:

5: Rich feature firing on diverse contexts with an interesting unifying theme, e.g., “feelings of togetherness”
4: Feature relating to high-level semantic structure, e.g., “return statements in code”
3: Moderate complexity, such as a phrase, category, or tracking sentence structure, e.g., “website URLs”
2: Single word or token feature but including multiple languages or spelling, e.g., “mentions of dog”
1: Single token feature, e.g., “the token ‘(‘”

Your final output should first provide the summary sentence in the form "This feature activates on...", then on a new line the Activation Consistency score in the form "Activation Consistency: X", and on another new line the Complexity score in the form "Complexity: X".

{examples_prompt}
"""



        user_prompt = "The activating documents are given below:\n\n"
        # Select a subset of examples to show
        examples_to_show = activating_examples[:self.cfg.n_activating_examples]

        for i, example in enumerate(examples_to_show, 1):
            highlighted = example.display_highlighted(self.cfg.activation_threshold)
            user_prompt += f"Example {i}: {highlighted}\n\n"

        return system_prompt, user_prompt

    def generate_explanation(self, activating_examples: list[TokenizedSample]) -> dict[str, Any]:
        """Generate an explanation for a feature based on activating examples.

        Args:
            activating_examples: List of examples where the feature activates

        Returns:
            Dictionary with explanation and metadata
        """
        system_prompt, user_prompt = self._generate_explanation_prompt(activating_examples)

        response = self.explainer_client.beta.chat.completions.parse(
            model=self.cfg.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=AutoInterpExplanation,
        )
        explanation = response.choices[0].message.parsed
        assert explanation is not None, "No explanation returned from OpenAI"
        return {"user_prompt": user_prompt, "system_prompt": system_prompt, "response": explanation}

    def _generate_detection_prompt(self, explanation: AutoInterpExplanation, examples: list[TokenizedSample]) -> tuple[str, str]:
        """Generate a prompt for detection evaluation.

        Args:
            explanation: The explanation to evaluate
            examples: List of examples (mix of activating and non-activating)

        Returns:
            Prompt string for the LLM
        """
        system_prompt = f"""We're studying features in a neural network. Each feature activates on some particular word/words/substring/concept in a short document. You will be given a short explanation of what this feature activates for, and then be shown {len(examples)} example sequences in random order. You will have to return a boolean list of the examples where you think the feature should activate at least once, on ANY of the words or substrings in the document, true if it does, false if it doesn't. Try not to be overly specific in your interpretation of the explanation."""
        user_prompt = f"Here is the explanation:\n\n{explanation.final_explanation}\n\nHere are the examples:\n\n"

        for i, example in enumerate(examples, 1):
            user_prompt += f"Example {i}: {example.display_plain()}\n"

        return system_prompt, user_prompt

    def evaluate_explanation_detection(
        self,
        explanation: AutoInterpExplanation,
        activating_examples: list[TokenizedSample],
        non_activating_examples: list[TokenizedSample],
    ) -> dict[str, Any]:
        """Evaluate an explanation using the detection method.

        Args:
            explanation: The explanation to evaluate
            activating_examples: Examples where the feature activates
            non_activating_examples: Examples where the feature doesn't activate

        Returns:
            Dictionary with evaluation results
        """
        # Select a subset of examples
        n_activating = min(self.cfg.detection_n_examples, len(activating_examples))
        n_non_activating = min(self.cfg.detection_n_examples, len(non_activating_examples))

        test_activating = random.sample(activating_examples, n_activating) if n_activating > 0 else []
        test_non_activating = random.sample(non_activating_examples, n_non_activating) if n_non_activating > 0 else []

        # Mix and shuffle examples
        all_examples = test_activating + test_non_activating
        if len(all_examples) < self.cfg.detection_n_examples:
            return {
                "method": "detection",
                "prompt": "",
                "response": "",
                "ground_truth": [],
                "predictions": [],
                "metrics": {
                    "accuracy": 0,
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                    "balanced_accuracy": 0,
                },
                "passed": False,
            }

        random.shuffle(all_examples)

        # Ground truth for each example (1 for activating, 0 for non-activating)
        ground_truth = [1 if ex in test_activating else 0 for ex in all_examples]

        # Generate prompt
        system_prompt, user_prompt = self._generate_detection_prompt(explanation, all_examples)

        # Get response from OpenAI
        response = self.explainer_client.beta.chat.completions.parse(
            model=self.cfg.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=AutoInterpEvaluation,
        )
        detection_response = response.choices[0].message.parsed
        assert detection_response is not None, "No detection response returned from OpenAI"
        predictions = detection_response.evaluation_results

        # Pad predictions if needed
        predictions = predictions[: len(ground_truth)]
        if len(predictions) < len(ground_truth):
            predictions.extend([False] * (len(ground_truth) - len(predictions)))

        # Calculate metrics
        tp = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 1 and pred == 1)
        tn = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 0 and pred == 0)
        fp = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 0 and pred == 1)
        fn = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 1 and pred == 0)

        accuracy = (tp + tn) / len(ground_truth) if ground_truth else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        balanced_accuracy = ((tp / (tp + fn) if (tp + fn) > 0 else 0) + (tn / (tn + fp) if (tn + fp) > 0 else 0)) / 2

        return {
            "method": "detection",
            "prompt": system_prompt + "\n\n" + user_prompt,
            "response": detection_response,
            "ground_truth": ground_truth,
            "predictions": predictions,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "balanced_accuracy": balanced_accuracy,
            },
            "passed": balanced_accuracy >= 0.7,  # Arbitrary threshold for passing
        }

    def _generate_fuzzing_prompt(
        self,
        explanation: AutoInterpExplanation,
        examples: list[tuple[TokenizedSample, bool]],  # (sample, is_correctly_marked)
    ) -> tuple[str, str]:
        """Generate a prompt for fuzzing evaluation.

        Args:
            explanation: The explanation to evaluate
            examples: List of tuples (example, is_correctly_marked)

        Returns:
            Prompt string for the LLM
        """
        system_prompt = f"""We're studying features in a neural network. Each feature activates on some particular word/words/substring/concept in a short document. You will be given a short explanation of what this feature activates for, and then be shown {len(examples)} example sequences in random order. In each example, text segments highlighted with << >> are presented as activating the feature as described in the explanation. You will have to return a boolean list of the examples where you think the highlighted parts CORRECTLY correspond to the explanation, true if they do, false if they don't. Try not to be overly specific in your interpretation of the explanation."""
        user_prompt = f"Here is the explanation:\n\n{explanation.final_explanation}\n\nHere are the examples:\n\n"

        for i, (example, _) in enumerate(examples, 1):
            highlighted = example.display_highlighted(self.cfg.activation_threshold)
            user_prompt += f"Example {i}: {highlighted}\n"

        return system_prompt, user_prompt

    def _create_incorrectly_marked_example(self, sample: TokenizedSample) -> TokenizedSample:
        """Create an incorrectly marked version of an example.

        Args:
            sample: The original sample

        Returns:
            A copy of the sample with incorrect highlighting
        """
        # Count how many tokens would be highlighted in the correct example
        threshold = self.cfg.activation_threshold
        n_highlighted = sum(1 for seg in sample.segments if seg.activation >= threshold * sample.max_activation)

        def highlight_random_tokens(sample: TokenizedSample, n_highlighted: int) -> TokenizedSample:
            non_activating_indices = [
                i for i, seg in enumerate(sample.segments) if seg.activation < threshold * sample.max_activation
            ]
            highlight_indices = random.sample(non_activating_indices, min(n_highlighted, len(non_activating_indices)))
            segments = [
                Segment(seg.text, sample.max_activation if i in highlight_indices else 0)
                for i, seg in enumerate(sample.segments)
            ]
            return TokenizedSample(segments, sample.max_activation)

        n_to_highlight = max(3, n_highlighted)  # Highlight at least 3 tokens
        return highlight_random_tokens(sample, n_to_highlight)

    def evaluate_explanation_fuzzing(
        self, explanation: AutoInterpExplanation, activating_examples: List[TokenizedSample]
    ) -> Dict[str, Any]:
        """Evaluate an explanation using the fuzzing method.

        Args:
            explanation: The explanation to evaluate
            activating_examples: Examples where the feature activates

        Returns:
            Dictionary with evaluation results
        """
        if len(activating_examples) < self.cfg.fuzzing_n_examples:
            # Not enough examples, return empty result
            return {
                "method": "fuzzing",
                "prompt": "",
                "response": "",
                "ground_truth": [],
                "predictions": [],
                "metrics": {
                    "accuracy": 0,
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                    "balanced_accuracy": 0,
                },
                "passed": False,
            }

        # Prepare examples:
        # - Correctly marked examples (original)
        # - Incorrectly marked examples (with wrong parts highlighted)
        n_correct = self.cfg.fuzzing_decile_correct
        n_incorrect = self.cfg.fuzzing_decile_incorrect

        # Get a sample of activating examples
        sample_examples = random.sample(activating_examples, min(n_correct + n_incorrect, len(activating_examples)))

        # Split into correct and incorrect
        correct_examples = sample_examples[:n_correct]
        incorrect_candidates = sample_examples[n_correct:]

        # Create incorrectly marked versions
        incorrect_examples = [self._create_incorrectly_marked_example(ex) for ex in incorrect_candidates]

        # Combine and mark with ground truth
        examples_with_labels = [(ex, True) for ex in correct_examples] + [(ex, False) for ex in incorrect_examples]

        # Shuffle
        random.shuffle(examples_with_labels)

        # Generate prompt
        system_prompt, user_prompt = self._generate_fuzzing_prompt(explanation, examples_with_labels)

        # Get response from OpenAI
        response = self.explainer_client.beta.chat.completions.parse(
            model=self.cfg.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=AutoInterpEvaluation,
        )
        fuzzing_response = response.choices[0].message.parsed
        assert fuzzing_response is not None, "No fuzzing response returned from OpenAI"
        # Parse response (CORRECT/INCORRECT for each example)
        predictions = fuzzing_response.evaluation_results
        # Pad predictions if needed
        predictions = predictions[: len(examples_with_labels)]
        if len(predictions) < len(examples_with_labels):
            predictions.extend([False] * (len(examples_with_labels) - len(predictions)))

        # Extract ground truth
        ground_truth = [is_correct for _, is_correct in examples_with_labels]

        # Calculate metrics
        tp = sum(1 for gt, pred in zip(ground_truth, predictions) if gt is True and pred is True)
        tn = sum(1 for gt, pred in zip(ground_truth, predictions) if gt is False and pred is False)
        fp = sum(1 for gt, pred in zip(ground_truth, predictions) if gt is False and pred is True)
        fn = sum(1 for gt, pred in zip(ground_truth, predictions) if gt is True and pred is False)

        accuracy = (tp + tn) / len(ground_truth) if ground_truth else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        balanced_accuracy = ((tp / (tp + fn) if (tp + fn) > 0 else 0) + (tn / (tn + fp) if (tn + fp) > 0 else 0)) / 2

        return {
            "method": "fuzzing",
            "prompt": user_prompt,
            "response": fuzzing_response,
            "ground_truth": ground_truth,
            "predictions": predictions,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "balanced_accuracy": balanced_accuracy,
            },
            "passed": balanced_accuracy >= 0.7,  # Arbitrary threshold for passing
        }

    def interpret_feature(
        self,
        sae_name: str,
        sae_series: str,
        feature_indices: List[int],
        model: LanguageModel,
        sae: BaseSAEConfig,
        datasets: Callable[[str, int, int], Dataset],
        analysis_name: str = "default",
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate and evaluate explanations for multiple features.

        Args:
            sae_name: Name of the SAE
            sae_series: Series of the SAE
            feature_indices: List of feature indices to interpret
            model: Language model to use for generating activations
            sae: SAE model to use for encoding
            dataset: Dataset to sample non-activating examples from
            analysis_name: Name of the analysis to use

        Returns:
            Dictionary mapping feature indices to their interpretation results
        """
        if not self.mongo_client:
            raise ValueError("MongoDB client not provided.")

        # Determine hook points based on SAE configuration
        hook_points = []
        layer = getattr(sae, "layer", 0)
        hook_points.append(f"blocks.{layer}.hook_resid_post")

        results = {}
        for feature_idx in feature_indices:
            # Generate activating examples from the database
            activating_examples, non_activating_examples = self.get_feature_examples(
                feature_index=feature_idx,
                model=model,
                datasets=datasets,
                mongo_client=self.mongo_client,
                sae_name=sae_name,
                sae_series=sae_series,
                analysis_name=analysis_name,
                max_length=self.cfg.max_length,
            )

            # Generate explanation for the feature
            explanation_result = self.generate_explanation(activating_examples)
            explanation: AutoInterpExplanation = explanation_result["response"]
            print(explanation.final_explanation + "\n\n")
            # Evaluate explanation
            evaluation_results = []

            if ScorerType.DETECTION in self.cfg.scorer_type:
                detection_result = self.evaluate_explanation_detection(
                    explanation, activating_examples, non_activating_examples
                )
                evaluation_results.append(detection_result)

            if ScorerType.FUZZING in self.cfg.scorer_type:
                fuzzing_result = self.evaluate_explanation_fuzzing(explanation, activating_examples)
                evaluation_results.append(fuzzing_result)


            yield {
                "feature_index": feature_idx,
                "sae_name": sae_name,
                "sae_series": sae_series,
                "analysis_name": analysis_name,
                "explanation": explanation.final_explanation,
                "complexity": explanation.complexity,
                "consistency": explanation.activation_consistency,
                "explanation_details": {k: v.model_dump() if isinstance(v, BaseModel) else v for k, v in explanation_result.items()},
                "evaluations": [
                    {k: v.model_dump() if isinstance(v, BaseModel) else v for k, v in eval_result.items()}
                    for eval_result in evaluation_results
                ],
                "passed": any(eval_result["passed"] for eval_result in evaluation_results),
            }
