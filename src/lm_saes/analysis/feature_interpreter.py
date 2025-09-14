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
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generator, Literal, Optional

import json_repair
import torch
from datasets import Dataset
from pydantic import BaseModel, Field

from lm_saes.backend.language_model import LanguageModel
from lm_saes.config import BaseConfig
from lm_saes.database import FeatureAnalysis, FeatureRecord, MongoClient
from lm_saes.utils.logging import get_logger

logger = get_logger("analysis.feature_interpreter")


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


class Step(BaseModel):
    """A step in the chain-of-thought process."""

    thought: str
    """The thought of the step."""

    # output: str
    # """The output of the step."""


Step_Schema = {
    "type": "object",
    "properties": {
        "thought": {"type": "string", "description": "The thought of the step."},
    },
}


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


AutoInterpExplanation_Schema = {
    "type": "object",
    "properties": {
        "steps": {"type": "array", "items": Step_Schema},
        "final_explanation": {
            "type": "string",
            "description": "The explanation of the feature, in the form of 'This feature activates on... '",
        },
        "activation_consistency": {
            "type": "integer",
            "description": "The consistency of the feature, on a scale of 1 to 5.",
        },
        "complexity": {"type": "integer", "description": "The complexity of the feature, on a scale of 1 to 5."},
    },
}


class AutoInterpEvaluation(BaseModel):
    """The result of an auto-interpretation of a SAE feature."""

    steps: list[Step]
    """The steps of the chain-of-thought process."""

    evaluation_results: list[bool]
    """The evaluation results for each example. Should be a list of YES/NO values."""


AutoInterpEvaluation_Schema = {
    "type": "object",
    "properties": {
        "steps": {"type": "array", "items": Step_Schema},
        "evaluation_results": {
            "type": "array",
            "items": {"type": "boolean"},
            "description": "The evaluation results for each example. Should be a list of True/False values.",
        },
    },
}


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
        if self.activation > abs_threshold:
            return f"{self.text}\n"
        else:
            return ""


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
    
    def display_max(self, threshold: float = 0.7) -> str:
        # max_activation_text = "".join([seg.display_max(threshold * self.max_activation) for seg in self.segments])
        max_activation_text = ""
        hash_ = {}
        for seg in self.segments:
            if seg.activation>threshold * self.max_activation:
                text = seg.text
                if text != "" and hash_.get(text, None) is None:
                    hash_[text] = 1
                    max_activation_text = text+"\n"
        return max_activation_text
    
    def display_next(self, threshold: float = 0.7) -> str:
        # max_activation_text = "".join([seg.display_max(threshold * self.max_activation) for seg in self.segments])
        next_activation_text = ""
        hash_ = {}
        Flag = False
        for seg in self.segments:
            if Flag:
                text = seg.text
                if text != "" and hash_.get(text, None) is None:
                    hash_[text] = 1
                    next_activation_text = text+"\n"
            if seg.activation>threshold * self.max_activation:
                Flag = True
            else:
                Flag = False
        return next_activation_text 

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
            # try:
            segment_activation = max(
                act
                for origin, act in zip(origins, activations)
                if origin and origin["key"] == "text" and origin["range"][0] >= start and origin["range"][1] <= end
            )
            # except Exception:
            #     logger.error(f"Error processing segment:\nstart={start}, end={end}, segment={text[start:end]}\n\n")
            #     continue
            segments.append(Segment(text[start:end], segment_activation))

        return TokenizedSample(segments, max_activation)

import numpy as np
def generate_activating_examples(
    feature: FeatureRecord,
    model: LanguageModel,
    datasets: Callable[[str, int, int], Dataset],
    analysis: FeatureAnalysis,
    n: int = 10,
    max_length: int = 50,
) -> list[TokenizedSample]:
    """Generate examples where a feature strongly activates using database records.

    Args:
        feature: FeatureRecord to analyze
        model: Language model to use
        datasets: Callable to fetch datasets
        analysis: FeatureAnalysis to use
        n: Maximum number of examples to generate
        max_length: Maximum length of examples to generate

    Returns:
        List of TokenizedExample with high activation for the feature
    """
    samples: list[TokenizedSample] = []
    error_prefix = f"Error processing activating examples of feature {feature.index}: "

    # Get examples from each sampling
    sampling = analysis.samplings[0]
    # print(f'{sampling.context_idx.shape=}')
    # print(f'{sampling.feature_acts_values.shape=} {sampling.feature_acts_indices=}')
    # feature_acts_ = torch.sparse_coo_tensor(torch.Tensor(sampling.feature_acts_indices), torch.Tensor(sampling.feature_acts_values), (1024, sampling.context_idx.shape[0]))
    feature_acts_ = torch.sparse_coo_tensor(torch.Tensor(sampling.feature_acts_indices), torch.Tensor(sampling.feature_acts_values), (int(np.max(sampling.feature_acts_indices[0])), 2048))
    feature_acts_ = feature_acts_.to_dense()
    
    for i, (dataset_name, shard_idx, n_shards, context_idx, feature_acts) in enumerate(
        zip(
            sampling.dataset_name,
            sampling.shard_idx if sampling.shard_idx is not None else [0] * len(sampling.dataset_name),
            sampling.n_shards if sampling.n_shards is not None else [1] * len(sampling.dataset_name),
            sampling.context_idx,
            feature_acts_
        )
    ):
        try:
        
            dataset = datasets(dataset_name, shard_idx, n_shards)
            # context_idx = context_idx.astype(int)
            data = dataset[int(context_idx)]

            # Process the sample using model's trace method
            origins = model.trace({k: [v] for k, v in data.items()})[0]

            max_act_pos = torch.argmax(torch.tensor(feature_acts)).item()
            # print(f'{max_act_pos=}')
            # print(f'{feature_acts=}')

            left_end = max(0, max_act_pos - max_length // 2)
            right_end = min(len(origins), max_act_pos + max_length // 2)

            # Create TokenizedExample using the trace information
            sample = TokenizedSample.construct(
                text=data["text"],
                activations=feature_acts[left_end:right_end],
                origins=origins[left_end:right_end],
                max_activation=analysis.max_feature_acts,
            )
            # print('run activating')
            samples.append(sample)

        except Exception as e:
            logger.error(f"{error_prefix} {e}")
            continue

        if len(samples) >= n:
            break

    return samples


def generate_non_activating_examples(
    feature: FeatureRecord,
    model: LanguageModel,
    datasets: Callable[[str, int, int], Dataset],
    analysis: FeatureAnalysis,
    n: int = 10,
    max_length: int = 50,
) -> list[TokenizedSample]:
    """Generate examples where a feature doesn't activate much.

    Args:
        feature: FeatureRecord to analyze
        model: Language model to use
        datasets: Callable to fetch datasets
        analysis: FeatureAnalysis to use
        n: Maximum number of examples to generate
        max_length: Maximum length of examples to generate

    Returns:
        List of TokenizedExample with low activation for the feature
    """

    samples: list[TokenizedSample] = []
    error_prefix = f"Error processing non-activating examples of feature {feature.index}:"

    sampling_idx = -1
    for i in range(len(analysis.samplings)):
        if analysis.samplings[i].name == "non_activating":
            sampling_idx = i
            break
    if sampling_idx == -1:
        return samples
    sampling = analysis.samplings[sampling_idx]
    # print(f'{len(analysis.samplings)=}')
    # for sample in analysis.samplings:
    #     print(sample.name)
    assert sampling.name == "non_activating", f"{error_prefix} Sampling {sampling.name} is not non_activating"
    for i, (dataset_name, shard_idx, n_shards, context_idx, feature_acts_indices, feature_acts_values) in enumerate(
        zip(
            sampling.dataset_name,
            sampling.shard_idx if sampling.shard_idx else [0] * len(sampling.dataset_name),
            sampling.n_shards if sampling.n_shards else [1] * len(sampling.dataset_name),
            sampling.context_idx,
            # sampling.feature_acts,
            sampling.feature_acts_indices,
            sampling.feature_acts_values,
        )
    ):
        try:
            feature_acts = torch.sparse_coo_tensor(torch.Tensor(feature_acts_indices), torch.Tensor(feature_acts_values), (1024, sampling.context_idx.shape[0]))
            feature_acts = feature_acts.to_dense()
            
            dataset = datasets(dataset_name, shard_idx, n_shards)
            data = dataset[context_idx]

            # Process the sample using model's trace method
            # lock.acquire()
            origins = model.trace({k: [v] for k, v in data.items()})[0]
            # lock.release()

            # Create TokenizedExample using the trace information
            sample = TokenizedSample.construct(
                text=data["text"],
                activations=feature_acts[:max_length],
                origins=origins[:max_length],
                max_activation=analysis.max_feature_acts,
            )

            samples.append(sample)

        except Exception as e:
            logger.error(f"{error_prefix} {e}")
            continue

        if len(samples) >= n:
            break

    return samples


class FeatureInterpreter:
    """A class for generating and evaluating explanations for SAE features."""

    def __init__(self, cfg: AutoInterpConfig, mongo_client: MongoClient):
        """Initialize the feature interpreter.

        Args:
            cfg: Configuration for interpreter
            mongo_client: Optional MongoDB client for retrieving data
        """
        self.cfg = cfg
        self.mongo_client = mongo_client
        self.logits = None
        # Set up LLM client for explanation generation
        self._setup_llm_clients()

    def _setup_llm_clients(self):
        """Set up OpenAI client for explanation generation and evaluation."""
        try:
            import httpx
            import openai
            from openai import DefaultHttpxClient

            self.explainer_client = openai.Client(
                base_url=self.cfg.openai_base_url,
                api_key=self.cfg.openai_api_key,
                http_client=DefaultHttpxClient(
                    proxy=self.cfg.openai_proxy,
                    transport=httpx.HTTPTransport(local_address="0.0.0.0"),
                )
                if self.cfg.openai_proxy
                else None,
            )
        except ImportError:
            raise ImportError("OpenAI package not installed. Please install it with `uv add openai`.")

    def get_feature_examples(
        self,
        feature: FeatureRecord,
        model: LanguageModel,
        datasets: Callable[[str, int, int], Dataset],
        analysis_name: str = "default",
        max_length: int = 50,
    ) -> tuple[list[TokenizedSample], list[TokenizedSample]]:
        """Get activating and non-activating examples for a feature."""
        analysis = next((a for a in feature.analyses if a.name == analysis_name), None)
        if not analysis:
            raise ValueError(f"Analysis {analysis_name} not found for feature {feature.index}")

        if analysis.max_feature_acts == 0:
            raise ValueError(f"Feature {feature.index} has no activation. Skipping interpretation.")

        # Get examples from each sampling
        activating_examples = generate_activating_examples(
            feature=feature,
            model=model,
            datasets=datasets,
            analysis=analysis,
            max_length=max_length,
        )
        non_activating_examples = generate_non_activating_examples(
            feature=feature,
            model=model,
            datasets=datasets,
            analysis=analysis,
            max_length=max_length,
        )
        # self.logits = None
        return activating_examples, non_activating_examples

    def _generate_explanation_prompt_neuronpedia(self, activating_examples: list[TokenizedSample]) -> tuple[str, str]:
        """Generate a prompt for explanation generation with neuronpedia.

        Args:
            activating_examples: List of activating examples

        Returns:
            Prompt string for the LLM
        """
        system_prompt = """You are explaining the behavior of a neuron in a neural network. Your response should be a very concise explanation (1-6 words) that captures what the neuron detects or predicts by finding patterns in lists.\n\n
To determine the explanation, you are given four lists:\n\n
- MAX_ACTIVATING_TOKENS, which are the top activating tokens in the top activating texts.\n
- TOKENS_AFTER_MAX_ACTIVATING_TOKEN, which are the tokens immediately after the max activating token.\n
- TOP_POSITIVE_LOGITS, which are the most likely words or tokens associated with this neuron.\n
- TOP_ACTIVATING_TEXTS, which are top activating texts.\n\n
You should look for a pattern by trying the following methods in order. Once you find a pattern, stop and return that pattern. Do not proceed to the later methods.\n
Method 1: Look at MAX_ACTIVATING_TOKENS. If they share something specific in common, or are all the same token or a variation of the same token (like different cases or conjugations), respond with that token.\n
Method 2: Look at TOKENS_AFTER_MAX_ACTIVATING_TOKEN. Try to find a specific pattern or similarity in all the tokens. A common pattern is that they all start with the same letter. If you find a pattern (like \'s word\', \'the ending -ing\', \'number 8\'), respond with \'say [the pattern]\'. You can ignore uppercase/lowercase differences for this.\n
Method 3: Look at TOP_POSITIVE_LOGITS for similarities and describe it very briefly (1-3 words).\n
Method 4: Look at TOP_ACTIVATING_TEXTS and make a best guess by describing the broad theme or context, ignoring the max activating tokens.\n\n
Rules:\n
- Keep your explanation extremely concise (1-6 words, mostly 1-3 words).\n
- Do not add unnecessary phrases like "words related to", "concepts related to", or "variations of the word".\n
- Do not mention "tokens" or "patterns" in your explanation.\n
- The explanation should be specific. For example, "unique words" is not a specific enough pattern, nor is "foreign words".\n
- Remember to use the \'say [the pattern]\' when using Method 2 above (pattern found in TOKENS_AFTER_MAX_ACTIVATING_TOKEN).\n
- If you absolutely cannot make any guesses, return the first token in MAX_ACTIVATING_TOKENS.\n\n
Respond by going through each method number until you find one that helps you find an explanation for what this neuron is detecting or predicting. If a method does not help you find an explanation, briefly explain why it does not, then go on to the next method. Finally, end your response with the method number you used, the reason for your explanation, and then the explanation.\n

Exsample:
{
<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\nwas\nwatching\n\n</TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\n\n<MAX_ACTIVATING_TOKENS>\n\nShe\nenjoy\n\n</MAX_ACTIVATING_TOKENS>\n\n\n<TOP_POSITIVE_LOGITS>\n\nwalking\nWA\nwaiting\nwas\nwe\nWHAM\nwish\nwin\nwake\nwhisper\n\n</TOP_POSITIVE_LOGITS>\n\n\n<TOP_ACTIVATING_TEXTS>\n\nShe was taking a nap when her phone started ringing.\nI enjoy watching movies with my family.\n\n</TOP_ACTIVATING_TEXTS>\n\n\nExplanation of neuron behavior: \n
Method 1 fails: MAX_ACTIVATING_TOKENS (She, enjoy) are not similar tokens.\nMethod 2 succeeds: All TOKENS_AFTER_MAX_ACTIVATING_TOKEN have a pattern in common: they all start with "w".\nExplanation: say "w" words
}

{
<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\nwarm\nthe\n\n</TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\n\n<MAX_ACTIVATING_TOKENS>\n\nand\nAnd\n\n</MAX_ACTIVATING_TOKENS>\n\n\n<TOP_POSITIVE_LOGITS>\n\nelephant\nguitar\nmountain\nbicycle\nocean\ntelescope\ncandle\numbrella\ntornado\nbutterfly\n\n</TOP_POSITIVE_LOGITS>\n\n\n<TOP_ACTIVATING_TEXTS>\n\nIt was a beautiful day outside with clear skies and warm sunshine.\nAnd the garden has roses and tulips and daisies and sunflowers blooming together.\n\n</TOP_ACTIVATING_TEXTS>\n\n\nExplanation of neuron behavior: \n
Method 1 succeeds: All MAX_ACTIVATING_TOKENS are the word "and".\nExplanation: and
}

{
<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\nare\n,\n\n</TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\n\n<MAX_ACTIVATING_TOKENS>\n\nbanana\nblueberries\n\n</MAX_ACTIVATING_TOKENS>\n\n\n<TOP_POSITIVE_LOGITS>\n\napple\norange\npineapple\nwatermelon\nkiwi\npeach\npear\ngrape\ncherry\nplum\n\n</TOP_POSITIVE_LOGITS>\n\n\n<TOP_ACTIVATING_TEXTS>\n\nThe apple and banana are delicious foods that provide essential vitamins and nutrients.\nI enjoy eating fresh strawberries, blueberries, and mangoes during the summer months.\n\n</TOP_ACTIVATING_TEXTS>\n\n\nExplanation of neuron behavior: \n
Method 1 succeeds: All MAX_ACTIVATING_TOKENS (banana, blueberries) are fruits.\nExplanation: fruits\n
}

{
<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\nwas\nplaces\n\n</TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\n\n<MAX_ACTIVATING_TOKENS>\n\nwar\nsome\n\n</MAX_ACTIVATING_TOKENS>\n\n\n<TOP_POSITIVE_LOGITS>\n\n4\nfour\nfourth\n4th\nIV\nFour\nFOUR\n~4\n4.0\nquartet\n\n</TOP_POSITIVE_LOGITS>\n\n\n<TOP_ACTIVATING_TEXTS>\n\nthe civil war was a major topic in history class .\n seasons of the year are winter , spring , summer , and fall or autumn in some places .\n\n</TOP_ACTIVATING_TEXTS>\n\n\nExplanation of neuron behavior: \n
Method 1 fails: MAX_ACTIVATING_TOKENS (war, some) are not all the same token.\nMethod 2 fails: TOKENS_AFTER_MAX_ACTIVATING_TOKEN (was, places) are not all similar tokens and don't have a text pattern in common.\nMethod 3 succeeds: All TOP_POSITIVE_LOGITS are the number 4.\nExplanation: 4\n
}
"""
        examples_to_show = activating_examples[: self.cfg.n_activating_examples]
        next_activating_tokens = ""
        max_activating_tokens = ""
        plain_activating_tokens = ""
        logit_activating_tokens = ""
        
        for i, example in enumerate(examples_to_show, 1):
            next_activating_tokens  = next_activating_tokens + example.display_next(self.cfg.activation_threshold)
            max_activating_tokens = max_activating_tokens + example.display_max(self.cfg.activation_threshold)
            plain_activating_tokens = plain_activating_tokens + example.display_plain()+"\n"
        
        if self.logits is not None:
            for text in self.logits['top_positive']:
                logit_activating_tokens = logit_activating_tokens + text['token']+"\n"
        else:
            logit_activating_tokens = next_activating_tokens
            
        user_prompt:str = f"""
<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\n{next_activating_tokens}\n</TOKENS_AFTER_MAX_ACTIVATING_TOKEN>\n\n\n<MAX_ACTIVATING_TOKENS>\n\n{max_activating_tokens}\n</MAX_ACTIVATING_TOKENS>\n\n\n<TOP_POSITIVE_LOGITS>\n\n{logit_activating_tokens}\n<\TOP_POSITIVE_LOGITS>\n\n\n<TOP_ACTIVATING_TEXTS>\n\n{plain_activating_tokens}\n<\TOP_ACTIVATING_TEXTS>\n\n\nExplanation of neuron behavior: \n
"""
        return system_prompt, user_prompt
        
    
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

{
    "steps": ["Activating token: <<knows>>. Contextual tokens: Who, ?. Pattern: <<knows>> is consistently activated, often found in sentences starting with interrogative words like 'Who' and ending with a question mark.", "Shared features include consistent activation on the word 'knows'. The surrounding text always forms a question. The questions do not seem to expect a literal answer, suggesting they are rhetorical.", "This feature activates on the word knows in rhetorical questions"],
    "final_explanation": "The feature activates on the word 'knows' in rhetorical questions.",
    "activation_consistency": 5,
    "complexity": 4
}

{
    "steps": ["Activating tokens: <<Entwickler>>, <<Enterprise>>, <<Entertainment>>, <<Entity>>, <<Entrance>>. Pattern: All activating instances are on words that begin with the specific substring 'Ent'. The activation is on the 'Ent' portion itself.", "The shared feature across all examples is the presence of words starting with the capitalized substring 'Ent'. The feature appears to be case-sensitive and position-specific (start of the word). No other contextual or semantic patterns are observed."],
    "final_explanation": "The feature activates on the substring 'Ent' at the start of words",
    "activation_consistency": 5,
    "complexity": 1
}

{
    "steps": ["Activating tokens: <<budget deficit>>, <<interest rates>>, <<fiscal stimulus>>, <<trade policy>>, <<unemployment benefits>>. Pattern: Activations highlight phrases and concepts central to economic discussions and government actions.","The examples consistently involve discussions of economic indicators, government spending, financial regulation, or international trade agreements. While most activations clearly relate to economic policies enacted or debated by governmental bodies, some activations might be on broader economic news or expert commentary where the direct link to a specific government policy is less explicit, or on related but not identical topics like corporate financial health in response to policy."],
    "final_explanation": "The feature activates on text about government economic policy",
    "activation_consistency": 3,
    "complexity": 5
}

"""
        system_prompt: str = f"""We're studying features in a neural network. Each feature activates on some particular word/words/substring/concept in a short document. The activating words in each document are indicated with << ... >>. We will give you a list of documents on which the feature activates, in order from most strongly activating to least strongly activating.

Your task is to:

First, Summarize the Activation: Look at the parts of the document the feature activates for and summarize in a single sentence what the feature is activating on. Try not to be overly specific in your explanation. Note that some features will activate only on specific words or substrings, but others will activate on most/all words in a sentence provided that sentence contains some particular concept. Your explanation should cover most or all activating words (for example, don't give an explanation which is specific to a single word if all words in a sentence cause the feature to activate). Pay attention to things like the capitalization and punctuation of the activating words or concepts, if that seems relevant. Keep the explanation as short and simple as possible, limited to 20 words or less. Omit punctuation and formatting. You should avoid giving long lists of words.{cot_prompt}

Second, Assess Activation Consistency: Based on your summary and the provided examples, evaluate the consistency of the feature's activation. Return your assessment as a single integer from the following scale:

5: Clear pattern with no deviating examples
4: Clear pattern with one or two deviating examples
3: Clear overall pattern but quite a few examples not fitting that pattern
2: Broad consistent theme but lacking structure
1: No discernible pattern

Third, Assess Feature Complexity: Based on your summary and the nature of the activation, evaluate the complexity of the feature. Return your assessment as a single integer from the following scale:

5: Rich feature firing on diverse contexts with an interesting unifying theme, e.g., "feelings of togetherness"
4: Feature relating to high-level semantic structure, e.g., "return statements in code"
3: Moderate complexity, such as a phrase, category, or tracking sentence structure, e.g., "website URLs"
2: Single word or token feature but including multiple languages or spelling, e.g., "mentions of dog"
1: Single token feature, e.g., "the token '('"

Your output should be a JSON object that has the following fields: `steps`, `final_explanation`, `activation_consistency`, `complexity`. `steps` should be an array of strings with a length not exceeding 3, each representing a step in the chain-of-thought process. `final_explanation` should be a string in the form of 'This feature activates on... '. `activation_consistency` should be an integer between 1 and 5, representing the consistency of the feature. `complexity` should be an integer between 1 and 5, representing the complexity of the feature.

{examples_prompt}
"""

        user_prompt = "The activating documents are given below:\n\n"
        # Select a subset of examples to show
        examples_to_show = activating_examples[: self.cfg.n_activating_examples]

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
        if self.cfg.explainer_type is ExplainerType.OPENAI:
            system_prompt, user_prompt = self._generate_explanation_prompt(activating_examples)
        else:
            system_prompt, user_prompt = self._generate_explanation_prompt_neuronpedia(activating_examples)
        start_time = time.time()
        # print(f'{system_prompt=}')
        print(f'{user_prompt=}')
        
        if self.cfg.explainer_type is ExplainerType.OPENAI:
            response = self.explainer_client.chat.completions.create(
                model=self.cfg.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            assert response.choices[0].message.content is not None, (
                f"No explanation returned from OpenAI\n\nsystem_prompt: {system_prompt}\n\nuser_prompt: {user_prompt}\n\nresponse: {response}"
            )
            explanation = json_repair.loads(response.choices[0].message.content)
        else:
            response = self.explainer_client.chat.completions.create(
                model=self.cfg.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            # assert response.choices[0].message.content is not None, (
            #     f"No explanation returned from OpenAI\n\nsystem_prompt: {system_prompt}\n\nuser_prompt: {user_prompt}\n\nresponse: {response}"
            # )
            # explanation = json_repair.loads(response.choices[0].message.content)
            def extract_explanation(s:str):
                keyword = "Explanation: "
                start_index = s.find(keyword)
                if start_index == -1:
                    return None
                else:
                    return s[start_index + len(keyword):]
            explanation = {
                "final_explanation" : extract_explanation(response.choices[0].message.content),
                "activation_consistency": 5,
                "complexity": 5,
            }
        response_time = time.time() - start_time
        return {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "response": explanation,
            "time": response_time,
        }

    def _generate_detection_prompt(
        self, explanation: dict[str, Any], examples: list[TokenizedSample]
    ) -> tuple[str, str]:
        """Generate a prompt for detection evaluation.

        Args:
            explanation: The explanation to evaluate
            examples: List of examples (mix of activating and non-activating)

        Returns:
            Prompt string for the LLM
        """
        system_prompt = f"""We're studying features in a neural network. Each feature activates on some particular word/words/substring/concept in a short document. You will be given a short explanation of what this feature activates for, and then be shown {len(examples)} example sequences in random order. You will have to return a boolean list of the examples where you think the feature should activate at least once, on ANY of the words or substrings in the document, true if it does, false if it doesn't. Try not to be overly specific in your interpretation of the explanation."""
        system_prompt += """
Your output should be a JSON object that has the following fields: `steps`, `evaluation_results`. `steps` should be an array of strings, each representing a step in the chain-of-thought process within 50 words. `evaluation_results` should be an array of booleans, each representing whether the feature should activate on the corresponding example.
"""
        user_prompt = f"Here is the explanation:\n\n{explanation['final_explanation']}\n\nHere are the examples:\n\n"

        for i, example in enumerate(examples, 1):
            user_prompt += f"Example {i}: {example.display_plain()}\n"

        return system_prompt, user_prompt

    def evaluate_explanation_detection(
        self,
        explanation: dict[str, Any],
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
                "time": 0,
            }

        random.shuffle(all_examples)

        # Ground truth for each example (1 for activating, 0 for non-activating)
        ground_truth = [1 if ex in test_activating else 0 for ex in all_examples]

        # Generate prompt
        system_prompt, user_prompt = self._generate_detection_prompt(explanation, all_examples)

        # Get response from OpenAI
        start_time = time.time()
        response = self.explainer_client.chat.completions.create(
            model=self.cfg.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        assert response.choices[0].message.content is not None, (
            f"No detection response returned from OpenAI\n\nsystem_prompt: {system_prompt}\n\nuser_prompt: {user_prompt}\n\nresponse: {response}"
        )
        detection_response: dict[str, Any] = json_repair.loads(response.choices[0].message.content)  # type: ignore
        # print(f"Detection for feature :\n{detection_response}\n\n")
        predictions: list[bool] = detection_response["evaluation_results"]
        response_time = time.time() - start_time

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
            "time": response_time,
        }

    def _generate_fuzzing_prompt(
        self,
        explanation: dict[str, Any],
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
        system_prompt += """
Your output should be a JSON object that has the following fields: `steps`, `evaluation_results`. `steps` should be an array of strings, each representing a step in the chain-of-thought process within 50 words. `evaluation_results` should be an array of booleans, each representing whether the feature should activate on the corresponding example.
"""
        user_prompt = f"Here is the explanation:\n\n{explanation['final_explanation']}\n\nHere are the examples:\n\n"

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
        n_highlighted = sum(1 for seg in sample.segments if seg.activation > threshold * sample.max_activation)

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
        self, explanation: dict[str, Any], activating_examples: list[TokenizedSample]
    ) -> dict[str, Any]:
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
                'time': 0,
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
        start_time = time.time()
        response = self.explainer_client.chat.completions.create(
            model=self.cfg.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        assert response.choices[0].message.content is not None, (
            f"No fuzzing response returned from OpenAI\n\nsystem_prompt: {system_prompt}\n\nuser_prompt: {user_prompt}\n\nresponse: {response}"
        )
        fuzzing_response: dict[str, Any] = json_repair.loads(response.choices[0].message.content)  # type: ignore
        # print(f"Fuzzing for feature :\n{fuzzing_response}\n\n")
        # Parse response (CORRECT/INCORRECT for each example)
        predictions: list[bool] = fuzzing_response["evaluation_results"]
        response_time = time.time() - start_time
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
            "time": response_time,
        }

    def interpret_single_feature(
        self,
        feature: FeatureRecord,
        model: LanguageModel,
        datasets: Callable[[str, int, int], Dataset],
        analysis_name: str = "default",
    ) -> dict[str, Any]:
        """Generate and evaluate explanations for multiple features.

        Args:
            feature: Feature to interpret
            model: Language model to use for generating activations
            datasets: Dataset to sample non-activating examples from
            analysis_name: Name of the analysis to use

        Returns:
            Dictionary mapping feature indices to their interpretation results
        """

        start_time = time.time()
        response_time = 0

        # if self.cfg.explainer_type is ExplainerType.NEURONPEDIA:
        self.logits = feature.logits
        
        activating_examples, non_activating_examples = self.get_feature_examples(
            feature=feature,
            model=model,
            datasets=datasets,
            analysis_name=analysis_name,
            max_length=self.cfg.max_length,
        )

        # print(f'{len(activating_examples)=} {len(non_activating_examples)=}')
        
        # Generate explanation for the feature
        explanation_result = self.generate_explanation(activating_examples)
        explanation: dict[str, Any] = explanation_result["response"]
        response_time += explanation_result["time"]
        # print(f"Explanation for feature {feature.index}:\n{explanation}\n\n")
        # Evaluate explanation
        evaluation_results = []

        if ScorerType.DETECTION in self.cfg.scorer_type:
            detection_result = self.evaluate_explanation_detection(
                explanation, activating_examples, non_activating_examples
            )
            # print(f"Detection result for feature {feature.index}:\n{detection_result}\n\n")
            evaluation_results.append(detection_result)
            # print(detection_result)
            response_time += detection_result["time"]

        if ScorerType.FUZZING in self.cfg.scorer_type:
            fuzzing_result = self.evaluate_explanation_fuzzing(explanation, activating_examples)
            # print(f"Fuzzing result for feature {feature.index}:\n{fuzzing_result}\n\n")
            evaluation_results.append(fuzzing_result)
            response_time += fuzzing_result["time"]

        total_time = time.time() - start_time

        return {
            "analysis_name": analysis_name,
            "explanation": explanation["final_explanation"],
            "complexity": explanation["complexity"],
            "consistency": explanation["activation_consistency"],
            "explanation_details": {
                k: v.model_dump() if isinstance(v, BaseModel) else v for k, v in explanation_result.items()
            },
            "evaluations": [
                {k: v.model_dump() if isinstance(v, BaseModel) else v for k, v in eval_result.items()}
                for eval_result in evaluation_results
            ],
            "passed": any(eval_result["passed"] for eval_result in evaluation_results),
            "time": {
                "total": total_time,
                "response": response_time,
            },
        }
    
    def interpret_features(
        self,
        sae_name: str,
        sae_series: str,
        feature_indices: list[int],
        model: LanguageModel,
        datasets: Callable[[str, int, int], Dataset],
        analysis_name: str = "default",
    ) -> Generator[dict[str, Any], None, None]:
        """Generate and evaluate explanations for multiple features.

        Args:
            sae_name: Name of the SAE
            sae_series: Series of the SAE
            feature_indices: Indices of the features to interpret
            model: Language model to use for generating activations
            datasets: Dataset to sample non-activating examples from
            analysis_name: Name of the analysis to use

        Returns:
            Dictionary mapping feature indices to their interpretation results
        """

        for feature_index in feature_indices:
            feature = self.mongo_client.get_feature(sae_name, sae_series, feature_index)
            if feature is not None and feature.interpretation is None:
                yield {
                    "feature_index": feature.index,
                    "sae_name": sae_name,
                    "sae_series": sae_series,
                } | self.interpret_single_feature(feature, model, datasets, analysis_name)