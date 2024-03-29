from typing import Dict, Optional
import torch
import os
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import tiktoken
import random
import traceback
from openai import OpenAI
from core.config import AutoInterpConfig, SAEConfig
from transformer_lens import HookedTransformer
from core.sae import SparseAutoEncoder


def _num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def _calculate_cost(num_input_tokens: int, num_output_tokens: int):
    return num_input_tokens * 0.01 / 1000 + num_output_tokens * 0.03 / 1000


def _extract_context(cfg: AutoInterpConfig, tokenizer, context_id, feature_acts):
    """
    Extracts a context around the token with the maximum activation value from the feature activations.

    This function identifies the position of the token with the highest activation value in `feature_acts`.
    It then selects up to `cfg.num_left_token` tokens to the left and `cfg.num_right_token` tokens to the right
    of this max-activation token to form a context window. If any tokens are split or corrupted (indicated by
    the presence of "�" in the decoded token), they are concatenated and their activation values are averaged
    before adding to the result list.
    """
    max_pos = torch.argmax(feature_acts, dim=0)
    left = max(max_pos - cfg.num_left_token, 0)
    right = min(max_pos + cfg.num_right_token, len(context_id))
    res = []
    temp_token_id = []
    temp_avg_value = 0
    for i in range(left, right):
        token = tokenizer.decode(context_id[i])
        value = feature_acts[i]
        if "�" in token:
            temp_token_id.append(context_id[i])
            temp_avg_value += value
        else:
            if len(temp_token_id) > 0:
                concat_token = tokenizer.decode(temp_token_id)
                avg_value = temp_avg_value / len(temp_token_id)
                res.append((concat_token, avg_value))
                temp_token_id = []
                temp_avg_value = 0

            res.append((token, round(float(value), 1)))
    return res


def _construct_prompt(context_list):
    task_description = """
We are analyzing the activation levels of features in a neural network, where each feature activates certain tokens in a text. Each token's activation value indicates its relevance to the feature, with higher values showing stronger association. Your task is to infer the common characteristic that these tokens collectively suggest based on their activation values.

Consider the following activations for a feature in the neural network. Activation values are non-negative, with higher values indicating a stronger connection between the token and the feature. Summarize in a single sentence what characteristic the feature is identifying in the text. Don't list examples of words.

"""
    # The activation format is token<tab>activation.
    for id, context in enumerate(context_list):
        token_acts_str = "\n".join([f"{token}\t{value}" for token, value in context])
        sentence_prompt = f"Sentence {id + 1}: \n<START>\n{token_acts_str}\n<END>\n\n"
        task_description += sentence_prompt

    return task_description


def _sample_sentences(cfg: AutoInterpConfig, tokenizer, feature_activation):
    """
    Samples sentences from a feature where the maximum activation value of tokens within a sentence
    is greater than a certain percentage (p) of the highest activation value across all tokens.
    """
    context_list = []
    max_acts = max(feature_activation["feature_acts"][0])
    filtered_sentences_id = [
        i
        for i in range(len(feature_activation["feature_acts"]))
        if max(feature_activation["feature_acts"][i]) > cfg.p * max_acts
    ]
    if len(filtered_sentences_id) < cfg.num_sample:
        sentences_id = list(range(min(len(feature_activation["feature_acts"]), cfg.num_sample)))
    else:
        sentences_id = random.sample(filtered_sentences_id, cfg.num_sample)
    for i in sentences_id:
        contexts = torch.tensor(feature_activation["contexts"][i])
        feature_acts = torch.tensor(feature_activation["feature_acts"][i])
        context_list.append(_extract_context(cfg, tokenizer, contexts, feature_acts))
    prompt = _construct_prompt(context_list)
    return prompt


def _chat_completion(client, prompt, max_retry=3):
    for i in range(max_retry):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            if i == max_retry - 1:
                return f"ERROR: {e}"


def generate_description(
    model: HookedTransformer,
    feature_activation: Dict,
    cfg: AutoInterpConfig,
):
    tokenizer = model.tokenizer
    client = OpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai_base_url)
    prompt = _sample_sentences(
        cfg, tokenizer, feature_activation
    )
    input_tokens = _num_tokens_from_string(prompt)
    response = _chat_completion(client, prompt)
    output_tokens = _num_tokens_from_string(response)
    cost = _calculate_cost(input_tokens, output_tokens)
    result = {
        "prompt": prompt,
        "response": response,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
    }
    return result



def check_description(
    model: HookedTransformer,
    cfg: AutoInterpConfig,
    index: int,
    description: str,
    using_sae: bool = False,
    feature_activation: Optional[Dict] = None,
    sae: Optional[SparseAutoEncoder] = None,
):
    """
    If `using_sae` is set to true, an SAE model must be provided.
    Otherwise, a `feature_activations` dataset is required for further processing.
    """
    tokenizer = model.tokenizer
    client = OpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai_base_url)
    if using_sae:
        assert sae is not None, "Sparse Auto Encoder is not provided."
        prompt_prefix = "We are analyzing the activation levels of features in a neural network, where each feature activates certain tokens in a text. Each token's activation value indicates its relevance to the feature, with higher values showing stronger association. We  will describe a feature's meaning and traits. Your output must be multiple sentences that activates the feature."
        prompt = prompt_prefix + f"\nFeature:{description}\nSentence:"
        input_tokens = _num_tokens_from_string(prompt)
        response = _chat_completion(client, prompt)
        output_tokens = _num_tokens_from_string(response)
        cost = _calculate_cost(input_tokens, output_tokens)
        input_index, input_text = index, response
        input_token = model.to_tokens(input_text)
        _, cache = model.run_with_cache(input_token, names_filter=[cfg.hook_point])
        activation = cache[cfg.hook_point][0]
        _, (_, aux) = sae(activation)
        feature_acts = aux["feature_acts"][:, input_index]
        max_value, max_pos = torch.max(feature_acts, dim=0)
        passed = torch.max(feature_acts) > 1
        result = {
            "index": input_index,
            "prompt": prompt,
            "response": input_text,
            "passed": passed.item(),
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "activation_token": model.tokenizer.decode(input_token[0][max_pos]),
            "max_value": max_value.item(),
        }
        return result

    else:
        assert feature_activation is not None, "Feature activations are not provided."
        prompt_prefix = "We are analyzing the activation levels of features in a neural network, where each feature activates certain tokens in a text. Each token's activation value indicates its relevance to the feature, with higher values showing stronger association. We  will describe a feature's meaning and traits. Identify the token that most activates the feature in the given sentence and provide your answer with a single token. The sentence will use a <tab> to separate each token.\n\nSentence:\n"
        
        context = _extract_context(
            cfg=cfg,
            tokenizer=tokenizer,
            context_id=torch.tensor(feature_activation["contexts"][0]),
            feature_acts=torch.tensor(feature_activation["feature_acts"][0]),
        )
        target_token = sorted(context, key=lambda x: x[1])[-1]
        prompt = prompt_prefix + "\t".join([token for token, _ in context])
        prompt += f"\nFeature:\n{description}\nToken:\n"
        input_tokens = _num_tokens_from_string(prompt)
        response = _chat_completion(client, prompt)
        output_tokens = _num_tokens_from_string(response)
        cost = _calculate_cost(input_tokens, output_tokens)
        passed = response.replace(" ", "") == target_token[0].replace(" ", "")
        result = {
            "index": index,
            "prompt": prompt,
            "response": response,
            "passed": passed,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "target_token": target_token[0],
        }
        return result
