from typing import Optional
import torch
import os
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import tiktoken
import random
import jsonlines
import traceback
from openai import OpenAI
from core.config import AutoInterpConfig, SAEConfig
from transformer_lens import HookedTransformer
from core.sae import SparseAutoEncoder


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def calculate_cost(num_input_tokens: int, num_output_tokens: int):
    return num_input_tokens * 0.01 / 1000 + num_output_tokens * 0.03 / 1000


def extract_context(cfg: AutoInterpConfig, tokenizer, context_id, feature_acts):
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


def construct_prompt(context_list):
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


def sample_sentences(tokenizer, feature, num_sentences, p):
    context_list = []
    max_acts = max(feature["feature_acts"][0])
    filtered_sentences_id = [
        i
        for i in range(len(feature["feature_acts"]))
        if max(feature["feature_acts"][i]) > p * max_acts
    ]
    if len(filtered_sentences_id) < num_sentences:
        sentences_id = list(range(min(len(feature["feature_acts"]), num_sentences)))
    # sample sentences from the filtered_activations
    else:
        # num_sentences = min(num_sentences, len(filtered_sentences_id))
        sentences_id = random.sample(filtered_sentences_id, num_sentences)
    for i in sentences_id:
        contexts = torch.tensor(feature["contexts"][i])
        feature_acts = torch.tensor(feature["feature_acts"][i])
        context_list.append(extract_context(tokenizer, contexts, feature_acts))
    prompt = construct_prompt(context_list)
    return prompt


def chat_completion(client, prompt, max_retry=3):
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
    config: AutoInterpConfig,
    index_list: list,
    output_file: str,
):
    tokenizer = model.tokenizer
    dataset_path = os.path.join(
        config.exp_result_dir, config.exp_name, "analysis/top_activations"
    )
    feature_activations = Dataset.load_from_disk(dataset_path)
    client = OpenAI(
        api_key=os.getenv("OPENAI_API"), base_url=os.getenv("OPENAI_BASE_URL")
    )
    try:
        with jsonlines.open(output_file) as reader:
            cur_feature = [item["index"] for item in reader]
            total_cost = sum([item["cost"] for item in reader])
    except FileNotFoundError:
        cur_feature = []
        total_cost = 0
    proc_bar = tqdm(total=len(index_list))

    for index in index_list:
        if index in cur_feature:
            proc_bar.update(1)
            continue
        feature = feature_activations[index]
        prompt = sample_sentences(
            tokenizer, feature, num_sentences=config.num_sample, p=config.p
        )
        input_tokens = num_tokens_from_string(prompt)
        response = chat_completion(client, prompt)
        output_tokens = num_tokens_from_string(response)
        cost = calculate_cost(input_tokens, output_tokens)
        total_cost += cost
        result = {
            "index": index,
            "prompt": prompt,
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
        }
        with jsonlines.open(output_file, "a") as writer:
            writer.write(result)
        proc_bar.set_description_str(f"Total cost: {total_cost}  Current cost: {cost}")
        proc_bar.update(1)


def check_description(
    model: HookedTransformer,
    config: AutoInterpConfig,
    description_file: str,
    output_file: str,
    using_sae: bool = False,
    sae: Optional[SparseAutoEncoder] = None,
):
    tokenizer = model.tokenizer
    client = OpenAI(
        api_key=os.getenv("OPENAI_API"), base_url=os.getenv("OPENAI_BASE_URL")
    )
    description = {}
    with jsonlines.open(description_file) as reader:
        for obj in reader:
            description[obj["index"]] = obj["response"]
    try:
        with jsonlines.open(output_file) as reader:
            cur_feature = [item["index"] for item in reader]
            total_cost = sum([item["cost"] for item in reader])
    except FileNotFoundError:
        cur_feature = []
        total_cost = 0

    if using_sae:
        assert sae is not None, "Sparse Auto Encoder is not provided."
        total_cost = 0
        prompt_prefix = "We are analyzing the activation levels of features in a neural network, where each feature activates certain tokens in a text. Each token's activation value indicates its relevance to the feature, with higher values showing stronger association. We  will describe a feature's meaning and traits. Your output must be multiple sentences that activates the feature."
        proc_bar = tqdm(total=len(description))
        for index, response in description.items():
            if index in cur_feature:
                proc_bar.update(1)
                continue
            prompt = prompt_prefix + f"\nFeature:{response}\nSentence:"
            input_tokens = num_tokens_from_string(prompt)
            response = chat_completion(client, prompt)
            output_tokens = num_tokens_from_string(response)
            cost = calculate_cost(input_tokens, output_tokens)
            total_cost += cost
            result = {
                "index": index,
                "prompt": prompt,
                "response": response,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
            }

            input_index, input_text = index, response
            input_token = model.to_tokens(input_text)
            _, cache = model.run_with_cache(
                input_token, names_filter=[config.hook_point]
            )
            activation = cache[config.hook_point][0]
            _, (_, aux) = sae(activation)
            # print(activation.shape, aux["feature_acts"].shape)
            feature_acts = aux["feature_acts"][:, input_index]
            max_value, max_pos = torch.max(feature_acts, dim=0)
            is_passed = "PASS" if torch.max(feature_acts) > 1 else "FAIL"
            result = {
                "index": input_index,
                "sentence": input_text,
                "is_passed": is_passed,
                "activation_token": model.tokenizer.decode(input_token[0][max_pos]),
                "max_value": max_value.item(),
            }
            proc_bar.set_description_str(
                f"Total cost: {total_cost}  Current cost: {cost}"
            )
            proc_bar.update(1)
            with jsonlines.open(output_file, mode="a") as writer:
                writer.write(result)

    else:
        dataset_path = os.path.join(
            config.exp_result_dir, config.exp_name, "analysis/top_activations"
        )
        feature_activations = Dataset.load_from_disk(dataset_path)
        prompt_prefix = "We are analyzing the activation levels of features in a neural network, where each feature activates certain tokens in a text. Each token's activation value indicates its relevance to the feature, with higher values showing stronger association. We  will describe a feature's meaning and traits. Identify the token that most activates the feature in the given sentence and provide your answer with a single token. The sentence will use a <tab> to separate each token.\n\nSentence:\n"
        proc_bar = tqdm(total=len(description))
        for index, response in description.items():
            if index in cur_feature:
                proc_bar.update(1)
                continue
            feature = feature_activations[index]
            context = extract_context(
                tokenizer=tokenizer,
                context_id=torch.tensor(feature["contexts"][0]),
                feature_acts=torch.tensor(feature["feature_acts"][0]),
            )
            target_token = sorted(context, key=lambda x: x[1])[-1]
            prompt = prompt_prefix + "\t".join([token for token, _ in context])
            prompt += f"\nFeature:\n{response}\nToken:\n"
            input_tokens = num_tokens_from_string(prompt)
            response = chat_completion(client, prompt)
            output_tokens = num_tokens_from_string(response)
            cost = calculate_cost(input_tokens, output_tokens)
            total_cost += cost
            check_result = (
                "PASS"
                if response.replace(" ", "") == target_token[0].replace(" ", "")
                else "FAIL"
            )
            result = {
                "index": index,
                "prompt": prompt,
                "response": response,
                "target_token": target_token[0],
                "check_result": check_result,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
            }
            proc_bar.set_description_str(
                f"Total cost: {total_cost}  Current cost: {cost}"
            )
            proc_bar.update(1)
            with jsonlines.open(output_file, mode="a") as writer:
                writer.write(result)
