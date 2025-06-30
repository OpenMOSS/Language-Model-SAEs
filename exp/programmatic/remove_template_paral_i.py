import argparse
import os
from multiprocessing import Pool, cpu_count

import torch
from safetensors.torch import load_file, save_file


def filter_and_pad_tensor_foro(tokens, activation):
    original_length = len(tokens)
    tokens_list = tokens.tolist()
    activation_list = activation.tolist()
    indices_to_remove = set()

    k = len(tokens) - 1
    pad_token_id = 128001
    if tokens[k] >= 128000:
        pad_token_id = tokens[k]
        while k >= 0 and tokens[k] == pad_token_id:
            k -= 1

    i = 0
    while i <= k:
        if tokens_list[i] >= 128000:
            indices_to_remove.add(i)
        i += 1

    filtered_tokens_list = [tokens_list[idx] for idx in range(len(tokens_list)) if idx not in indices_to_remove]
    filtered_activation_list = [activation_list[idx] for idx in range(len(tokens_list)) if idx not in indices_to_remove]

    num_to_pad = original_length - len(filtered_tokens_list)
    if num_to_pad > 0:
        filtered_tokens_list.extend([pad_token_id] * num_to_pad)
        filtered_activation_list.extend([filtered_activation_list[-1]] * num_to_pad)

    # when the context is longer than 8192, filterd o would have more valid tokens than base and instruct(it filtered less tokens)
    # so we have to hardcode some of the valid tokens to pad_token
    if filtered_tokens_list[8159] != 128001:
        filtered_tokens_list[8159:] = [128001] * 32  # TO CHECK

    return torch.tensor(filtered_tokens_list), torch.tensor(filtered_activation_list)


def filter_and_pad_tensor_fori(tokens, activation, otemplate, filename):
    original_length = len(tokens)
    tokens_list = tokens.tolist()
    activation_list = activation.tolist()
    o_list = otemplate.tolist()
    objlen = len(o_list)
    indices_to_remove = set()

    k = len(tokens) - 1
    pad_token_id = 128009
    if tokens[k] >= 128000:
        pad_token_id = tokens[k]
        while k >= 0 and tokens[k] == pad_token_id:
            k -= 1

    j = 0
    for i in range(k + 1):
        if j < objlen and tokens_list[i] == o_list[j]:
            j += 1
        else:
            indices_to_remove.add(i)

    filtered_tokens_list = [tokens_list[idx] for idx in range(len(tokens_list)) if idx not in indices_to_remove]
    filtered_activation_list = [activation_list[idx] for idx in range(len(tokens_list)) if idx not in indices_to_remove]

    num_to_pad = original_length - len(filtered_tokens_list)

    # 添加断言检查 tokens_list 和 activation_list 长度是否一致                ##################
    try:
        assert len(filtered_tokens_list) == len(filtered_activation_list), f"Length mismatch in file: {filename}"
    except AssertionError as e:
        print(f"AssertionError: {e}")
        raise

    # 添加断言检查 filtered_activation_list 是否为空                         ######################
    try:
        assert filtered_activation_list, f"Empty filtered_activation_list in file: {filename}"
    except AssertionError as e:
        print(f"AssertionError: {e}")
        raise

    if num_to_pad > 0:
        filtered_tokens_list.extend([pad_token_id] * num_to_pad)
        filtered_activation_list.extend([filtered_activation_list[-1]] * num_to_pad)

    # 添加断言检查 token 是否对齐                                  ######################
    try:
        assert filtered_tokens_list == [128009 if x == 128001 else x for x in o_list], f"tokens not aligned: {filename}"
    except AssertionError as e:
        print(f"AssertionError: {e}")
        raise

    return torch.tensor(filtered_tokens_list), torch.tensor(filtered_activation_list)


def filter_and_pad_tensor_forb(tokens, activation, otemplate):
    original_length = len(tokens)
    tokens_list = tokens.tolist()
    activation_list = activation.tolist()
    o_list = otemplate.tolist()
    objlen = len(o_list)
    indices_to_remove = set()

    k = len(tokens) - 1
    pad_token_id = 128001
    if tokens[k] >= 128000:
        pad_token_id = tokens[k]
        while k >= 0 and tokens[k] == pad_token_id:
            k -= 1

    j = 0
    for i in range(k + 1):
        if j < objlen and tokens_list[i] == o_list[j]:
            j += 1
        else:
            indices_to_remove.add(i)

    filtered_tokens_list = [tokens_list[idx] for idx in range(len(tokens_list)) if idx not in indices_to_remove]
    filtered_activation_list = [activation_list[idx] for idx in range(len(tokens_list)) if idx not in indices_to_remove]

    num_to_pad = original_length - len(filtered_tokens_list)
    if num_to_pad > 0:
        filtered_tokens_list.extend([pad_token_id] * num_to_pad)
        filtered_activation_list.extend([filtered_activation_list[-1]] * num_to_pad)

    return torch.tensor(filtered_tokens_list), torch.tensor(filtered_activation_list)


def process_file(args):
    input_folder, output_folder, template_folder, filename, model_name = args
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    template_path = os.path.join(template_folder, filename)

    if model_name == "o":
        filter_func = filter_and_pad_tensor_foro
    elif model_name == "i":
        filter_func = filter_and_pad_tensor_fori
    elif model_name == "b":
        filter_func = filter_and_pad_tensor_forb
    else:
        return f"there is no such model : {model_name}"

    if os.path.exists(input_path):
        tensor_data = load_file(input_path)
        template_data = load_file(template_path)

        tokens = tensor_data["tokens"][0]
        activation = tensor_data["activation"][0]
        template_tokens = template_data["tokens"][0]

        tensor_data["tokens"][0], tensor_data["activation"][0] = filter_func(
            tokens, activation, template_tokens, filename
        )  ####### ####################

        save_file(tensor_data, output_path)
        return f"Processed and saved: {output_path}"
    else:
        return f"File not found: {input_path}"


def main(
    input_folder="/inspire/hdd/global_user/hezhengfu-240208120186/jiaxing_activations/reasondata-i-2d-801M/blocks.15.hook_resid_post",
    output_folder="/inspire/hdd/global_user/hezhengfu-240208120186/jiaxing_activations/reasondata-i-2d-801M-f/blocks.15.hook_resid_post",
    template_folder="/inspire/hdd/global_user/hezhengfu-240208120186/jiaxing_activations/reasondata-o-2d-801M-f/blocks.15.hook_resid_post",
    shards=range(16),
):
    os.makedirs(output_folder, exist_ok=True)

    all_files = [f"shard-{shard}-chunk-{chunk:08d}.safetensors" for shard in shards for chunk in range(6112)]

    cpu_cores = min(cpu_count(), 64)
    with Pool(cpu_cores) as pool:
        results = pool.map(process_file, [(input_folder, output_folder, template_folder, f, "i") for f in all_files])

    for result in results:
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process safetensors data with filtering and padding.")
    parser.add_argument("--shards", type=int, nargs="+", required=True, help="List of shard numbers to process.")
    args = parser.parse_args()

    main(shards=args.shards)
