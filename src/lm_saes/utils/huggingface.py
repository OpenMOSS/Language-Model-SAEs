# Some helper function to interact with huggingface API

import os
import re

from huggingface_hub import snapshot_download

from .misc import print_once


def _parse_repo_id(pretrained_name_or_path):
    pattern = r"L(\d{1,2})([RAMTC])-(8|32)x"

    def replace_match(match):
        sublayer = match.group(2)
        exp_factor = match.group(3)

        return f"LX{sublayer}-{exp_factor}x"

    output_string = re.sub(pattern, replace_match, pretrained_name_or_path)

    return output_string


def parse_pretrained_name_or_path(pretrained_name_or_path: str):
    if os.path.exists(pretrained_name_or_path):
        return pretrained_name_or_path
    else:
        print_once(f"Local path `{pretrained_name_or_path}` not found. Downloading from huggingface model hub.")
        if pretrained_name_or_path.startswith("fnlp"):
            print_once("Downloading Llama Scope SAEs.")
            repo_id = _parse_repo_id(pretrained_name_or_path)
            hook_point = pretrained_name_or_path.split("/")[1]
        else:
            repo_id = "/".join(pretrained_name_or_path.split("/")[:2])
            hook_point = "/".join(pretrained_name_or_path.split("/")[2:])
        snapshot_path = snapshot_download(repo_id=repo_id, allow_patterns=[f"{hook_point}/*"])
        return os.path.join(snapshot_path, hook_point)
