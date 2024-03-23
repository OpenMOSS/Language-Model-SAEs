import os
from typing import Dict

from functools import cmp_to_key

import torch

from transformers import GPT2Tokenizer

from datasets import Dataset

import msgpack

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

result_dir = os.environ.get("RESULT_DIR", "results")
device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=1000)

tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

feature_activation_cache = {}

def get_feature_activation(dictionary_name: str) -> Dict[str, Dataset]:
    analysis_names = os.listdir(os.path.join(result_dir, dictionary_name, "analysis"))
    if dictionary_name not in feature_activation_cache:
        feature_activation_cache[dictionary_name] = {}
    for analysis_name in analysis_names:
        if analysis_name not in feature_activation_cache[dictionary_name]:
            feature_activation_cache[dictionary_name][analysis_name] = Dataset.load_from_disk(os.path.join(result_dir, dictionary_name, "analysis", analysis_name))
    return feature_activation_cache[dictionary_name]

@app.get("/dictionaries")
def list_dictionaries():
    dictionaries = os.listdir(result_dir)
    return [d for d in dictionaries if os.path.isdir(os.path.join(result_dir, d)) and os.path.exists(os.path.join(result_dir, d, "analysis"))]

@app.get("/dictionaries/{dictionary_name}/features/{feature_index}")
def feature_info(dictionary_name: str, feature_index: str):
    try:
        feature_activations = get_feature_activation(dictionary_name)
    except FileNotFoundError:
        return Response(content=f"Dictionary {dictionary_name} not found", status_code=404)
    
    if "top_activations" not in feature_activations:
        return Response(content=f"Dictionary {dictionary_name} does not have top feature activations", status_code=404)
    
    if isinstance(feature_index, str):
        if feature_index == "random":
            nonzero_feature_indices = torch.tensor(feature_activations["top_activations"]["max_feature_acts"]).nonzero(as_tuple=True)[0]
            feature_index = nonzero_feature_indices[torch.randint(len(nonzero_feature_indices), (1,))].item()
        else:
            try:
                feature_index = int(feature_index)
            except ValueError:
                return Response(content=f"Feature index {feature_index} is not a valid integer", status_code=400)
        
    if feature_index < 0 or feature_index >= len(feature_activations["top_activations"]):
        return Response(content=f"Feature index {feature_index} is out of range", status_code=400)
    
    sample_groups = []
    for analysis_name, dataset in feature_activations.items():
        feature_activation = dataset[feature_index]
        samples = [
            {
                "context": [bytearray([tokenizer.byte_decoder[c] for c in t]) for t in tokenizer.convert_ids_to_tokens(feature_activation["contexts"][i])],
                "feature_acts": feature_activation["feature_acts"][i],
            }
            for i in range(len(feature_activation["feature_acts"]))
        ]
        sample_groups.append({
            "analysis_name": analysis_name,
            "samples": samples,
        })
    
    # Sort results so that top activations are first
    sample_groups.sort(key=cmp_to_key(lambda a, b: -1 if a["analysis_name"] == "top_activations" else 1 if b["analysis_name"] == "top_activations" else 0))
    
    return Response(content=msgpack.packb({
        "feature_index": feature_index,
        "act_times": feature_activations["top_activations"]["act_times"][feature_index],
        "max_feature_act": feature_activations["top_activations"]["max_feature_acts"][feature_index],
        "sample_groups": sample_groups,
    }), media_type="application/x-msgpack")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)