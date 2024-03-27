import os
from typing import Dict

from functools import cmp_to_key

import numpy as np
import torch

from transformers import GPT2Tokenizer, AutoModelForCausalLM

from transformer_lens import HookedTransformer

from datasets import Dataset

import msgpack

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from core.config import SAEConfig
from core.sae import SparseAutoEncoder
import plotly.express as px

result_dir = os.environ.get("RESULT_DIR", "results")
device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=1000)

tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

hf_model = AutoModelForCausalLM.from_pretrained('gpt2')
model = HookedTransformer.from_pretrained('gpt2', device=device, hf_model=hf_model)
model.eval()

feature_activation_cache = {}
sae_cache = {}

def get_feature_activation(dictionary_name: str) -> Dict[str, Dataset]:
    analysis_names = os.listdir(os.path.join(result_dir, dictionary_name, "analysis"))
    if dictionary_name not in feature_activation_cache:
        feature_activation_cache[dictionary_name] = {}
    for analysis_name in analysis_names:
        if os.path.isdir(os.path.join(result_dir, dictionary_name, "analysis", analysis_name)) and os.path.exists(os.path.join(result_dir, dictionary_name, "analysis", analysis_name, "state.json")):
            if analysis_name not in feature_activation_cache[dictionary_name]:
                feature_activation_cache[dictionary_name][analysis_name] = Dataset.load_from_disk(os.path.join(result_dir, dictionary_name, "analysis", analysis_name))
    return feature_activation_cache[dictionary_name]

def get_sae(dictionary_name: str) -> SparseAutoEncoder:
    if dictionary_name not in sae_cache:
        cfg = SAEConfig(
            **SAEConfig.get_hyperparameters(dictionary_name, "results", "pruned.pt", True),
            
            # RunnerConfig
            use_ddp = False,
            device = device,
            seed = 42,
            dtype = torch.float32,

            exp_name = dictionary_name,
        )
        sae = SparseAutoEncoder(cfg=cfg)
        sae.load_state_dict(torch.load(cfg.from_pretrained_path, map_location=cfg.device)['sae'])
        sae.eval()
        sae_cache[dictionary_name] = sae
    return sae_cache[dictionary_name]

def make_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj

@app.get("/dictionaries")
def list_dictionaries():
    dictionaries = os.listdir(result_dir)
    return [d for d in dictionaries if os.path.isdir(os.path.join(result_dir, d)) and os.path.exists(os.path.join(result_dir, d, "analysis"))]

@app.post("/dictionaries/{dictionary_name}/features/load")
def load_feature_activations(dictionary_name: str):
    try:
        feature_activations = get_feature_activation(dictionary_name)
    except FileNotFoundError:
        return Response(content=f"Dictionary {dictionary_name} not found", status_code=404)
    
    if "top_activations" not in feature_activations:
        return Response(content=f"Dictionary {dictionary_name} does not have top feature activations", status_code=404)

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

    fig = px.histogram(feature_activation["feature_acts_all"], width=600, nbins=50)
    # fig.update_xaxes(title_text="Feature Activation Level")
    # fig.update_yaxes(title_text="Count")
    # fig.update_layout(showlegend=False)
    
    return Response(content=msgpack.packb({
        "feature_index": feature_index,
        "dictionary_name": dictionary_name,
        "feature_activation_histogram": make_serializable(fig.to_dict()['data']),
        "act_times": feature_activations["top_activations"]["act_times"][feature_index],
        "max_feature_act": feature_activations["top_activations"]["max_feature_acts"][feature_index],
        "sample_groups": sample_groups,
    }), media_type="application/x-msgpack")

@app.post("/dictionaries/{dictionary_name}/features/{feature_index}/custom")
def feature_activation_custom_input(dictionary_name: str, feature_index: int, input_text: str):
    try:
        sae = get_sae(dictionary_name)
        hook_point = open(os.path.join(result_dir, dictionary_name, "hook_point.txt")).read().strip()
    except FileNotFoundError:
        return Response(content=f"Dictionary {dictionary_name} not found", status_code=404)
    
    if feature_index < 0 or feature_index >= sae.cfg.d_sae:
        return Response(content=f"Feature index {feature_index} is out of range", status_code=400)
    
    with torch.no_grad():
        input = model.to_tokens(input_text, prepend_bos=False)
        _, cache = model.run_with_cache(input, names_filter=[hook_point])
        activation = cache[hook_point][0]

        _, (_, aux) = sae(activation)
        feature_acts = aux["feature_acts"]
        sample = {
            "context": [bytearray([tokenizer.byte_decoder[c] for c in t]) for t in tokenizer.convert_ids_to_tokens(input[0])],
            "feature_acts": feature_acts[:, feature_index].cpu().numpy().tolist(),
        }

    return Response(content=msgpack.packb(sample), media_type="application/x-msgpack")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)