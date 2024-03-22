import os

from transformers import GPT2Tokenizer

from datasets import Dataset

import msgpack

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

result_dir = os.environ.get("RESULT_DIR", "results")

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=1000)

tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

feature_activation_cache = {}

def get_feature_activation(dictionary_name: str):
    if dictionary_name not in feature_activation_cache:
        feature_activation_cache[dictionary_name] = Dataset.load_from_disk(os.path.join(result_dir, dictionary_name, "analysis", "top_activations"))
    return feature_activation_cache[dictionary_name]

@app.get("/dictionaries")
def list_dictionaries():
    dictionaries = os.listdir(result_dir)
    return [d for d in dictionaries if os.path.isdir(os.path.join(result_dir, d)) and os.path.exists(os.path.join(result_dir, d, "analysis"))]

@app.get("/dictionaries/{dictionary_name}/features/{feature_index}")
def feature_info(dictionary_name: str, feature_index: int):
    feature_activation = get_feature_activation(dictionary_name)[feature_index]
    n_samples = len(feature_activation["feature_acts"])
    samples = [
        {
            "context": [bytearray([tokenizer.byte_decoder[c] for c in t]) for t in tokenizer.convert_ids_to_tokens(feature_activation["contexts"][i])],
            "feature_acts": feature_activation["feature_acts"][i],
        }
        for i in range(n_samples)
    ]
    return Response(content=msgpack.packb({
        "feature_index": feature_index,
        "act_times": feature_activation["act_times"],
        "max_feature_act": feature_activation["max_feature_acts"],
        "samples": samples,
    }), media_type="application/x-msgpack")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)