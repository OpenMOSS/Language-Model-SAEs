import uvicorn
from transformers import GPT2Tokenizer

from datasets import Dataset

import msgpack

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=1000)

tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

feature_activation_cache = {}

def get_feature_activation(analysis_name: str):
    if analysis_name not in feature_activation_cache:
        feature_activation_cache[analysis_name] = Dataset.load_from_disk(f"analysis/{analysis_name}")
    return feature_activation_cache[analysis_name]

@app.get("/features/{feature_index}")
def feature_info(feature_index: int):
    feature_activation = get_feature_activation("test")[feature_index]
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


if __name__ == "__main__":
    uvicorn.run("app:app", port=5432, log_level="info")