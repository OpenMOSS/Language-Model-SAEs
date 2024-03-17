import torch
import uvicorn
from transformer_lens import HookedTransformer

from fastapi import FastAPI
from datasets import Dataset

app = FastAPI()

model = HookedTransformer.from_pretrained('gpt2', device="cuda")
model.eval()

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
            "context": model.to_str_tokens(torch.tensor(feature_activation["contexts"][i], device="cuda")),
            "feature_acts": feature_activation["feature_acts"][i],
        }
        for i in range(n_samples)
    ]
    return {
        "feature_index": feature_index,
        "act_times": feature_activation["act_times"],
        "max_feature_act": feature_activation["max_feature_acts"],
        "samples": samples,
    }

if __name__ == "__main__":
    uvicorn.run("app:app", port=5432, log_level="info")