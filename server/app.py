import io
import os

import msgpack
import numpy as np
import plotly.graph_objects as go
import torch
from datasets import Dataset
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from torchvision import transforms

from lm_saes.backend import LanguageModel
from lm_saes.config import MongoDBConfig, SAEConfig
from lm_saes.database import MongoClient
from lm_saes.resource_loaders import load_dataset_shard, load_model
from lm_saes.sae import SparseAutoEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=1000)

client = MongoClient(MongoDBConfig())
sae_series = os.environ.get("SAE_SERIES", "default")

sae_cache: dict[str, SparseAutoEncoder] = {}
lm_cache: dict[str, LanguageModel] = {}
dataset_cache: dict[tuple[str, int, int], Dataset] = {}


def get_model(name: str) -> LanguageModel:
    cfg = client.get_model_cfg(name)
    if cfg is None:
        raise ValueError(f"Model {name} not found")
    if name not in lm_cache:
        model = load_model(cfg)
        lm_cache[name] = model
    return lm_cache[name]


def get_dataset(name: str, shard_idx: int = 0, n_shards: int = 1) -> Dataset:
    cfg = client.get_dataset_cfg(name)
    assert cfg is not None, f"Dataset {name} not found"
    if (name, shard_idx, n_shards) not in dataset_cache:
        dataset_cache[name, shard_idx, n_shards] = load_dataset_shard(cfg, shard_idx, n_shards)
    return dataset_cache[name, shard_idx, n_shards]


def get_sae(name: str) -> SparseAutoEncoder:
    path = client.get_sae_path(name, sae_series)
    assert path is not None, f"SAE {name} not found"
    if name not in sae_cache:
        cfg = SAEConfig.from_pretrained(path)
        sae = SparseAutoEncoder.from_config(cfg)
        sae.eval()
        sae_cache[name] = sae
    return sae_cache[name]


def make_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


@app.exception_handler(AssertionError)
async def assertion_error_handler(request, exc):
    return Response(content=str(exc), status_code=400)


@app.exception_handler(torch.cuda.OutOfMemoryError)
async def oom_error_handler(request, exc):
    print("CUDA Out of memory. Clearing cache.")
    print("Current cache:", sae_cache.keys())
    # Clear cache
    sae_cache.clear()
    return Response(content="CUDA Out of memory", status_code=500)


@app.get("/dictionaries")
def list_dictionaries():
    return client.list_saes(sae_series=sae_series, has_analyses=True)


@app.get("/images/{dataset_name}")
def get_image(dataset_name: str, context_idx: int, image_idx: int, shard_idx: int = 0, n_shards: int = 1):
    dataset = get_dataset(dataset_name, shard_idx, n_shards)
    data = dataset[int(context_idx)]

    image_key = "image" if "image" in data else "images" if "images" in data else None
    if image_key is None:
        return Response(content="Image not found", status_code=404)

    if len(data[image_key]) <= image_idx:
        return Response(content="Image not found", status_code=404)

    image_tensor = data[image_key][image_idx]

    # Convert tensor to PIL Image and then to bytes
    image = transforms.ToPILImage()(image_tensor.to(torch.uint8))
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    return Response(content=img_byte_arr, media_type="image/png")


@app.get("/dictionaries/{name}/features/{feature_index}")
def get_feature(name: str, feature_index: str | int):
    if isinstance(feature_index, str) and feature_index != "random":
        try:
            feature_index = int(feature_index)
        except ValueError:
            return Response(
                content=f"Feature index {feature_index} is not a valid integer",
                status_code=400,
            )
    if feature_index == "random":
        feature = client.get_random_alive_feature(sae_name=name, sae_series=sae_series)
    else:
        feature = client.get_feature(sae_name=name, sae_series=sae_series, index=feature_index)

    if feature is None:
        return Response(
            content=f"Feature {feature_index} not found in SAE {name}",
            status_code=404,
        )

    sample_groups = []
    for sampling in feature.analyses[0].samplings:
        samples = []
        for i in range(len(sampling.feature_acts)):
            feature_acts = sampling.feature_acts[i]
            context_idx = sampling.context_idx[i]
            dataset_name = sampling.dataset_name[i]
            model_name = sampling.model_name[i]
            model = get_model(model_name)
            shard_idx = sampling.shard_idx[i] if sampling.shard_idx is not None else 0
            n_shards = sampling.n_shards[i] if sampling.n_shards is not None else 1

            data = get_dataset(dataset_name, shard_idx, n_shards)[context_idx]

            origins = model.trace({k: [v] for k, v in data.items()})[0]

            # Replace image_key with image_url
            image_key = "image" if "image" in data else "images" if "images" in data else None
            if image_key is not None:
                image_urls = [
                    f"/images/{dataset_name}?context_idx={context_idx}&shard_idx={shard_idx}&n_shards={n_shards}&image_idx={image_idx}"
                    for image_idx in range(len(data[image_key]))
                ]
                del data[image_key]
                data["images"] = image_urls

            origins = origins[: len(feature_acts)]
            feature_acts = feature_acts[: len(origins)]

            if "text" in data:
                text_ranges = [origin["range"] for origin in origins if origin is not None and origin["key"] == "text"]
                max_text_origin = max(text_ranges, key=lambda x: x[1])
                data["text"] = data["text"][: max_text_origin[1]]

            samples.append(
                {
                    **data,
                    "origins": origins,
                    "feature_acts": feature_acts,
                }
            )

        sample_groups.append(
            {
                "analysis_name": sampling.name,
                "samples": samples,
            }
        )

    return Response(
        content=msgpack.packb(
            make_serializable(
                {
                    "feature_index": feature.index,
                    "dictionary_name": feature.sae_name,
                    "act_times": feature.analyses[0].act_times,
                    "max_feature_act": feature.analyses[0].max_feature_acts,
                    "sample_groups": sample_groups,
                }
            )
        ),
        media_type="application/x-msgpack",
    )


@app.get("/dictionaries/{name}")
def get_dictionary(name: str):
    feature_activation_times = client.get_feature_act_times(name, sae_series=sae_series)
    if feature_activation_times is None:
        return Response(content=f"Dictionary {name} not found", status_code=404)
    log_act_times = np.log10(np.array(list(feature_activation_times.values())))
    feature_activation_times_histogram = go.Histogram(
        x=log_act_times,
        nbinsx=100,
        hovertemplate="Count: %{y}<br>Range: %{x}<extra></extra>",
        marker_color="#636EFA",
        showlegend=False,
    ).to_plotly_json()

    alive_feature_count = client.get_alive_feature_count(name, sae_series=sae_series)
    if alive_feature_count is None:
        return Response(content=f"SAE {name} not found", status_code=404)

    return Response(
        content=msgpack.packb(
            make_serializable(
                {
                    "dictionary_name": name,
                    "feature_activation_times_histogram": [feature_activation_times_histogram],
                    "alive_feature_count": alive_feature_count,
                }
            )
        ),
        media_type="application/x-msgpack",
    )


# @app.post("/dictionaries/{dictionary_name}/features/{feature_index}/custom")
# def feature_activation_custom_input(dictionary_name: str, feature_index: int, input_text: str):
#     try:
#         sae = get_sae(dictionary_name)
#     except FileNotFoundError:
#         return Response(content=f"Dictionary {dictionary_name} not found", status_code=404)

#     if feature_index < 0 or feature_index >= sae.cfg.d_sae:
#         return Response(content=f"Feature index {feature_index} is out of range", status_code=400)

#     model = get_model(dictionary_name)
#     with torch.no_grad():
#         input = model.to_tokens(input_text, prepend_bos=False)
#         _, cache = model.run_with_cache_until(
#             input,
#             names_filter=[sae.cfg.hook_point_in, sae.cfg.hook_point_out],
#             until=sae.cfg.hook_point_out,
#         )

#         feature_acts = sae.encode(cache[sae.cfg.hook_point_in][0])
#         sample = {
#             "context": [
#                 bytearray([byte_decoder[c] for c in t])
#                 # Method `convert_ids_to_tokens` should exist on GPT2Tokenizer and other BPE tokenizers.
#                 for t in model.tokenizer.convert_ids_to_tokens(input[0])  # type: ignore
#             ],
#             "feature_acts": feature_acts[:, feature_index].tolist(),
#         }

#     return Response(content=msgpack.packb(sample), media_type="application/x-msgpack")


# @app.post("/dictionaries/{dictionary_name}/custom")
# def dictionary_custom_input(dictionary_name: str, input_text: str):
#     try:
#         sae = get_sae(dictionary_name)
#     except FileNotFoundError:
#         return Response(content=f"Dictionary {dictionary_name} not found", status_code=404)

#     max_feature_acts = client.get_max_feature_acts(dictionary_name, dictionary_series=dictionary_series)
#     assert max_feature_acts is not None, "Max feature acts not found"

#     model = get_model(dictionary_name)

#     with torch.no_grad():
#         input = model.to_tokens(input_text, prepend_bos=False)
#         _, cache = model.run_with_cache_until(
#             input,
#             names_filter=[sae.cfg.hook_point_in, sae.cfg.hook_point_out],
#             until=sae.cfg.hook_point_out,
#         )

#         feature_acts = sae.encode(cache[sae.cfg.hook_point_in][0])
#         sample = {
#             "context": [
#                 bytearray([byte_decoder[c] for c in t])
#                 # Method `convert_ids_to_tokens` should exist on GPT2Tokenizer and other BPE tokenizers.
#                 for t in model.tokenizer.convert_ids_to_tokens(input[0])  # type: ignore
#             ],
#             "feature_acts_indices": [
#                 feature_acts[i].nonzero(as_tuple=True)[0].tolist() for i in range(feature_acts.shape[0])
#             ],
#             "feature_acts": [
#                 feature_acts[i][feature_acts[i].nonzero(as_tuple=True)[0]].tolist()
#                 for i in range(feature_acts.shape[0])
#             ],
#             "max_feature_acts": [
#                 [max_feature_acts[j] for j in feature_acts[i].nonzero(as_tuple=True)[0].tolist()]
#                 for i in range(feature_acts.shape[0])
#             ],
#         }

#     return Response(content=msgpack.packb(sample), media_type="application/x-msgpack")


# class SteeringConfig(BaseModel):
#     sae: str
#     feature_index: int
#     steering_type: Literal["times", "add", "set", "ablate"]
#     steering_value: float | None = None


# class FeatureNode(BaseModel):
#     type: Literal["feature"]
#     sae: str
#     feature_index: int
#     position: int


# class LogitsNode(BaseModel):
#     type: Literal["logits"]
#     position: int
#     token_id: int


# class AttnScoreNode(BaseModel):
#     type: Literal["attn-score"]
#     layer: int
#     head: int
#     query: int
#     key: int


# Node = Annotated[Union[FeatureNode, LogitsNode, AttnScoreNode], Field(discriminator="type")]


# class ModelGenerateRequest(BaseModel):
#     input_text: str | list[int]
#     max_new_tokens: int = 128
#     top_k: int = 50
#     top_p: float = 0.95
#     return_logits_top_k: int = 5
#     saes: list[str] = []
#     steerings: list[SteeringConfig] = []


# @app.post("/model/generate")
# def model_generate(request: ModelGenerateRequest):
#     dictionaries = client.list_dictionaries(dictionary_series=dictionary_series)
#     assert len(dictionaries) > 0, "No dictionaries found. Model name cannot be inferred."
#     model = get_model(dictionaries[0])
#     saes = [(get_sae(name), name) for name in request.saes]
#     max_feature_acts = {
#         name: client.get_max_feature_acts(name, dictionary_series=dictionary_series) for _, name in saes
#     }
#     assert all(
#         max_feature_acts is not None for max_feature_acts in max_feature_acts.values()
#     ), "Max feature acts not found"
#     max_feature_acts = cast(dict[str, dict[int, int]], max_feature_acts)
#     assert all(steering.sae in request.saes for steering in request.steerings), "Steering SAE not found"

#     def generate_steering_hook(steering: SteeringConfig):
#         feature_acts = None

#         def steer(tensor: torch.Tensor):
#             assert len(tensor.shape) == 3
#             tensor = tensor.clone()
#             if steering.steering_type == "times":
#                 assert steering.steering_value is not None
#                 tensor[:, :, steering.feature_index] *= steering.steering_value
#             elif steering.steering_type == "ablate":
#                 tensor[:, :, steering.feature_index] = 0
#             elif steering.steering_type == "add":
#                 assert steering.steering_value is not None
#                 tensor[:, :, steering.feature_index] += steering.steering_value
#             elif steering.steering_type == "set":
#                 assert steering.steering_value is not None
#                 tensor[:, :, steering.feature_index] = steering.steering_value
#             return tensor

#         def save_feature_acts_hook(tensor: torch.Tensor, hook: HookPoint):
#             nonlocal feature_acts
#             feature_acts = tensor
#             return steer(tensor)

#         def steering_hook(tensor: torch.Tensor, hook: HookPoint):
#             assert feature_acts is not None, "Feature acts should be saved before steering"
#             difference = (steer(feature_acts) - feature_acts) @ sae.decoder.weight.T
#             tensor += difference.detach()
#             return tensor

#         sae = get_sae(steering.sae)
#         return [
#             (f"{sae.cfg.hook_point_out}.sae.hook_feature_acts", save_feature_acts_hook),
#             (f"{sae.cfg.hook_point_out}", steering_hook),
#         ]

#     steering_hooks = sum([generate_steering_hook(steering) for steering in request.steerings], [])

#     with torch.no_grad():
#         with apply_sae(model, [sae for sae, _ in saes]):
#             with model.hooks(steering_hooks):
#                 input = (
#                     model.to_tokens(request.input_text, prepend_bos=False)
#                     if isinstance(request.input_text, str)
#                     else torch.tensor([request.input_text], device=device)
#                 )
#                 if request.max_new_tokens > 0:
#                     output = cast(
#                         torch.Tensor,
#                         model.generate(
#                             input,
#                             max_new_tokens=request.max_new_tokens,
#                             top_k=request.top_k,
#                             top_p=request.top_p,
#                         ),
#                     )
#                     input = output.clone()
#                 name_filter = (
#                     [f"{sae.cfg.hook_point_out}.sae.hook_feature_acts" for sae, _ in saes]
#                     + [f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.pre" for sae, _ in saes]
#                     + [f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.post" for sae, _ in saes]
#                 )
#                 logits, cache = model.run_with_ref_cache(input, names_filter=name_filter)
#                 logits_topk = [torch.topk(l, request.return_logits_top_k) for l in logits[0]]

#                 result = {
#                     "context": [
#                         bytearray([byte_decoder[c] for c in t])
#                         # Method `convert_ids_to_tokens` should exist on GPT2Tokenizer and other BPE tokenizers.
#                         for t in model.tokenizer.convert_ids_to_tokens(input[0])  # type: ignore
#                     ],
#                     "token_ids": input[0].tolist(),
#                     "logits": {
#                         "logits": [l.values.tolist() for l in logits_topk],
#                         "tokens": [
#                             [
#                                 bytearray([byte_decoder[c] for c in t])
#                                 # Method `convert_ids_to_tokens` should exist on GPT2Tokenizer and other BPE tokenizers.
#                                 for t in model.tokenizer.convert_ids_to_tokens(l.indices)  # type: ignore
#                             ]
#                             for l in logits_topk
#                         ],
#                         "token_ids": [l.indices.tolist() for l in logits_topk],
#                     },
#                     "input_mask": [1 for _ in range(len(input[0]))] + [0 for _ in range(len(input[0]) - len(input[0]))],
#                     "sae_info": [
#                         {
#                             "name": name,
#                             "feature_acts_indices": [
#                                 cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts"][0][i]
#                                 .nonzero(as_tuple=True)[0]
#                                 .tolist()
#                                 for i in range(cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts"][0].shape[0])
#                             ],
#                             "feature_acts": [
#                                 cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts"][0][i][
#                                     cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts"][0][i].nonzero(
#                                         as_tuple=True
#                                     )[0]
#                                 ].tolist()
#                                 for i in range(cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts"][0].shape[0])
#                             ],
#                             "max_feature_acts": [
#                                 [
#                                     max_feature_acts[name][j]
#                                     for j in cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts"][0][i]
#                                     .nonzero(as_tuple=True)[0]
#                                     .tolist()
#                                 ]
#                                 for i in range(cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts"][0].shape[0])
#                             ],
#                         }
#                         for sae, name in saes
#                     ],
#                 }
#     return Response(content=msgpack.packb(result), media_type="application/x-msgpack")


# class ModelTraceRequest(BaseModel):
#     input_text: str | list[int]
#     saes: list[str] = []
#     steerings: list[SteeringConfig] = []
#     tracings: list[Node] = []
#     tracing_threshold: float = 0.1
#     tracing_top_k: int | None = None
#     detach_at_attn_scores: bool = False


# @app.post("/model/trace")
# def model_trace(request: ModelTraceRequest):
#     dictionaries = client.list_dictionaries(dictionary_series=dictionary_series)
#     assert len(dictionaries) > 0, "No dictionaries found. Model name cannot be inferred."
#     model = get_model(dictionaries[0])
#     assert model.tokenizer is not None, "Tokenizer not found"
#     saes = [(get_sae(name), name) for name in request.saes]
#     max_feature_acts = {
#         name: client.get_max_feature_acts(name, dictionary_series=dictionary_series) for _, name in saes
#     }
#     assert all(
#         max_feature_acts is not None for max_feature_acts in max_feature_acts.values()
#     ), "Max feature acts not found"
#     max_feature_acts = cast(dict[str, dict[int, int]], max_feature_acts)
#     assert all(steering.sae in request.saes for steering in request.steerings), "Steering SAE not found"
#     assert all(
#         tracing.sae in request.saes for tracing in request.tracings if isinstance(tracing, FeatureNode)
#     ), "Tracing SAE not found"

#     def generate_steering_hook(steering: SteeringConfig):
#         feature_acts = None
#         sae = get_sae(steering.sae)

#         def steer(tensor: torch.Tensor):
#             assert len(tensor.shape) == 3
#             tensor = tensor.clone()
#             if steering.steering_type == "times":
#                 assert steering.steering_value is not None
#                 tensor[:, :, steering.feature_index] *= steering.steering_value
#             elif steering.steering_type == "ablate":
#                 tensor[:, :, steering.feature_index] = 0
#             elif steering.steering_type == "add":
#                 assert steering.steering_value is not None
#                 tensor[:, :, steering.feature_index] += steering.steering_value
#             elif steering.steering_type == "set":
#                 assert steering.steering_value is not None
#                 tensor[:, :, steering.feature_index] = steering.steering_value
#             return tensor

#         def save_feature_acts_hook(tensor: torch.Tensor, hook: HookPoint):
#             nonlocal feature_acts
#             feature_acts = tensor
#             return steer(tensor)

#         def steering_hook(tensor: torch.Tensor, hook: HookPoint):
#             assert feature_acts is not None, "Feature acts should be saved before steering"
#             difference = (steer(feature_acts) - feature_acts) @ sae.decoder.weight.T
#             tensor += difference.detach()
#             return tensor

#         sae = get_sae(steering.sae)
#         return [
#             (f"{sae.cfg.hook_point_out}.sae.hook_feature_acts", save_feature_acts_hook),
#             (f"{sae.cfg.hook_point_out}", steering_hook),
#         ]

#     steering_hooks = sum([generate_steering_hook(steering) for steering in request.steerings], [])

#     candidates = [f"{sae.cfg.hook_point_out}.sae.hook_feature_acts" for sae, _ in saes]
#     if request.detach_at_attn_scores:
#         candidates += [f"blocks.{i}.attn.hook_attn_scores" for i in range(model.cfg.n_layers)]

#     with apply_sae(model, [sae for sae, _ in saes]):
#         with model.hooks(steering_hooks):
#             with detach_at(model, candidates):
#                 input = (
#                     model.to_tokens(request.input_text, prepend_bos=False)
#                     if isinstance(request.input_text, str)
#                     else torch.tensor([request.input_text], device=device)
#                 )
#                 name_filter = (
#                     [f"{sae.cfg.hook_point_out}.sae.hook_feature_acts" for sae, _ in saes]
#                     + [f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.pre" for sae, _ in saes]
#                     + [f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.post" for sae, _ in saes]
#                 )
#                 if request.detach_at_attn_scores:
#                     name_filter += [f"blocks.{i}.attn.hook_attn_scores.pre" for i in range(model.cfg.n_layers)] + [
#                         f"blocks.{i}.attn.hook_attn_scores.post" for i in range(model.cfg.n_layers)
#                     ]
#                     name_filter += [f"blocks.{i}.attn.hook_pattern" for i in range(model.cfg.n_layers)]
#                 logits, cache = model.run_with_ref_cache(input, names_filter=name_filter)
#                 tracing_results = []
#                 for tracing in request.tracings:
#                     model.zero_grad()
#                     if isinstance(tracing, LogitsNode):
#                         assert tracing.position < logits.shape[1], "Position out of range"
#                         assert tracing.token_id < logits.shape[2], "Token id out of range"
#                         logits[:, tracing.position, tracing.token_id].backward(retain_graph=True)
#                         node = {
#                             **tracing.model_dump(),
#                             "activation": logits[0, tracing.position, tracing.token_id].item(),
#                             "id": f"logits-{tracing.position}-{tracing.token_id}",
#                         }
#                     elif isinstance(tracing, FeatureNode):
#                         sae = get_sae(tracing.sae)
#                         assert (
#                             tracing.position < cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.pre"][0].shape[0]
#                         ), "Position out of range"
#                         assert (
#                             tracing.feature_index
#                             < cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.pre"][0].shape[1]
#                         ), "Feature index out of range"
#                         cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.pre"][0][
#                             tracing.position, tracing.feature_index
#                         ].backward(retain_graph=True)
#                         node = {
#                             **tracing.model_dump(),
#                             "activation": cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.pre"][0][
#                                 tracing.position, tracing.feature_index
#                             ].item(),
#                             "max_activation": max_feature_acts[tracing.sae][tracing.feature_index],
#                             "id": f"feature-{tracing.sae}-{tracing.position}-{tracing.feature_index}",
#                         }
#                     elif isinstance(tracing, AttnScoreNode):
#                         assert tracing.layer < model.cfg.n_layers, "Layer out of range"
#                         attn_scores = cache[f"blocks.{tracing.layer}.attn.hook_attn_scores.pre"]
#                         assert tracing.head < attn_scores.shape[1], "Head out of range"
#                         assert tracing.query < attn_scores.shape[2], "Query out of range"
#                         assert tracing.key < attn_scores.shape[3], "Key out of range"
#                         attn_scores[:, tracing.head, tracing.query, tracing.key].backward(retain_graph=True)
#                         node = {
#                             **tracing.model_dump(),
#                             "activation": attn_scores[0, tracing.head, tracing.query, tracing.key].item(),
#                             "id": f"attn-score-{tracing.layer}-{tracing.head}-{tracing.query}-{tracing.key}",
#                             "pattern": cache[f"blocks.{tracing.layer}.attn.hook_pattern"][0][
#                                 tracing.head, tracing.query, tracing.key
#                             ].item(),
#                         }
#                     else:
#                         raise AssertionError("Unknown node type")

#                     contributors = []
#                     for sae, name in saes:
#                         feature_acts = cache[f"{sae.cfg.hook_point_out}.sae.hook_feature_acts.post"]
#                         if feature_acts.grad is None:
#                             continue
#                         attributions = feature_acts.grad[0] * feature_acts[0]
#                         for index in (attributions > request.tracing_threshold).nonzero():
#                             index = tuple(index.tolist())
#                             contributors.append(
#                                 {
#                                     "node": {
#                                         "type": "feature",
#                                         "sae": name,
#                                         "feature_index": index[1],
#                                         "position": index[0],
#                                         "activation": feature_acts[0][index].item(),
#                                         "max_activation": max_feature_acts[name][index[1]],
#                                         "id": f"feature-{name}-{index[0]}-{index[1]}",
#                                     },
#                                     "attribution": attributions[index].item(),
#                                 }
#                             )
#                         feature_acts.grad.zero_()

#                     if request.detach_at_attn_scores:
#                         for i in range(model.cfg.n_layers):
#                             attn_scores = cache[f"blocks.{i}.attn.hook_attn_scores.post"]
#                             if attn_scores.grad is None:
#                                 continue
#                             attributions = attn_scores.grad[0] * attn_scores[0]
#                             for index in (attributions > request.tracing_threshold).nonzero():
#                                 index = tuple(index.tolist())
#                                 contributors.append(
#                                     {
#                                         "node": {
#                                             "type": "attn-score",
#                                             "layer": i,
#                                             "head": index[0],
#                                             "query": index[1],
#                                             "key": index[2],
#                                             "activation": attn_scores[0][index].item(),
#                                             "pattern": cache[f"blocks.{i}.attn.hook_pattern"][0][index].item(),
#                                             "id": f"attn-score-{i}-{index[0]}-{index[1]}-{index[2]}",
#                                         },
#                                         "attribution": attributions[index].item(),
#                                     }
#                                 )
#                             attn_scores.grad.zero_()

#                     if request.tracing_top_k is not None:
#                         contributors = sorted(contributors, key=lambda c: -c["attribution"])[: request.tracing_top_k]

#                     tracing_results.append({"node": node, "contributors": contributors})

#                 result = {
#                     "context": [
#                         bytearray([byte_decoder[c] for c in t])
#                         # Method `convert_ids_to_tokens` should exist on GPT2Tokenizer and other BPE tokenizers.
#                         for t in model.tokenizer.convert_ids_to_tokens(input[0])  # type: ignore
#                     ],
#                     "token_ids": input[0].tolist(),
#                     "tracings": tracing_results,
#                 }
#     return Response(content=msgpack.packb(result), media_type="application/x-msgpack")


# @app.post("/dictionaries/{dictionary_name}/features/{feature_index}/interpret")
# def feature_interpretation(
#     dictionary_name: str,
#     feature_index: int,
#     type: str,
#     custom_interpretation: str | None = None,
# ):
#     model = get_model(dictionary_name)
#     dictionary = client.get_dictionary(dictionary_name, dictionary_series=dictionary_series)
#     assert dictionary is not None, "Dictionary not found"
#     path = dictionary["path"]
#     if type == "custom":
#         interpretation: Any = {
#             "text": custom_interpretation,
#             "validation": [
#                 {
#                     "method": "manual",
#                     "passed": True,
#                 }
#             ],
#         }
#     elif type == "auto":
#         cfg = AutoInterpConfig(
#             **{
#                 "sae": SAEConfig.from_pretrained(path).to_dict(),
#                 "lm": LanguageModelConfig.from_pretrained_sae(path).to_dict(),
#                 "openai_api_key": os.environ.get("OPENAI_API_KEY"),
#                 "openai_base_url": os.environ.get("OPENAI_BASE_URL"),
#             }
#         )
#         feature = client.get_feature(dictionary_name, feature_index, dictionary_series=dictionary_series)
#         assert feature is not None, "Feature not found"
#         result = generate_description(model, feature["analysis"][0], cfg)
#         interpretation = {
#             "text": result["response"],
#             "validation": [],
#             "detail": result,
#         }
#     elif type == "validate":
#         cfg = AutoInterpConfig(
#             **{
#                 "sae": SAEConfig.from_pretrained(path).to_dict(),
#                 "lm": LanguageModelConfig.from_pretrained_sae(path).to_dict(),
#                 "openai_api_key": os.environ.get("OPENAI_API_KEY"),
#                 "openai_base_url": os.environ.get("OPENAI_BASE_URL"),
#             }
#         )
#         feature = client.get_feature(dictionary_name, feature_index, dictionary_series=dictionary_series)
#         assert feature is not None, "Feature not found"
#         interpretation = feature["interpretation"] if "interpretation" in feature else None
#         if interpretation is None:
#             return Response(content="Feature interpretation not found", status_code=404)
#         validation = cast(Any, interpretation["validation"])
#         if not any(v["method"] == "activation" for v in validation):
#             validation_result = check_description(
#                 model,
#                 cfg,
#                 feature_index,
#                 cast(str, interpretation["text"]),
#                 False,
#                 feature_activation=feature["analysis"][0],
#             )
#             validation.append(
#                 {
#                     "method": "activation",
#                     "passed": validation_result["passed"],
#                     "detail": validation_result,
#                 }
#             )
#         if not any(v["method"] == "generative" for v in validation):
#             validation_result = check_description(
#                 model,
#                 cfg,
#                 feature_index,
#                 cast(str, interpretation["text"]),
#                 True,
#                 sae=get_sae(dictionary_name),
#             )
#             validation.append(
#                 {
#                     "method": "generative",
#                     "passed": validation_result["passed"],
#                     "detail": validation_result,
#                 }
#             )
#     else:
#         return Response(content="Invalid interpretation type", status_code=400)

#     try:
#         client.update_feature(
#             dictionary_name,
#             feature_index,
#             {"interpretation": interpretation},
#             dictionary_series=dictionary_series,
#         )
#     except ValueError as e:
#         return Response(content=str(e), status_code=400)
#     return interpretation


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
