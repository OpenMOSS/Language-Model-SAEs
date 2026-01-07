import json
import os
import threading
from contextlib import asynccontextmanager
from functools import lru_cache, wraps
from typing import Any, Callable, Generator, Generic, Optional, ParamSpec, TypeVar

import msgpack
import numpy as np
import plotly.graph_objects as go
import torch
from datasets import Dataset
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.backend import LanguageModel
from lm_saes.backend.language_model import TransformerLensLanguageModel
from lm_saes.circuit.tracing.attribution import attribute
from lm_saes.circuit.tracing.replacement_model import ReplacementModel
from lm_saes.circuit.utils.create_graph_files import serialize_graph
from lm_saes.circuit.utils.transcoder_set import TranscoderSet, TranscoderSetConfig
from lm_saes.config import BaseSAEConfig, MongoDBConfig
from lm_saes.database import FeatureAnalysisSampling, FeatureRecord, MongoClient
from lm_saes.lorsa import LowRankSparseAttention
from lm_saes.resource_loaders import load_dataset_shard, load_model
from lm_saes.sae import SparseAutoEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"

client = MongoClient(MongoDBConfig())
sae_series = os.environ.get("SAE_SERIES", "default")
tokenizer_only = os.environ.get("TOKENIZER_ONLY", "false").lower() == "true"


P = ParamSpec("P")
R = TypeVar("R")


class synchronized(Generic[P, R]):
    """Decorator to ensure sequential execution of a function based on parameters.

    Different parameters can be acquired in parallel, but the same parameters
    will be executed sequentially.
    """

    _func: Callable[P, R]

    def __init__(self, func: Callable[P, R]) -> None:
        self._func = func
        self._locks: dict[frozenset[tuple[str, Any]], threading.Lock] = {}
        self._global_lock = threading.Lock()
        wraps(func)(self)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        assert len(args) == 0, "Positional arguments are not supported"
        key = frozenset(kwargs.items())

        # The lock creation is locked by the global lock to avoid race conditions on locks.
        with self._global_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            lock = self._locks[key]

        with lock:
            return self._func(*args, **kwargs)  # type: ignore[call-arg]

    def __getattr__(self, name: str):
        return getattr(self._func, name)


@synchronized
@lru_cache(maxsize=8)
def get_model(*, name: str) -> LanguageModel:
    """Load and cache a language model."""
    cfg = client.get_model_cfg(name)
    if cfg is None:
        raise ValueError(f"Model {name} not found")
    cfg.tokenizer_only = tokenizer_only
    return load_model(cfg)


@synchronized
@lru_cache(maxsize=16)
def get_dataset(*, name: str, shard_idx: int = 0, n_shards: int = 1) -> Dataset:
    """Load and cache a dataset shard."""
    cfg = client.get_dataset_cfg(name)
    assert cfg is not None, f"Dataset {name} not found"
    return load_dataset_shard(cfg, shard_idx, n_shards)


@synchronized
@lru_cache(maxsize=8)
def get_sae(*, name: str) -> AbstractSparseAutoEncoder:
    """Load and cache a sparse autoencoder."""
    path = client.get_sae_path(name, sae_series)
    assert path is not None, f"SAE {name} not found"
    cfg = BaseSAEConfig.from_pretrained(path)
    sae = AbstractSparseAutoEncoder.from_config(cfg)
    sae.eval()
    return sae


@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_models = os.environ["PRELOAD_MODELS"].strip().split(",") if os.environ.get("PRELOAD_MODELS") else []
    for model in preload_models:
        get_model(name=model)

    preload_saes = os.environ["PRELOAD_SAES"].strip().split(",") if os.environ.get("PRELOAD_SAES") else []
    for sae in preload_saes:
        get_sae(name=sae)

    yield

    get_model.cache_clear()
    get_dataset.cache_clear()
    get_sae.cache_clear()


app = FastAPI(lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)


def make_serializable(obj):
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def trim_minimum(
    origins: list[dict[str, Any] | None],
    feature_acts_indices: np.ndarray,
    feature_acts_values: np.ndarray,
) -> tuple[list[dict[str, Any] | None], np.ndarray, np.ndarray]:
    """Trim multiple arrays to the length of the shortest non-None array.

    Args:
        origins: Origins
        feature_acts_indices: Feature acts indices
        feature_acts_values: Feature acts values

    Returns:
        list: List of trimmed arrays
    """

    min_length = min(len(origins), feature_acts_indices[-1] + 10)
    feature_acts_indices_mask = feature_acts_indices <= min_length
    return (
        origins[: int(min_length)],
        feature_acts_indices[feature_acts_indices_mask],
        feature_acts_values[feature_acts_indices_mask],
    )


@app.exception_handler(AssertionError)
async def assertion_error_handler(request, exc):
    return Response(content=str(exc), status_code=400)


@app.exception_handler(torch.cuda.OutOfMemoryError)
async def oom_error_handler(request, exc):
    print("CUDA Out of memory. Clearing cache.")
    # Clear LRU caches
    get_model.cache_clear()
    get_dataset.cache_clear()
    get_sae.cache_clear()
    return Response(content="CUDA Out of memory", status_code=500)


@app.get("/dictionaries")
def list_dictionaries():
    return client.list_saes(sae_series=sae_series, has_analyses=True)


@app.get("/dictionaries/{name}/metrics")
def get_available_metrics(name: str):
    """Get available metrics for a dictionary.

    Args:
        name: Name of the dictionary/SAE

    Returns:
        List of available metric names
    """
    metrics = client.get_available_metrics(name, sae_series=sae_series)
    return {"metrics": metrics}


@app.get("/dictionaries/{name}/features/count")
def count_features_with_filters(
    name: str,
    feature_analysis_name: str | None = None,
    metric_filters: str | None = None,
):
    """Count features that match the given filters.

    Args:
        name: Name of the dictionary/SAE
        feature_analysis_name: Optional analysis name
        metric_filters: Optional JSON string of metric filters

    Returns:
        Count of features matching the filters
    """
    # Parse metric filters if provided
    parsed_metric_filters = None
    if metric_filters:
        try:
            parsed_metric_filters = json.loads(metric_filters)
        except (json.JSONDecodeError, TypeError):
            return Response(
                content=f"Invalid metric_filters format: {metric_filters}",
                status_code=400,
            )

    count = client.count_features_with_filters(
        sae_name=name, sae_series=sae_series, name=feature_analysis_name, metric_filters=parsed_metric_filters
    )

    return {"count": count}


def extract_samples(
    sampling: FeatureAnalysisSampling,
    start: int | None = None,
    end: int | None = None,
    visible_range: int | None = None,
) -> list[dict[str, Any]]:
    def process_sample(
        *,
        sparse_feature_acts: tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None],
        context_idx: int,
        dataset_name: str,
        model_name: str,
        shard_idx: int | None = None,
        n_shards: int | None = None,
    ):
        model = get_model(name=model_name)
        data = get_dataset(name=dataset_name, shard_idx=shard_idx, n_shards=n_shards)[context_idx]

        origins = model.trace({k: [v] for k, v in data.items()})[0]

        (
            feature_acts_indices,
            feature_acts_values,
            z_pattern_indices,
            z_pattern_values,
        ) = sparse_feature_acts

        assert origins is not None and feature_acts_indices is not None and feature_acts_values is not None, (
            "Origins and feature acts must not be None"
        )

        token_offset = 0
        if visible_range is not None:  # Drop tokens before and after the highest activating token
            if len(feature_acts_indices) == 0:
                max_feature_act_index = 0
            else:
                max_feature_act_index = int(feature_acts_indices[np.argmax(feature_acts_values).item()].item())

            feature_acts_mask = np.logical_and(
                feature_acts_indices > max_feature_act_index - visible_range,
                feature_acts_indices < max_feature_act_index + visible_range,
            )
            feature_acts_indices = feature_acts_indices[feature_acts_mask]
            feature_acts_values = feature_acts_values[feature_acts_mask]

            if z_pattern_indices is not None and z_pattern_values is not None:
                z_pattern_mask = np.logical_and(
                    z_pattern_indices > max_feature_act_index - visible_range,
                    z_pattern_indices < max_feature_act_index + visible_range,
                ).all(axis=0)
                z_pattern_indices = z_pattern_indices[:, z_pattern_mask]
                z_pattern_values = z_pattern_values[z_pattern_mask]

            token_offset = max(0, max_feature_act_index - visible_range)

            origins = origins[token_offset : max_feature_act_index + visible_range]

        text_offset = None
        if "text" in data:
            text_ranges = [origin["range"] for origin in origins if origin is not None and origin["key"] == "text"]
            if text_ranges:
                max_text_origin = max(text_ranges, key=lambda x: x[1])
                data["text"] = data["text"][: max_text_origin[1]]
                if visible_range is not None:
                    text_offset = min(text_ranges, key=lambda x: x[0])[0]
                    data["text"] = data["text"][text_offset:]

        return {
            **data,
            "token_offset": token_offset,
            "text_offset": text_offset,
            "origins": origins,
            "feature_acts_indices": feature_acts_indices,
            "feature_acts_values": feature_acts_values,
            "z_pattern_indices": z_pattern_indices,
            "z_pattern_values": z_pattern_values,
        }

    def index_select(
        indices: np.ndarray,
        values: np.ndarray,
        i: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select i-th sample from sparse tensor indices and values."""
        mask = indices[0] == i
        return indices[1:, mask], values[mask]

    def process_sparse_feature_acts(
        feature_acts_indices: np.ndarray,
        feature_acts_values: np.ndarray,
        z_pattern_indices: np.ndarray | None,
        z_pattern_values: np.ndarray | None,
        start: int,
        end: int,
    ) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None], Any, None]:
        for i in range(start, end):
            feature_acts_indices_i, feature_acts_values_i = index_select(feature_acts_indices, feature_acts_values, i)
            if z_pattern_indices is not None and z_pattern_values is not None:
                z_pattern_indices_i, z_pattern_values_i = index_select(z_pattern_indices, z_pattern_values, i)
            else:
                z_pattern_indices_i, z_pattern_values_i = None, None
            yield feature_acts_indices_i[0], feature_acts_values_i, z_pattern_indices_i, z_pattern_values_i

    start = start if start is not None else 0
    end = end if end is not None else len(sampling.context_idx)

    return [
        process_sample(
            sparse_feature_acts=sparse_feature_acts,
            context_idx=context_idx,
            dataset_name=dataset_name,
            model_name=model_name,
            shard_idx=shard_idx,
            n_shards=n_shards,
        )
        for sparse_feature_acts, context_idx, dataset_name, model_name, shard_idx, n_shards in zip(
            process_sparse_feature_acts(
                sampling.feature_acts_indices,
                sampling.feature_acts_values,
                sampling.z_pattern_indices,
                sampling.z_pattern_values,
                start,
                end,
            ),
            sampling.context_idx[start:end],
            sampling.dataset_name[start:end],
            sampling.model_name[start:end],
            sampling.shard_idx[start:end] if sampling.shard_idx is not None else [0] * (end - start),
            sampling.n_shards[start:end] if sampling.n_shards is not None else [1] * (end - start),
        )
    ]


@app.get("/dictionaries/{name}/features/{feature_index}")
def get_feature(
    name: str,
    feature_index: str | int,
    feature_analysis_name: str | None = None,
    metric_filters: str | None = None,
    no_samplings: bool = False,
):
    # Parse feature_index if it's a string
    if isinstance(feature_index, str) and feature_index != "random":
        try:
            feature_index = int(feature_index)
        except ValueError:
            return Response(
                content=f"Feature index {feature_index} is not a valid integer",
                status_code=400,
            )

    # Parse metric filters if provided
    parsed_metric_filters = None
    if metric_filters:
        try:
            parsed_metric_filters = json.loads(metric_filters)
        except (json.JSONDecodeError, TypeError):
            return Response(
                content=f"Invalid metric_filters format: {metric_filters}",
                status_code=400,
            )

    # Get feature data
    feature = (
        client.get_random_alive_feature(
            sae_name=name, sae_series=sae_series, name=feature_analysis_name, metric_filters=parsed_metric_filters
        )
        if feature_index == "random"
        else client.get_feature(sae_name=name, sae_series=sae_series, index=feature_index)
    )

    if feature is None:
        return Response(
            content=f"Feature {feature_index} not found in SAE {name}",
            status_code=404,
        )

    analysis = next(
        (a for a in feature.analyses if a.name == feature_analysis_name or feature_analysis_name is None),
        None,
    )
    if analysis is None:
        return Response(
            content=f"Feature analysis {feature_analysis_name} not found in SAE {name}"
            if feature_analysis_name is not None
            else f"No feature analysis found in SAE {name}",
            status_code=404,
        )

    sample_groups = (
        [
            {"analysis_name": sampling.name, "samples": extract_samples(sampling, 0, len(sampling.context_idx))}
            for sampling in analysis.samplings
        ]
        if not no_samplings
        else None
    )

    result = {
        "feature_index": feature.index,
        "analysis_name": analysis.name,
        "interpretation": feature.interpretation,
        "dictionary_name": feature.sae_name,
        "logits": feature.logits,
        "decoder_norms": analysis.decoder_norms,
        "decoder_similarity_matrix": analysis.decoder_similarity_matrices,
        "decoder_inner_product_matrix": analysis.decoder_inner_product_matrices,
        "act_times": analysis.act_times,
        "max_feature_act": analysis.max_feature_acts,
        "n_analyzed_tokens": analysis.n_analyzed_tokens,
        "sample_groups": sample_groups,
        "is_bookmarked": client.is_bookmarked(sae_name=name, sae_series=sae_series, feature_index=feature.index),
    }

    return Response(
        content=msgpack.packb(make_serializable(result)),
        media_type="application/x-msgpack",
    )


@app.get("/dictionaries/{name}/features")
def list_features(
    name: str,
    start: int,
    end: int,
    analysis_name: str | None = None,
    sample_length: int = 1,
    visible_range: int | None = None,
):
    features = client.list_features(sae_name=name, sae_series=sae_series, indices=list(range(start, end)))

    def process_feature(feature: FeatureRecord):
        analysis = next(
            (a for a in feature.analyses if a.name == analysis_name or analysis_name is None),
            None,
        )
        if analysis is None:
            return None

        sampling = next(
            (s for s in analysis.samplings if s.name == "top_activations"),
            None,
        )

        samples = (
            extract_samples(sampling, 0, sample_length, visible_range=visible_range) if sampling is not None else None
        )

        return {
            "feature_index": feature.index,
            "analysis_name": analysis.name,
            "interpretation": feature.interpretation,
            "dictionary_name": feature.sae_name,
            "act_times": analysis.act_times,
            "max_feature_act": analysis.max_feature_acts,
            "n_analyzed_tokens": analysis.n_analyzed_tokens,
            "samples": samples,
        }

    results = [process_feature(feature) for feature in features]

    return make_serializable(results)


@app.get("/dictionaries/{name}/features/{feature_index}/samplings")
def get_samplings(name: str, feature_index: int, analysis_name: str | None = None):
    """Get all available samplings for a feature."""
    feature = client.get_feature(sae_name=name, sae_series=sae_series, index=feature_index)
    if feature is None:
        return Response(content=f"Feature {feature_index} not found in SAE {name}", status_code=404)
    analysis = next(
        (a for a in feature.analyses if a.name == analysis_name or analysis_name is None),
        None,
    )
    if analysis is None:
        return Response(content=f"Analysis {analysis_name} not found in SAE {name}", status_code=404)
    return [{"name": sampling.name, "length": len(sampling.context_idx)} for sampling in analysis.samplings]


@app.get("/dictionaries/{name}/features/{feature_index}/sampling/{sampling_name}")
def get_samples(
    name: str,
    feature_index: int,
    sampling_name: str,
    analysis_name: str | None = None,
    start: int = 0,
    length: int | None = None,
    visible_range: int | None = None,
):
    """Get all samples for a feature."""
    feature = client.get_feature(sae_name=name, sae_series=sae_series, index=feature_index)
    if feature is None:
        return Response(content=f"Feature {feature_index} not found in SAE {name}", status_code=404)

    analysis = next(
        (a for a in feature.analyses if a.name == analysis_name or analysis_name is None),
        None,
    )
    if analysis is None:
        return Response(content=f"Analysis {analysis_name} not found in SAE {name}", status_code=404)

    sampling = next(
        (s for s in analysis.samplings if s.name == sampling_name),
        None,
    )
    if sampling is None:
        return Response(content=f"Sampling {sampling_name} not found in Analysis {analysis_name}", status_code=404)

    samples = extract_samples(sampling, start, None if length is None else start + length, visible_range=visible_range)
    return Response(
        content=msgpack.packb(make_serializable(samples)),
        media_type="application/x-msgpack",
    )


@app.get("/dictionaries/{name}")
def get_dictionary(name: str):
    # Get feature activation times
    feature_activation_times = client.get_feature_act_times(name, sae_series=sae_series)
    if feature_activation_times is None:
        return Response(content=f"Dictionary {name} not found", status_code=404)

    # Create histogram of log activation times
    log_act_times = np.log10(np.array(list(feature_activation_times.values())))
    feature_activation_times_histogram = go.Histogram(
        x=log_act_times,
        nbinsx=100,
        hovertemplate="Count: %{y}<br>Range: %{x}<extra></extra>",
        marker_color="#636EFA",
        showlegend=False,
    ).to_plotly_json()

    # Get alive feature count
    alive_feature_count = client.get_alive_feature_count(name, sae_series=sae_series)
    if alive_feature_count is None:
        return Response(content=f"SAE {name} not found", status_code=404)

    # Prepare and return response
    response_data = {
        "dictionary_name": name,
        "feature_activation_times_histogram": [feature_activation_times_histogram],
        "alive_feature_count": alive_feature_count,
    }

    return Response(
        content=msgpack.packb(make_serializable(response_data)),
        media_type="application/x-msgpack",
    )


@app.get("/dictionaries/{name}/analyses")
def get_analyses(name: str):
    """Get all available analyses for a dictionary.

    Args:
        name: Name of the dictionary/SAE

    Returns:
        List of analysis names
    """
    # Get a random feature to check its available analyses
    feature = client.get_random_alive_feature(sae_name=name, sae_series=sae_series)
    if feature is None:
        return Response(content=f"Dictionary {name} not found", status_code=404)

    # Extract unique analysis names from feature
    analyses = list(set(analysis.name for analysis in feature.analyses))
    return analyses


@app.post("/dictionaries/{name}/features/{feature_index}/infer")
def infer_feature(name: str, feature_index: int, text: str):
    """Infer feature activations for a given text."""
    model_name = client.get_sae_model_name(name, sae_series)
    assert model_name is not None, f"SAE {name} not found or no model name is associated with it"
    model = get_model(name=model_name)
    sae = get_sae(name=name)

    activations = model.to_activations({"text": [text]}, hook_points=sae.cfg.associated_hook_points)
    activations = sae.normalize_activations(activations)
    x, encoder_kwargs, _ = sae.prepare_input(activations)
    feature_acts = sae.encode(x, **encoder_kwargs)[0, :, feature_index]

    feature_acts = feature_acts.to_sparse()

    origins = model.trace({"text": [text]})[0]

    return Response(
        content=msgpack.packb(
            make_serializable(
                {
                    "text": text,
                    "origins": origins,
                    "feature_acts_indices": feature_acts.indices()[0],
                    "feature_acts_values": feature_acts.values(),
                }
            )
        ),
        media_type="application/x-msgpack",
    )


@app.get("/sae-sets")
def list_sae_sets():
    """List all available SAE sets."""
    return [sae_set.name for sae_set in client.list_sae_sets()]


class TraceRequest(BaseModel):
    text: str


@app.post("/circuit/{sae_set_name}")
def trace(sae_set_name: str, request: TraceRequest):
    """Trace a given text through a given SAE set."""
    text = request.text
    sae_set = client.get_sae_set(name=sae_set_name)
    assert sae_set is not None, f"SAE set {sae_set_name} not found"
    sae_names = sae_set.sae_names
    saes = {sae_name: get_sae(name=sae_name) for sae_name in sae_names}

    model_name = client.get_sae_model_name(sae_names[0], sae_set.sae_series)
    model = get_model(name=model_name)
    assert isinstance(model, TransformerLensLanguageModel) and model.model is not None, (
        "Circuit tracing only supports exact model of TransformerLens backend"
    )

    lorsas = {sae_name: sae for sae_name, sae in saes.items() if isinstance(sae, LowRankSparseAttention)}
    transcoders = {sae_name: sae for sae_name, sae in saes.items() if isinstance(sae, SparseAutoEncoder)}
    assert len(lorsas) == len(transcoders) == model.model.cfg.n_layers, (
        "Currently only supports SAE sets with all layers"
    )

    plt_set = TranscoderSet(
        TranscoderSetConfig(
            n_layers=model.model.cfg.n_layers,
            d_sae=list(transcoders.values())[0].cfg.d_sae,
            feature_input_hook="ln2.hook_normalized",
            feature_output_hook="hook_mlp_out",
        ),
        {i: transcoder for i, transcoder in enumerate(transcoders.values())},
    )

    replacement_model = ReplacementModel.from_pretrained(model.cfg, plt_set, list(lorsas.values()))

    graph = attribute(
        prompt=text,
        model=replacement_model,
        max_n_logits=1,
        desired_logit_prob=0.98,
        batch_size=1,
        max_feature_nodes=256,
        sae_series=sae_series,
        qk_tracing_topk=10,
    )
    graph.cfg.tokenizer_name = model.cfg.model_from_pretrained_path or model.cfg.model_name
    graph_data = serialize_graph(
        graph=graph,
        node_threshold=0.8,
        edge_threshold=0.98,
        clt_names=list(transcoders.keys()),
        lorsa_names=list(lorsas.keys()),
    )

    return graph_data


@app.post("/dictionaries/{name}/features/{feature_index}/bookmark")
def add_bookmark(name: str, feature_index: int):
    """Add a bookmark for a feature.

    Args:
        name: Name of the dictionary/SAE
        feature_index: Index of the feature to bookmark

    Returns:
        Success response or error
    """
    try:
        success = client.add_bookmark(sae_name=name, sae_series=sae_series, feature_index=feature_index)
        if success:
            return {"message": "Bookmark added successfully"}
        else:
            return Response(content="Feature is already bookmarked", status_code=409)
    except ValueError as e:
        return Response(content=str(e), status_code=404)


@app.delete("/dictionaries/{name}/features/{feature_index}/bookmark")
def remove_bookmark(name: str, feature_index: int):
    """Remove a bookmark for a feature.

    Args:
        name: Name of the dictionary/SAE
        feature_index: Index of the feature to remove bookmark from

    Returns:
        Success response or error
    """
    success = client.remove_bookmark(sae_name=name, sae_series=sae_series, feature_index=feature_index)
    if success:
        return {"message": "Bookmark removed successfully"}
    else:
        return Response(content="Bookmark not found", status_code=404)


@app.get("/dictionaries/{name}/features/{feature_index}/bookmark")
def check_bookmark(name: str, feature_index: int):
    """Check if a feature is bookmarked.

    Args:
        name: Name of the dictionary/SAE
        feature_index: Index of the feature

    Returns:
        Bookmark status
    """
    is_bookmarked = client.is_bookmarked(sae_name=name, sae_series=sae_series, feature_index=feature_index)
    return {"is_bookmarked": is_bookmarked}


@app.get("/bookmarks")
def list_bookmarks(sae_name: Optional[str] = None, sae_series: Optional[str] = None, limit: int = 100, skip: int = 0):
    """List bookmarks with optional filtering.

    Args:
        sae_name: Optional SAE name filter
        sae_series: Optional SAE series filter
        limit: Maximum number of bookmarks to return
        skip: Number of bookmarks to skip (for pagination)

    Returns:
        List of bookmarks
    """
    bookmarks = client.list_bookmarks(sae_name=sae_name, sae_series=sae_series, limit=limit, skip=skip)

    # Convert to dict for JSON serialization
    bookmark_data = []
    for bookmark in bookmarks:
        bookmark_dict = bookmark.model_dump()
        # Convert datetime to ISO string for JSON
        bookmark_dict["created_at"] = bookmark.created_at.isoformat()
        bookmark_data.append(bookmark_dict)

    return {
        "bookmarks": bookmark_data,
        "total_count": client.get_bookmark_count(sae_name=sae_name, sae_series=sae_series),
    }


@app.put("/dictionaries/{name}/features/{feature_index}/bookmark")
def update_bookmark(name: str, feature_index: int, tags: Optional[list[str]] = None, notes: Optional[str] = None):
    """Update a bookmark with new tags or notes.

    Args:
        name: Name of the dictionary/SAE
        feature_index: Index of the feature
        tags: Optional new tags for the bookmark
        notes: Optional new notes for the bookmark

    Returns:
        Success response or error
    """
    success = client.update_bookmark(
        sae_name=name, sae_series=sae_series, feature_index=feature_index, tags=tags, notes=notes
    )
    if success:
        return {"message": "Bookmark updated successfully"}
    else:
        return Response(content="Bookmark not found", status_code=404)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
