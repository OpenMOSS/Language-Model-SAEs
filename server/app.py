import json
import os
import threading
from functools import lru_cache, wraps
from typing import Any, Generator, Optional

import msgpack
import numpy as np
import plotly.graph_objects as go
import torch
from datasets import Dataset
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

try:
    from torchvision import transforms
except ImportError:
    transforms = None
    print("WARNING: torchvision not found, image processing will be disabled")

from lm_saes.backend import LanguageModel
from lm_saes.config import MongoDBConfig, SAEConfig
from lm_saes.database import FeatureAnalysisSampling, FeatureRecord, MongoClient
from lm_saes.resource_loaders import load_dataset_shard, load_model
from lm_saes.sae import SparseAutoEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=1000)

client = MongoClient(MongoDBConfig())
sae_series = os.environ.get("SAE_SERIES", "default")
tokenizer_only = os.environ.get("TOKENIZER_ONLY", "false").lower() == "true"


def synchronized(func):
    """Decorator to ensure sequential execution of a function based on parameters.

    Different parameters can be acquired in parallel, but the same parameters
    will be executed sequentially.
    """
    locks: dict[frozenset[tuple[str, Any]], threading.Lock] = {}
    global_lock = threading.Lock()

    @wraps(func)
    def wrapper(*args, **kwargs):
        assert len(args) == 0, "Positional arguments are not supported"
        key = frozenset(kwargs.items())

        # The lock creation is locked by the global lock to avoid race conditions on locks.
        with global_lock:
            if key not in locks:
                locks[key] = threading.Lock()
            lock = locks[key]

        with lock:
            return func(*args, **kwargs)

    return wrapper


@lru_cache(maxsize=8)
@synchronized
def get_model(*, name: str) -> LanguageModel:
    """Load and cache a language model."""
    cfg = client.get_model_cfg(name)
    if cfg is None:
        raise ValueError(f"Model {name} not found")
    cfg.tokenizer_only = tokenizer_only
    return load_model(cfg)


@lru_cache(maxsize=16)
@synchronized
def get_dataset(*, name: str, shard_idx: int = 0, n_shards: int = 1) -> Dataset:
    """Load and cache a dataset shard."""
    cfg = client.get_dataset_cfg(name)
    assert cfg is not None, f"Dataset {name} not found"
    return load_dataset_shard(cfg, shard_idx, n_shards)


@lru_cache(maxsize=8)
@synchronized
def get_sae(*, name: str) -> SparseAutoEncoder:
    """Load and cache a sparse autoencoder."""
    path = client.get_sae_path(name, sae_series)
    assert path is not None, f"SAE {name} not found"
    cfg = SAEConfig.from_pretrained(path)
    sae = SparseAutoEncoder.from_config(cfg)
    sae.eval()
    return sae


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
    sampling: FeatureAnalysisSampling, start: int | None = None, end: int | None = None
) -> list[dict[str, Any]]:
    def process_sample(
        *,
        sparse_feature_acts,
        context_idx,
        dataset_name,
        model_name,
        shard_idx=None,
        n_shards=None,
    ):
        model = get_model(name=model_name)
        data = get_dataset(name=dataset_name, shard_idx=shard_idx, n_shards=n_shards)[context_idx.item()]

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

        # Process text data if present
        if "text" in data:
            text_ranges = [origin["range"] for origin in origins if origin is not None and origin["key"] == "text"]
            if text_ranges:
                max_text_origin = max(text_ranges, key=lambda x: x[1])
                data["text"] = data["text"][: max_text_origin[1]]

        return {
            **data,
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
def list_features(name: str, start: int, end: int, analysis_name: str | None = None, sample_length: int = 1):
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

        samples = extract_samples(sampling, 0, sample_length) if sampling is not None else None

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

    return Response(
        content=msgpack.packb(make_serializable(results)),
        media_type="application/x-msgpack",
    )


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

    samples = extract_samples(sampling, start, None if length is None else start + length)
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
