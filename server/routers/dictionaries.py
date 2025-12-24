import json
from typing import Optional

import msgpack
import numpy as np
import plotly.graph_objects as go
from fastapi import APIRouter, Response
from pydantic import BaseModel

from lm_saes.database import FeatureRecord
from server.config import client, sae_series
from server.logic.loaders import get_model, get_sae
from server.logic.samples import extract_samples
from server.utils.common import make_serializable, natural_sort_key

router = APIRouter(prefix="/dictionaries", tags=["dictionaries"])


@router.get("")
def list_dictionaries():
    sae_names = client.list_saes(sae_series=sae_series, has_analyses=True)
    return sorted(sae_names, key=natural_sort_key)


@router.get("/{name}/metrics")
def get_available_metrics(name: str):
    """Get available metrics for a dictionary."""
    metrics = client.get_available_metrics(name, sae_series=sae_series)
    return {"metrics": metrics}


@router.get("/{name}/features/count")
def count_features_with_filters(
    name: str,
    feature_analysis_name: str | None = None,
    metric_filters: str | None = None,
):
    """Count features that match the given filters."""
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


@router.get("/{name}/features/{feature_index}")
def get_feature(
    name: str,
    feature_index: str | int,
    feature_analysis_name: str | None = None,
    metric_filters: str | None = None,
    no_samplings: bool = False,
):
    if isinstance(feature_index, str) and feature_index != "random":
        try:
            feature_index = int(feature_index)
        except ValueError:
            return Response(
                content=f"Feature index {feature_index} is not a valid integer",
                status_code=400,
            )

    parsed_metric_filters = None
    if metric_filters:
        try:
            parsed_metric_filters = json.loads(metric_filters)
        except (json.JSONDecodeError, TypeError):
            return Response(
                content=f"Invalid metric_filters format: {metric_filters}",
                status_code=400,
            )

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


@router.get("/{name}/features")
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


@router.get("/{name}/features/{feature_index}/samplings")
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


@router.get("/{name}/features/{feature_index}/sampling/{sampling_name}")
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


@router.get("/{name}")
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


@router.get("/{name}/analyses")
def get_analyses(name: str):
    """Get all available analyses for a dictionary."""
    feature = client.get_random_alive_feature(sae_name=name, sae_series=sae_series)
    if feature is None:
        return Response(content=f"Dictionary {name} not found", status_code=404)

    analyses = list(set(analysis.name for analysis in feature.analyses))
    return analyses


@router.post("/{name}/features/{feature_index}/infer")
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


class UpdateInterpretationRequest(BaseModel):
    text: str


@router.put("/{name}/features/{feature_index}/interpretation")
def update_interpretation(name: str, feature_index: int, request: UpdateInterpretationRequest):
    """Update a feature's interpretation."""
    interpretation = {"text": request.text}
    result = client.update_feature(
        sae_name=name,
        sae_series=sae_series,
        feature_index=feature_index,
        update_data={"interpretation": interpretation},
    )
    if result.matched_count == 0:
        return Response(content=f"Feature {feature_index} not found in SAE {name}", status_code=404)
    return {"message": "Interpretation updated successfully", "interpretation": interpretation}


@router.post("/{name}/features/{feature_index}/bookmark")
def add_bookmark(name: str, feature_index: int):
    """Add a bookmark for a feature."""
    try:
        success = client.add_bookmark(sae_name=name, sae_series=sae_series, feature_index=feature_index)
        if success:
            return {"message": "Bookmark added successfully"}
        else:
            return Response(content="Feature is already bookmarked", status_code=409)
    except ValueError as e:
        return Response(content=str(e), status_code=404)


@router.delete("/{name}/features/{feature_index}/bookmark")
def remove_bookmark(name: str, feature_index: int):
    """Remove a bookmark for a feature."""
    success = client.remove_bookmark(sae_name=name, sae_series=sae_series, feature_index=feature_index)
    if success:
        return {"message": "Bookmark removed successfully"}
    else:
        return Response(content="Bookmark not found", status_code=404)


@router.get("/{name}/features/{feature_index}/bookmark")
def check_bookmark(name: str, feature_index: int):
    """Check if a feature is bookmarked."""
    is_bookmarked = client.is_bookmarked(sae_name=name, sae_series=sae_series, feature_index=feature_index)
    return {"is_bookmarked": is_bookmarked}


@router.put("/{name}/features/{feature_index}/bookmark")
def update_bookmark(name: str, feature_index: int, tags: Optional[list[str]] = None, notes: Optional[str] = None):
    """Update a bookmark with new tags or notes."""
    success = client.update_bookmark(
        sae_name=name, sae_series=sae_series, feature_index=feature_index, tags=tags, notes=notes
    )
    if success:
        return {"message": "Bookmark updated successfully"}
    else:
        return Response(content="Bookmark not found", status_code=404)
