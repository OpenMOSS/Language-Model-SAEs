from typing import Any, Optional

from fastapi import APIRouter

from lm_saes.database import BookmarkRecord
from server.config import client, sae_series
from server.logic.samples import extract_samples
from server.utils.common import make_serializable

router = APIRouter(prefix="/bookmarks", tags=["bookmarks"])


@router.get("")
def list_bookmarks(
    sae_name: Optional[str] = None,
    limit: int = 100,
    skip: int = 0,
    include_features: bool = True,
):
    """List bookmarks with optional filtering and feature data."""
    bookmarks = client.list_bookmarks(sae_name=sae_name, sae_series=sae_series, limit=limit, skip=skip)

    # Group bookmarks by SAE name for efficient feature fetching
    bookmarks_by_sae: dict[str, list[BookmarkRecord]] = {}
    for bookmark in bookmarks:
        if bookmark.sae_name not in bookmarks_by_sae:
            bookmarks_by_sae[bookmark.sae_name] = []
        bookmarks_by_sae[bookmark.sae_name].append(bookmark)

    # Fetch features for each SAE
    features_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    if include_features:
        for sae_name, sae_bookmarks in bookmarks_by_sae.items():
            feature_indices = [b.feature_index for b in sae_bookmarks]
            features = client.list_features(sae_name=sae_name, sae_series=sae_series, indices=feature_indices)

            for feature in features:
                analysis = next(
                    (a for a in feature.analyses if a.name == "default"),
                    None,
                )
                if analysis is None:
                    analysis = next((a for a in feature.analyses), None)

                if analysis is None:
                    continue

                sampling = next(
                    (s for s in analysis.samplings if s.name == "top_activations"),
                    None,
                )

                samples = extract_samples(sampling, 0, 1, visible_range=5) if sampling is not None else []

                features_by_key[(sae_name, feature.index)] = {
                    "feature_index": feature.index,
                    "analysis_name": analysis.name,
                    "interpretation": feature.interpretation,
                    "dictionary_name": feature.sae_name,
                    "act_times": analysis.act_times,
                    "max_feature_act": analysis.max_feature_acts,
                    "n_analyzed_tokens": analysis.n_analyzed_tokens,
                    "samples": samples,
                }

    # Convert to dict for JSON serialization
    bookmark_data = []
    for bookmark in bookmarks:
        bookmark_dict = bookmark.model_dump()
        # Convert datetime to ISO string for JSON
        bookmark_dict["created_at"] = bookmark.created_at.isoformat() + "Z"

        # Add feature data if available
        if include_features:
            feature_key = (bookmark.sae_name, bookmark.feature_index)
            if feature_key in features_by_key:
                bookmark_dict["feature"] = make_serializable(features_by_key[feature_key])

        bookmark_data.append(bookmark_dict)

    return {
        "bookmarks": bookmark_data,
        "total_count": client.get_bookmark_count(sae_name=sae_name, sae_series=sae_series),
    }
