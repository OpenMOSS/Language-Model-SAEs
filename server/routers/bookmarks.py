import functools
import itertools
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import Response

from server.config import client, sae_series
from server.logic.samples import list_feature_data
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
    count = client.get_bookmark_count(sae_name=sae_name, sae_series=sae_series)

    features = (
        functools.reduce(
            lambda acc, x: acc | x,
            [
                list_feature_data(sae_name=sae_name, indices=[bookmark.feature_index for bookmark in bookmarks_i])
                for sae_name, bookmarks_i in itertools.groupby(
                    sorted(bookmarks, key=lambda x: x.sae_name), key=lambda x: x.sae_name
                )
            ],
        )  # Produce a dictionary of feature data by (sae_name, feature_index)
        if include_features
        else {}
    )

    bookmark_data = []
    for bookmark in bookmarks:
        bookmark_dict = bookmark.model_dump()
        bookmark_dict["created_at"] = bookmark.created_at.isoformat() + "Z"

        if include_features:
            if (bookmark.sae_name, bookmark.feature_index) not in features:
                return Response(
                    content=f"Feature {bookmark.feature_index} not found in SAE {bookmark.sae_name}", status_code=404
                )
            bookmark_dict["feature"] = make_serializable(features[(bookmark.sae_name, bookmark.feature_index)])

        bookmark_data.append(bookmark_dict)

    return {
        "bookmarks": bookmark_data,
        "total_count": count,
    }
