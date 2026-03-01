from datetime import datetime
from typing import Any, Optional

import gridfs
import numpy as np
import pymongo
import pymongo.database
import pymongo.errors
from bson import ObjectId
from pydantic import BaseModel

from lm_saes.config import (
    BaseSAEConfig,
    DatasetConfig,
    LanguageModelConfig,
    MongoDBConfig,
    SAEConfig,
)

from .utils.bytes import bytes_to_np, np_to_bytes


class DatasetRecord(BaseModel):
    name: str
    cfg: DatasetConfig


class FeatureAnalysisSampling(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    # feature_acts: list[list[float]]
    name: str
    feature_acts_indices: np.ndarray
    feature_acts_values: np.ndarray
    z_pattern_indices: np.ndarray | None = None
    z_pattern_values: np.ndarray | None = None
    dataset_name: list[str]
    shard_idx: np.ndarray | None = None
    n_shards: np.ndarray | None = None
    context_idx: np.ndarray
    model_name: list[str]


class FeatureAnalysis(BaseModel):
    name: str
    act_times: int
    max_feature_acts: float
    decoder_norms: Optional[list[float]] = None
    decoder_similarity_matrices: Optional[list[list[float]]] = None
    decoder_inner_product_matrices: Optional[list[list[float]]] = None
    n_analyzed_tokens: Optional[int] = None
    act_times_modalities: Optional[dict[str, float]] = None
    max_feature_acts_modalities: Optional[dict[str, float]] = None
    samplings: list[FeatureAnalysisSampling]


class FeatureRecord(BaseModel):
    sae_name: str
    sae_series: str
    index: int
    analyses: list[FeatureAnalysis] = []
    logits: Optional[dict[str, list[dict[str, Any]]]] = None
    interpretation: Optional[dict[str, Any]] = None
    metric: Optional[dict[str, float]] = None


class AnalysisRecord(BaseModel):
    name: str
    sae_name: str
    sae_series: str


class ModelRecord(BaseModel):
    name: str
    cfg: LanguageModelConfig


class SAERecord(BaseModel):
    name: str
    series: str
    path: str
    cfg: SAEConfig  # TODO: add more variants of SAEConfig


class BookmarkRecord(BaseModel):
    """Record for bookmarked features.

    Attributes:
        sae_name: Name of the SAE model
        sae_series: Series of the SAE model
        feature_index: Index of the bookmarked feature
        created_at: Timestamp when the bookmark was created
        tags: Optional list of tags for categorizing bookmarks
        notes: Optional user notes about the bookmark
    """

    sae_name: str
    sae_series: str
    feature_index: int
    created_at: datetime
    tags: list[str] = []
    notes: Optional[str] = None


class CircuitAnnotationRecord(BaseModel):
    """Record for circuit annotations.

    Attributes:
        circuit_id: Unique identifier for the circuit annotation
        circuit_interpretation: Interpretation text for the entire circuit
        sae_combo_id: SAE combination ID associated with this circuit
        features: List of features in the circuit, each containing:
            - sae_name: Name of the SAE
            - sae_series: Series of the SAE
            - layer: Layer number (actual layer in the model)
            - feature_index: Index of the feature
            - feature_type: Type of feature ("transcoder" or "lorsa")
            - interpretation: Interpretation text for this feature
            - level: Optional circuit level (independent of layer, for visualization)
            - feature_id: Optional unique identifier for this feature in the circuit
        edges: List of edges between features, each containing:
            - source_feature_id: ID of the source feature
            - target_feature_id: ID of the target feature
            - weight: Weight of the edge
            - interpretation: Optional interpretation text for this edge
        created_at: Timestamp when the circuit annotation was created
        updated_at: Timestamp when the circuit annotation was last updated
        metadata: Optional metadata dictionary for additional information
    """

    circuit_id: str
    circuit_interpretation: str
    sae_combo_id: str
    features: list[dict[str, Any]]
    edges: list[dict[str, Any]] = []
    created_at: datetime
    updated_at: datetime
    metadata: Optional[dict[str, Any]] = None


class MongoClient:
    def __init__(self, cfg: MongoDBConfig):
        self.client: pymongo.MongoClient = pymongo.MongoClient(cfg.mongo_uri)
        self.db = self.client[cfg.mongo_db]
        self.fs: gridfs.GridFS | None = None
        self.feature_collection = self.db["features"]
        self.sae_collection = self.db["saes"]
        self.analysis_collection = self.db["analyses"]
        self.dataset_collection = self.db["datasets"]
        self.model_collection = self.db["models"]
        self.bookmark_collection = self.db["bookmarks"]
        self.circuit_annotations_collection = self.db["circuit_annotations"]
        self.sae_collection.create_index([("name", pymongo.ASCENDING), ("series", pymongo.ASCENDING)], unique=True)
        self.sae_collection.create_index([("series", pymongo.ASCENDING)])
        self.analysis_collection.create_index(
            [("name", pymongo.ASCENDING), ("sae_name", pymongo.ASCENDING), ("sae_series", pymongo.ASCENDING)],
            unique=True,
        )
        self.feature_collection.create_index(
            [("sae_name", pymongo.ASCENDING), ("sae_series", pymongo.ASCENDING), ("index", pymongo.ASCENDING)],
            unique=True,
        )
        self.dataset_collection.create_index([("name", pymongo.ASCENDING)], unique=True)
        self.model_collection.create_index([("name", pymongo.ASCENDING)], unique=True)
        self.bookmark_collection.create_index(
            [("sae_name", pymongo.ASCENDING), ("sae_series", pymongo.ASCENDING), ("feature_index", pymongo.ASCENDING)],
            unique=True,
        )
        self.bookmark_collection.create_index([("created_at", pymongo.DESCENDING)])
        # Circuit annotations indexes
        self.circuit_annotations_collection.create_index([("circuit_id", pymongo.ASCENDING)], unique=True)
        self.circuit_annotations_collection.create_index([("sae_combo_id", pymongo.ASCENDING)])
        self.circuit_annotations_collection.create_index(
            [
                ("features.sae_name", pymongo.ASCENDING),
                ("features.layer", pymongo.ASCENDING),
                ("features.feature_index", pymongo.ASCENDING),
            ]
        )
        self.circuit_annotations_collection.create_index([("created_at", pymongo.DESCENDING)])
        self.circuit_annotations_collection.create_index([("updated_at", pymongo.DESCENDING)])

        # Initialize GridFS by default
        self._init_fs()

    def _init_fs(self):
        """Initialize GridFS for storing large binary data."""
        self.fs = gridfs.GridFS(self.db)

    def enable_gridfs(self) -> None:
        """Enable GridFS for storing large binary data."""
        if self.fs is None:
            self._init_fs()

    def disable_gridfs(self) -> None:
        """Disable GridFS usage."""
        self.fs = None

    def is_gridfs_enabled(self) -> bool:
        """Check if GridFS is enabled."""
        return self.fs is not None

    def _to_gridfs(self, data: Any) -> Any:
        """
        Recursively convert numpy arrays in data object to bytes, and store in GridFS
        """
        assert self.fs is not None, "GridFS is not initialized"
        if isinstance(data, dict):
            return {k: self._to_gridfs(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._to_gridfs(v) for v in data]
        if isinstance(data, np.ndarray):
            return self.fs.put(np_to_bytes(data))
        return data

    def _from_gridfs(self, data: Any) -> Any:
        """
        Recursively convert GridFS references in data object to numpy arrays
        """
        assert self.fs is not None, "GridFS is not initialized"
        if isinstance(data, dict):
            return {k: self._from_gridfs(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._from_gridfs(v) for v in data]
        if isinstance(data, ObjectId) and self.fs.exists(data):
            return bytes_to_np(self.fs.get(data).read())
        return data

    def _remove_gridfs_objs(self, data: Any) -> None:
        """
        Recursively remove GridFS objects in data object
        """
        assert self.fs is not None, "GridFS is not initialized"
        if isinstance(data, dict):
            for v in data.values():
                self._remove_gridfs_objs(v)
        if isinstance(data, list):
            for v in data:
                self._remove_gridfs_objs(v)
        if isinstance(data, ObjectId) and self.fs.exists(data):
            self.fs.delete(data)

    def create_sae(self, name: str, series: str, path: str, cfg: BaseSAEConfig):
        inserted_id = self.sae_collection.insert_one(
            {"name": name, "series": series, "path": path, "cfg": cfg.model_dump()}
        ).inserted_id
        self.feature_collection.insert_many(
            [{"sae_name": name, "sae_series": series, "index": i} for i in range(cfg.d_sae)]
        )
        return inserted_id

    def create_analysis(self, name: str, sae_name: str, sae_series: str):
        return self.analysis_collection.insert_one(
            {"name": name, "sae_name": sae_name, "sae_series": sae_series}
        ).inserted_id

    def remove_sae(self, sae_name: str, sae_series: str | None = None):
        self.analysis_collection.delete_many({"sae_name": sae_name, "sae_series": sae_series})
        self.feature_collection.delete_many({"sae_name": sae_name, "sae_series": sae_series})
        self.sae_collection.delete_one({"name": sae_name, "series": sae_series})

    def list_analyses(self, sae_name: str, sae_series: str | None = None) -> list[str]:
        return [
            d["name"]
            for d in self.analysis_collection.find(
                {"sae_name": sae_name, "sae_series": sae_series} if sae_series is not None else {}
            )
        ]

    def list_saes(self, sae_series: str | None = None, has_analyses: bool = False) -> list[str]:
        sae_names = [
            d["name"] for d in self.sae_collection.find({"series": sae_series} if sae_series is not None else {})
        ]
        if has_analyses:
            sae_names = [
                d["sae_name"]
                for d in self.analysis_collection.find({"sae_series": sae_series} if sae_series is not None else {})
            ]
            sae_names = list(set(sae_names))
        return sae_names

    def get_dataset(self, name: str) -> Optional[DatasetRecord]:
        dataset = self.dataset_collection.find_one({"name": name})
        if dataset is None:
            return None
        return DatasetRecord.model_validate(dataset)

    def get_feature(self, sae_name: str, sae_series: str | None, index: int) -> Optional[FeatureRecord]:
        feature = self.feature_collection.find_one({"sae_name": sae_name, "sae_series": sae_series, "index": index})
        if feature is None:
            return None

        # Convert GridFS references back to numpy arrays
        if self.is_gridfs_enabled():
            feature = self._from_gridfs(feature)

        return FeatureRecord.model_validate(feature)

    def get_analysis(self, name: str, sae_name: str, sae_series: str) -> Optional[AnalysisRecord]:
        analysis = self.analysis_collection.find_one({"name": name, "sae_name": sae_name, "sae_series": sae_series})
        if analysis is None:
            return None
        return AnalysisRecord.model_validate(analysis)

    def get_sae(self, sae_name: str, sae_series: str) -> Optional[SAERecord]:
        sae = self.sae_collection.find_one({"name": sae_name, "series": sae_series})
        if sae is None:
            return None
        return SAERecord.model_validate(sae)

    def get_random_alive_feature(
        self,
        sae_name: str,
        sae_series: str,
        name: str | None = None,
        metric_filters: Optional[dict[str, dict[str, float]]] = None,
    ) -> Optional[FeatureRecord]:
        """Get a random feature that has non-zero activation.

        Args:
            sae_name: Name of the SAE model
            sae_series: Series of the SAE model
            name: Name of the analysis
            metric_filters: Optional dict of metric filters in the format {"metric_name": {"$gte": value, "$lte": value}}

        Returns:
            A random feature record with non-zero activation, or None if no such feature exists
        """
        elem_match: dict[str, Any] = {"max_feature_acts": {"$gt": 0}}
        if name is not None:
            elem_match["name"] = name

        match_filter: dict[str, Any] = {
            "sae_name": sae_name,
            "sae_series": sae_series,
            "analyses": {"$elemMatch": elem_match},
        }

        # Add metric filters if provided
        if metric_filters:
            for metric_name, filters in metric_filters.items():
                match_filter[f"metric.{metric_name}"] = filters

        pipeline = [
            {"$match": match_filter},
            {"$sample": {"size": 1}},
        ]
        feature = next(self.feature_collection.aggregate(pipeline), None)
        if feature is None:
            return None

        # Convert GridFS references back to numpy arrays
        if self.is_gridfs_enabled():
            feature = self._from_gridfs(feature)

        return FeatureRecord.model_validate(feature)

    def get_alive_feature_count(self, sae_name: str, sae_series: str, name: str = "default"):
        pipeline = [
            {"$unwind": "$analyses"},
            {
                "$match": {
                    "sae_name": sae_name,
                    "sae_series": sae_series,
                    "analyses.name": name,
                    "analyses.max_feature_acts": {"$gt": 0},
                }
            },
            {"$count": "count"},
        ]
        return self.feature_collection.aggregate(pipeline).next()["count"]

    def get_max_feature_acts(self, sae_name: str, sae_series: str, name: str = "default") -> dict[int, int] | None:
        pipeline = [
            {"$unwind": "$analyses"},
            {
                "$match": {
                    "sae_name": sae_name,
                    "sae_series": sae_series,
                    "analyses.name": name,
                    "analyses.max_feature_acts": {"$gt": 0},
                }
            },
            {"$project": {"_id": 0, "index": 1, "max_feature_acts": "$analyses.max_feature_acts"}},
        ]
        return {f["index"]: f["max_feature_acts"] for f in self.feature_collection.aggregate(pipeline)}

    def get_feature_act_times(self, sae_name: str, sae_series: str, name: str = "default"):
        pipeline = [
            {"$unwind": "$analyses"},
            {
                "$match": {
                    "sae_name": sae_name,
                    "sae_series": sae_series,
                    "analyses.name": name,
                    "analyses.act_times": {"$gt": 0},
                }
            },
            {"$project": {"_id": 0, "index": 1, "act_times": "$analyses.act_times"}},
        ]
        return {f["index"]: f["act_times"] for f in self.feature_collection.aggregate(pipeline)}

    def get_sae_path(self, sae_name: str, sae_series: str):
        sae = self.sae_collection.find_one({"name": sae_name, "series": sae_series})
        if sae is None:
            return None
        return sae["path"]

    def add_dataset(self, name: str, cfg: DatasetConfig):
        self.dataset_collection.update_one({"name": name}, {"$set": {"cfg": cfg.model_dump()}}, upsert=True)

    def get_dataset_cfg(self, name: str) -> Optional[DatasetConfig]:
        dataset = self.dataset_collection.find_one({"name": name})
        if dataset is None:
            return None
        return DatasetConfig.model_validate(dataset["cfg"])

    def add_model(self, name: str, cfg: LanguageModelConfig):
        self.model_collection.update_one({"name": name}, {"$set": {"cfg": cfg.model_dump()}}, upsert=True)

    def get_model_cfg(self, name: str) -> Optional[LanguageModelConfig]:
        model = self.model_collection.find_one({"name": name})
        if model is None:
            return None
        return LanguageModelConfig.model_validate(model["cfg"])

    def add_feature_analysis(self, name: str, sae_name: str, sae_series: str, analysis: list[dict], start_idx: int = 0):
        # Initialize GridFS if not already done
        if not self.is_gridfs_enabled():
            self.enable_gridfs()

        operations = []
        for i, feature_analysis in enumerate(analysis):
            # Convert numpy arrays to GridFS references
            processed_analysis = self._to_gridfs(feature_analysis)
            update_operation = pymongo.UpdateOne(
                {"sae_name": sae_name, "sae_series": sae_series, "index": start_idx + i},
                {"$push": {"analyses": processed_analysis | {"name": name}}},
                upsert=True,
            )
            operations.append(update_operation)

        if operations:
            self.feature_collection.bulk_write(operations)

        if start_idx == 0:
            self.analysis_collection.insert_one({"name": name, "sae_name": sae_name, "sae_series": sae_series})

    def remove_feature_analysis(self, name: str, sae_name: str, sae_series: str):
        self.feature_collection.update_many(
            {"sae_name": sae_name, "sae_series": sae_series}, {"$pull": {"analyses": {"name": name}}}
        )
        self.analysis_collection.delete_one({"name": name, "sae_name": sae_name, "sae_series": sae_series})

    def remove_sae_analysis(self, sae_name: str, sae_series: str):
        self.feature_collection.delete_many({"sae_name": sae_name, "sae_series": sae_series})
        self.analysis_collection.delete_many({"sae_name": sae_name, "sae_series": sae_series})
        self.sae_collection.delete_one({"name": sae_name, "series": sae_series})

    def update_feature(self, sae_name: str, feature_index: int, update_data: dict, sae_series: str | None = None):
        """Update a feature with additional data.

        Args:
            sae_name: Name of the SAE
            feature_index: Index of the feature to update
            update_data: Dictionary with data to update
            sae_series: Optional series of the SAE

        Returns:
            Result of the update operation

        Raises:
            ValueError: If the feature doesn't exist
        """
        # Ensure we have a non-None sae_series
        if sae_series is None:
            raise ValueError("sae_series cannot be None")

        feature = self.get_feature(sae_name, sae_series, feature_index)
        if feature is None:
            raise ValueError(f"Feature {feature_index} not found for SAE {sae_name}/{sae_series}")

        # Initialize GridFS if not already done
        if not self.is_gridfs_enabled():
            self.enable_gridfs()

        # Convert numpy arrays to GridFS references
        processed_update_data = self._to_gridfs(update_data)

        result = self.feature_collection.update_one(
            {"sae_name": sae_name, "sae_series": sae_series, "index": feature_index}, {"$set": processed_update_data}
        )

        return result

    def update_features(self, sae_name: str, sae_series: str, update_data: list[dict], start_idx: int = 0):
        operations = []
        for i, feature_update in enumerate(update_data):
            update_operation = pymongo.UpdateOne(
                {"sae_name": sae_name, "sae_series": sae_series, "index": start_idx + i},
                {"$set": feature_update},
            )
            operations.append(update_operation)
        if operations:
            self.feature_collection.bulk_write(operations)

    def add_bookmark(
        self,
        sae_name: str,
        sae_series: str,
        feature_index: int,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Add a bookmark for a feature.

        Args:
            sae_name: Name of the SAE
            sae_series: Series of the SAE
            feature_index: Index of the feature to bookmark
            tags: Optional list of tags for the bookmark
            notes: Optional notes for the bookmark

        Returns:
            bool: True if bookmark was added, False if it already exists

        Raises:
            ValueError: If the feature doesn't exist
        """
        # Check if feature exists
        feature = self.get_feature(sae_name, sae_series, feature_index)
        if feature is None:
            raise ValueError(f"Feature {feature_index} not found for SAE {sae_name}/{sae_series}")

        bookmark_data = {
            "sae_name": sae_name,
            "sae_series": sae_series,
            "feature_index": feature_index,
            "created_at": datetime.utcnow(),
            "tags": tags or [],
            "notes": notes,
        }

        try:
            result = self.bookmark_collection.insert_one(bookmark_data)
            return result.inserted_id is not None
        except pymongo.errors.DuplicateKeyError:
            return False

    def remove_bookmark(self, sae_name: str, sae_series: str, feature_index: int) -> bool:
        """Remove a bookmark for a feature.

        Args:
            sae_name: Name of the SAE
            sae_series: Series of the SAE
            feature_index: Index of the feature to remove bookmark from

        Returns:
            bool: True if bookmark was removed, False if it didn't exist
        """
        result = self.bookmark_collection.delete_one(
            {
                "sae_name": sae_name,
                "sae_series": sae_series,
                "feature_index": feature_index,
            }
        )
        return result.deleted_count > 0

    def is_bookmarked(self, sae_name: str, sae_series: str, feature_index: int) -> bool:
        """Check if a feature is bookmarked.

        Args:
            sae_name: Name of the SAE
            sae_series: Series of the SAE
            feature_index: Index of the feature

        Returns:
            bool: True if the feature is bookmarked, False otherwise
        """
        bookmark = self.bookmark_collection.find_one(
            {
                "sae_name": sae_name,
                "sae_series": sae_series,
                "feature_index": feature_index,
            }
        )
        return bookmark is not None

    def get_bookmark(self, sae_name: str, sae_series: str, feature_index: int) -> Optional[BookmarkRecord]:
        """Get a specific bookmark.

        Args:
            sae_name: Name of the SAE
            sae_series: Series of the SAE
            feature_index: Index of the feature

        Returns:
            BookmarkRecord: The bookmark record if it exists, None otherwise
        """
        bookmark = self.bookmark_collection.find_one(
            {
                "sae_name": sae_name,
                "sae_series": sae_series,
                "feature_index": feature_index,
            }
        )
        if bookmark is None:
            return None
        return BookmarkRecord.model_validate(bookmark)

    def list_bookmarks(
        self,
        sae_name: Optional[str] = None,
        sae_series: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: Optional[int] = None,
        skip: int = 0,
    ) -> list[BookmarkRecord]:
        """List bookmarks with optional filtering.

        Args:
            sae_name: Optional SAE name filter
            sae_series: Optional SAE series filter
            tags: Optional list of tags to filter by (matches any tag)
            limit: Optional limit on number of results
            skip: Number of results to skip (for pagination)

        Returns:
            list[BookmarkRecord]: List of bookmark records
        """
        query = {}

        if sae_name is not None:
            query["sae_name"] = sae_name
        if sae_series is not None:
            query["sae_series"] = sae_series
        if tags:
            query["tags"] = {"$in": tags}

        cursor = self.bookmark_collection.find(query).sort("created_at", pymongo.DESCENDING)

        if skip > 0:
            cursor = cursor.skip(skip)
        if limit is not None:
            cursor = cursor.limit(limit)

        return [BookmarkRecord.model_validate(bookmark) for bookmark in cursor]

    def update_bookmark(
        self,
        sae_name: str,
        sae_series: str,
        feature_index: int,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Update an existing bookmark.

        Args:
            sae_name: Name of the SAE
            sae_series: Series of the SAE
            feature_index: Index of the feature
            tags: Optional new tags for the bookmark
            notes: Optional new notes for the bookmark

        Returns:
            bool: True if bookmark was updated, False if it doesn't exist
        """
        update_data = {}
        if tags is not None:
            update_data["tags"] = tags
        if notes is not None:
            update_data["notes"] = notes

        if not update_data:
            return True  # Nothing to update

        result = self.bookmark_collection.update_one(
            {
                "sae_name": sae_name,
                "sae_series": sae_series,
                "feature_index": feature_index,
            },
            {"$set": update_data},
        )
        return result.modified_count > 0

    def get_bookmark_count(self, sae_name: Optional[str] = None, sae_series: Optional[str] = None) -> int:
        """Get the total count of bookmarks with optional filtering.

        Args:
            sae_name: Optional SAE name filter
            sae_series: Optional SAE series filter

        Returns:
            int: Total number of bookmarks matching the criteria
        """
        query = {}
        if sae_name is not None:
            query["sae_name"] = sae_name
        if sae_series is not None:
            query["sae_series"] = sae_series

        return self.bookmark_collection.count_documents(query)

    def get_available_metrics(self, sae_name: str, sae_series: str) -> list[str]:
        """Get available metrics for an SAE by checking the first feature.

        Args:
            sae_name: Name of the SAE model
            sae_series: Series of the SAE model

        Returns:
            List of available metric names
        """
        # Use projection to avoid loading large arrays from analyses[0].samplings
        projection = {
            "metric": 1,
        }

        first_feature = self.feature_collection.find_one({"sae_name": sae_name, "sae_series": sae_series}, projection)

        if first_feature is None or first_feature.get("metric") is None:
            return []

        return list(first_feature["metric"].keys())

    def count_features_with_filters(
        self,
        sae_name: str,
        sae_series: str,
        name: str | None = None,
        metric_filters: Optional[dict[str, dict[str, float]]] = None,
    ) -> int:
        """Count features that match the given filters.

        Args:
            sae_name: Name of the SAE model
            sae_series: Series of the SAE model
            name: Name of the analysis
            metric_filters: Optional dict of metric filters in the format {"metric_name": {"$gte": value, "$lte": value}}

        Returns:
            Number of features matching the filters
        """
        elem_match: dict[str, Any] = {"max_feature_acts": {"$gt": 0}}
        if name is not None:
            elem_match["name"] = name

        match_filter: dict[str, Any] = {
            "sae_name": sae_name,
            "sae_series": sae_series,
            "analyses": {"$elemMatch": elem_match},
        }

        # Add metric filters if provided
        if metric_filters:
            for metric_name, filters in metric_filters.items():
                match_filter[f"metric.{metric_name}"] = filters

        return self.feature_collection.count_documents(match_filter)

    # ============================================================================
    # Circuit Annotation Methods
    # ============================================================================

    def create_circuit_annotation(
        self,
        circuit_id: str,
        circuit_interpretation: str,
        sae_combo_id: str,
        features: list[dict[str, Any]],
        edges: Optional[list[dict[str, Any]]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Create a new circuit annotation.

        Args:
            circuit_id: Unique identifier for the circuit annotation
            circuit_interpretation: Interpretation text for the entire circuit
            sae_combo_id: SAE combination ID associated with this circuit
            features: List of feature dictionaries, each containing:
                - sae_name: Name of the SAE
                - sae_series: Series of the SAE
                - layer: Layer number (actual layer in the model)
                - feature_index: Index of the feature
                - feature_type: Type of feature ("transcoder" or "lorsa")
                - interpretation: Interpretation text for this feature
                - level: Optional circuit level (independent of layer, for visualization)
                - feature_id: Optional unique identifier for this feature in the circuit
            edges: Optional list of edge dictionaries, each containing:
                - source_feature_id: ID of the source feature
                - target_feature_id: ID of the target feature
                - weight: Weight of the edge
                - interpretation: Optional interpretation text for this edge
            metadata: Optional metadata dictionary

        Returns:
            bool: True if circuit annotation was created, False if it already exists

        Raises:
            ValueError: If any feature in the list doesn't exist
        """
        # Validate that all features exist
        for feature in features:
            sae_name = feature.get("sae_name")
            sae_series = feature.get("sae_series")
            feature_index = feature.get("feature_index")
            if sae_name and sae_series is not None and feature_index is not None:
                existing_feature = self.get_feature(sae_name, sae_series, feature_index)
                if existing_feature is None:
                    raise ValueError(
                        f"Feature {feature_index} not found for SAE {sae_name}/{sae_series}"
                    )
        
        # Generate feature_id for each feature if not provided
        import uuid
        feature_id_map = {}
        for i, feature in enumerate(features):
            if "feature_id" not in feature:
                feature["feature_id"] = str(uuid.uuid4())
            feature_id_map[i] = feature["feature_id"]
        
        # Validate edges if provided
        if edges is not None:
            feature_ids = {f.get("feature_id") for f in features if f.get("feature_id")}
            for edge in edges:
                source_id = edge.get("source_feature_id")
                target_id = edge.get("target_feature_id")
                if source_id not in feature_ids:
                    raise ValueError(f"Source feature_id {source_id} not found in features")
                if target_id not in feature_ids:
                    raise ValueError(f"Target feature_id {target_id} not found in features")
                if "weight" not in edge:
                    edge["weight"] = 0.0

        now = datetime.utcnow()
        circuit_data = {
            "circuit_id": circuit_id,
            "circuit_interpretation": circuit_interpretation,
            "sae_combo_id": sae_combo_id,
            "features": features,
            "edges": edges or [],
            "created_at": now,
            "updated_at": now,
            "metadata": metadata or {},
        }

        try:
            result = self.circuit_annotations_collection.insert_one(circuit_data)
            return result.inserted_id is not None
        except pymongo.errors.DuplicateKeyError:
            return False

    def get_circuit_annotation(self, circuit_id: str) -> Optional[CircuitAnnotationRecord]:
        """Get a circuit annotation by ID.

        Args:
            circuit_id: Unique identifier for the circuit annotation

        Returns:
            CircuitAnnotationRecord: The circuit annotation record if it exists, None otherwise
        """
        circuit = self.circuit_annotations_collection.find_one({"circuit_id": circuit_id})
        if circuit is None:
            return None
        return CircuitAnnotationRecord.model_validate(circuit)

    def list_circuit_annotations(
        self,
        sae_combo_id: Optional[str] = None,
        limit: Optional[int] = None,
        skip: int = 0,
    ) -> list[CircuitAnnotationRecord]:
        """List circuit annotations with optional filtering.

        Args:
            sae_combo_id: Optional SAE combination ID filter
            limit: Optional limit on number of results
            skip: Number of results to skip (for pagination)

        Returns:
            list[CircuitAnnotationRecord]: List of circuit annotation records
        """
        query = {}
        if sae_combo_id is not None:
            query["sae_combo_id"] = sae_combo_id

        cursor = self.circuit_annotations_collection.find(query).sort("updated_at", pymongo.DESCENDING)

        if skip > 0:
            cursor = cursor.skip(skip)
        if limit is not None:
            cursor = cursor.limit(limit)

        return [CircuitAnnotationRecord.model_validate(circuit) for circuit in cursor]

    def update_circuit_interpretation(self, circuit_id: str, circuit_interpretation: str) -> bool:
        """Update the interpretation text for a circuit annotation.

        Args:
            circuit_id: Unique identifier for the circuit annotation
            circuit_interpretation: New interpretation text for the entire circuit

        Returns:
            bool: True if circuit annotation was updated, False if it doesn't exist
        """
        result = self.circuit_annotations_collection.update_one(
            {"circuit_id": circuit_id},
            {"$set": {"circuit_interpretation": circuit_interpretation, "updated_at": datetime.utcnow()}},
        )
        return result.modified_count > 0

    def add_feature_to_circuit(
        self,
        circuit_id: str,
        sae_name: str,
        sae_series: str,
        layer: int,
        feature_index: int,
        feature_type: str,
        interpretation: str,
    ) -> bool:
        """Add a feature to a circuit annotation.

        Args:
            circuit_id: Unique identifier for the circuit annotation
            sae_name: Name of the SAE
            sae_series: Series of the SAE
            layer: Layer number
            feature_index: Index of the feature
            feature_type: Type of feature ("transcoder" or "lorsa")
            interpretation: Interpretation text for this feature

        Returns:
            bool: True if feature was added, False if circuit doesn't exist or feature already in circuit

        Raises:
            ValueError: If the feature doesn't exist
        """
        # Validate that the feature exists
        existing_feature = self.get_feature(sae_name, sae_series, feature_index)
        if existing_feature is None:
            raise ValueError(f"Feature {feature_index} not found for SAE {sae_name}/{sae_series}")

        feature_data = {
            "sae_name": sae_name,
            "sae_series": sae_series,
            "layer": layer,
            "feature_index": feature_index,
            "feature_type": feature_type,
            "interpretation": interpretation,
        }

        # Check if feature already exists in circuit
        circuit = self.circuit_annotations_collection.find_one({"circuit_id": circuit_id})
        if circuit is None:
            return False

        # Check for duplicate feature
        existing_features = circuit.get("features", [])
        for feat in existing_features:
            if (
                feat.get("sae_name") == sae_name
                and feat.get("sae_series") == sae_series
                and feat.get("layer") == layer
                and feat.get("feature_index") == feature_index
                and feat.get("feature_type") == feature_type
            ):
                return False  # Feature already in circuit

        result = self.circuit_annotations_collection.update_one(
            {"circuit_id": circuit_id},
            {"$push": {"features": feature_data}, "$set": {"updated_at": datetime.utcnow()}},
        )
        return result.modified_count > 0

    def remove_feature_from_circuit(
        self,
        circuit_id: str,
        sae_name: str,
        sae_series: str,
        layer: int,
        feature_index: int,
        feature_type: str,
    ) -> bool:
        """Remove a feature from a circuit annotation.

        Args:
            circuit_id: Unique identifier for the circuit annotation
            sae_name: Name of the SAE
            sae_series: Series of the SAE
            layer: Layer number
            feature_index: Index of the feature
            feature_type: Type of feature ("transcoder" or "lorsa")

        Returns:
            bool: True if feature was removed, False if circuit doesn't exist or feature not in circuit
        """
        result = self.circuit_annotations_collection.update_one(
            {
                "circuit_id": circuit_id,
                "features": {
                    "$elemMatch": {
                        "sae_name": sae_name,
                        "sae_series": sae_series,
                        "layer": layer,
                        "feature_index": feature_index,
                        "feature_type": feature_type,
                    }
                },
            },
            {
                "$pull": {
                    "features": {
                        "sae_name": sae_name,
                        "sae_series": sae_series,
                        "layer": layer,
                        "feature_index": feature_index,
                        "feature_type": feature_type,
                    }
                },
                "$set": {"updated_at": datetime.utcnow()},
            },
        )
        return result.modified_count > 0

    def update_feature_interpretation_in_circuit(
        self,
        circuit_id: str,
        sae_name: str,
        sae_series: str,
        layer: int,
        feature_index: int,
        feature_type: str,
        interpretation: str,
    ) -> bool:
        """Update the interpretation text for a specific feature in a circuit annotation.

        Args:
            circuit_id: Unique identifier for the circuit annotation
            sae_name: Name of the SAE
            sae_series: Series of the SAE
            layer: Layer number
            feature_index: Index of the feature
            feature_type: Type of feature ("transcoder" or "lorsa")
            interpretation: New interpretation text for this feature

        Returns:
            bool: True if feature interpretation was updated, False if circuit or feature doesn't exist
        """
        result = self.circuit_annotations_collection.update_one(
            {
                "circuit_id": circuit_id,
                "features": {
                    "$elemMatch": {
                        "sae_name": sae_name,
                        "sae_series": sae_series,
                        "layer": layer,
                        "feature_index": feature_index,
                        "feature_type": feature_type,
                    }
                },
            },
            {
                "$set": {
                    "features.$.interpretation": interpretation,
                    "updated_at": datetime.utcnow(),
                }
            },
        )
        return result.modified_count > 0

    def get_circuits_by_feature(
        self,
        sae_name: str,
        sae_series: str,
        layer: int,
        feature_index: int,
        feature_type: Optional[str] = None,
    ) -> list[CircuitAnnotationRecord]:
        """Get all circuit annotations that contain a specific feature.

        Args:
            sae_name: Name of the SAE
            sae_series: Series of the SAE
            layer: Layer number
            feature_index: Index of the feature
            feature_type: Optional type of feature ("transcoder" or "lorsa") to filter by

        Returns:
            list[CircuitAnnotationRecord]: List of circuit annotation records containing the feature
        """
        query: dict[str, Any] = {
            "features": {
                "$elemMatch": {
                    "sae_name": sae_name,
                    "sae_series": sae_series,
                    "layer": layer,
                    "feature_index": feature_index,
                }
            }
        }

        if feature_type is not None:
            query["features"]["$elemMatch"]["feature_type"] = feature_type

        circuits = self.circuit_annotations_collection.find(query).sort("updated_at", pymongo.DESCENDING)
        return [CircuitAnnotationRecord.model_validate(circuit) for circuit in circuits]

    def delete_circuit_annotation(self, circuit_id: str) -> bool:
        """Delete a circuit annotation.

        Args:
            circuit_id: Unique identifier for the circuit annotation

        Returns:
            bool: True if circuit annotation was deleted, False if it doesn't exist
        """
        result = self.circuit_annotations_collection.delete_one({"circuit_id": circuit_id})
        return result.deleted_count > 0

    def get_circuit_annotation_count(self, sae_combo_id: Optional[str] = None) -> int:
        """Get the total count of circuit annotations with optional filtering.

        Args:
            sae_combo_id: Optional SAE combination ID filter

        Returns:
            int: Total number of circuit annotations matching the criteria
        """
        query = {}
        if sae_combo_id is not None:
            query["sae_combo_id"] = sae_combo_id

        return self.circuit_annotations_collection.count_documents(query)

    def add_edge_to_circuit(
        self,
        circuit_id: str,
        source_feature_id: str,
        target_feature_id: str,
        weight: float = 0.0,
        interpretation: Optional[str] = None,
    ) -> bool:
        """Add an edge between two features in a circuit annotation.

        Args:
            circuit_id: Unique identifier for the circuit annotation
            source_feature_id: ID of the source feature
            target_feature_id: ID of the target feature
            weight: Weight of the edge
            interpretation: Optional interpretation text for this edge

        Returns:
            bool: True if edge was added, False if circuit doesn't exist or edge already exists

        Raises:
            ValueError: If source or target feature doesn't exist in the circuit
        """
        circuit = self.circuit_annotations_collection.find_one({"circuit_id": circuit_id})
        if circuit is None:
            return False

        # Validate that both features exist in the circuit
        feature_ids = {f.get("feature_id") for f in circuit.get("features", [])}
        if source_feature_id not in feature_ids:
            raise ValueError(f"Source feature_id {source_feature_id} not found in circuit")
        if target_feature_id not in feature_ids:
            raise ValueError(f"Target feature_id {target_feature_id} not found in circuit")

        # Check if edge already exists
        existing_edges = circuit.get("edges", [])
        for edge in existing_edges:
            if (
                edge.get("source_feature_id") == source_feature_id
                and edge.get("target_feature_id") == target_feature_id
            ):
                return False  # Edge already exists

        edge_data = {
            "source_feature_id": source_feature_id,
            "target_feature_id": target_feature_id,
            "weight": weight,
        }
        if interpretation is not None:
            edge_data["interpretation"] = interpretation

        result = self.circuit_annotations_collection.update_one(
            {"circuit_id": circuit_id},
            {"$push": {"edges": edge_data}, "$set": {"updated_at": datetime.utcnow()}},
        )
        return result.modified_count > 0

    def remove_edge_from_circuit(
        self,
        circuit_id: str,
        source_feature_id: str,
        target_feature_id: str,
    ) -> bool:
        """Remove an edge from a circuit annotation.

        Args:
            circuit_id: Unique identifier for the circuit annotation
            source_feature_id: ID of the source feature
            target_feature_id: ID of the target feature

        Returns:
            bool: True if edge was removed, False if circuit doesn't exist or edge not found
        """
        result = self.circuit_annotations_collection.update_one(
            {"circuit_id": circuit_id},
            {
                "$pull": {
                    "edges": {
                        "source_feature_id": source_feature_id,
                        "target_feature_id": target_feature_id,
                    }
                },
                "$set": {"updated_at": datetime.utcnow()},
            },
        )
        return result.modified_count > 0

    def update_edge_weight(
        self,
        circuit_id: str,
        source_feature_id: str,
        target_feature_id: str,
        weight: float,
        interpretation: Optional[str] = None,
    ) -> bool:
        """Update the weight of an edge in a circuit annotation.

        Args:
            circuit_id: Unique identifier for the circuit annotation
            source_feature_id: ID of the source feature
            target_feature_id: ID of the target feature
            weight: New weight for the edge
            interpretation: Optional new interpretation text for the edge

        Returns:
            bool: True if edge was updated, False if circuit or edge doesn't exist
        """
        circuit = self.circuit_annotations_collection.find_one({"circuit_id": circuit_id})
        if circuit is None:
            return False

        edges = circuit.get("edges", [])
        edge_index = None
        for i, edge in enumerate(edges):
            if (
                edge.get("source_feature_id") == source_feature_id
                and edge.get("target_feature_id") == target_feature_id
            ):
                edge_index = i
                break

        if edge_index is None:
            return False  # Edge not found

        # Update the edge
        update_data = {"edges.$.weight": weight}
        if interpretation is not None:
            update_data["edges.$.interpretation"] = interpretation

        update_data["updated_at"] = datetime.utcnow()
        result = self.circuit_annotations_collection.update_one(
            {
                "circuit_id": circuit_id,
                "edges": {
                    "$elemMatch": {
                        "source_feature_id": source_feature_id,
                        "target_feature_id": target_feature_id,
                    }
                },
            },
            {"$set": update_data},
        )
        return result.modified_count > 0

    def set_feature_level(
        self,
        circuit_id: str,
        feature_id: str,
        level: int,
    ) -> bool:
        """Set the circuit level for a feature in a circuit annotation.

        Args:
            circuit_id: Unique identifier for the circuit annotation
            feature_id: ID of the feature
            level: Circuit level (independent of layer, for visualization)

        Returns:
            bool: True if feature level was updated, False if circuit or feature doesn't exist
        """
        circuit = self.circuit_annotations_collection.find_one({"circuit_id": circuit_id})
        if circuit is None:
            return False

        # Check if feature exists in circuit
        features = circuit.get("features", [])
        feature_index = None
        for i, feature in enumerate(features):
            if feature.get("feature_id") == feature_id:
                feature_index = i
                break

        if feature_index is None:
            return False  # Feature not found

        result = self.circuit_annotations_collection.update_one(
            {
                "circuit_id": circuit_id,
                "features": {"$elemMatch": {"feature_id": feature_id}},
            },
            {"$set": {"features.$.level": level, "updated_at": datetime.utcnow()}},
        )
        return result.modified_count > 0

    def update_feature_level(
        self,
        circuit_id: str,
        feature_id: str,
        level: int,
    ) -> bool:
        """Update the circuit level for a feature in a circuit annotation.
        
        Alias for set_feature_level for consistency.

        Args:
            circuit_id: Unique identifier for the circuit annotation
            feature_id: ID of the feature
            level: Circuit level (independent of layer, for visualization)

        Returns:
            bool: True if feature level was updated, False if circuit or feature doesn't exist
        """
        return self.set_feature_level(circuit_id, feature_id, level)
