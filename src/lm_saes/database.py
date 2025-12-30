from datetime import datetime
from typing import Annotated, Any, Literal, Optional, Union

import gridfs
import numpy as np
import pymongo
import pymongo.database
import pymongo.errors
from bson import ObjectId
from pydantic import BaseModel, Field
from tqdm import tqdm

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
    samplings: list[FeatureAnalysisSampling] = []


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


class SAESetRecord(BaseModel):
    name: str
    sae_series: str
    sae_names: list[str]


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


class CircuitConfig(BaseModel):
    """Configuration used to generate a circuit graph."""

    desired_logit_prob: float = 0.98
    max_feature_nodes: int = 256
    qk_tracing_topk: int = 10
    node_threshold: float = 0.8
    edge_threshold: float = 0.98
    max_n_logits: int = 1


class CircuitTextInput(BaseModel):
    input_type: Literal["plain_text"] = "plain_text"
    text: str


class CircuitChatTemplateInput(BaseModel):
    input_type: Literal["chat_template"] = "chat_template"
    messages: list[dict[str, str]]


CircuitInput = Annotated[Union[CircuitTextInput, CircuitChatTemplateInput], Field(discriminator="input_type")]


class CircuitRecord(BaseModel):
    """Record for a generated circuit graph."""

    id: str
    name: Optional[str] = None
    """Optional custom name/ID for the circuit."""
    sae_set_name: str
    sae_series: str
    prompt: str
    """The prompt used to generate the circuit."""
    input: CircuitInput
    config: CircuitConfig
    graph_data: dict[str, Any]
    """The serialized graph data."""
    created_at: datetime


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
        self.sae_set_collection = self.db["sae_sets"]
        self.circuit_collection = self.db["circuits"]
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
        self.sae_set_collection.create_index([("name", pymongo.ASCENDING)], unique=True)
        self.sae_set_collection.create_index([("sae_series", pymongo.ASCENDING)])
        self.bookmark_collection.create_index(
            [("sae_name", pymongo.ASCENDING), ("sae_series", pymongo.ASCENDING), ("feature_index", pymongo.ASCENDING)],
            unique=True,
        )
        self.bookmark_collection.create_index([("created_at", pymongo.DESCENDING)])
        self.circuit_collection.create_index([("sae_series", pymongo.ASCENDING)])
        self.circuit_collection.create_index([("created_at", pymongo.DESCENDING)])

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

    def create_sae(self, name: str, series: str, path: str, cfg: BaseSAEConfig, model_name: str | None = None):
        inserted_id = self.sae_collection.insert_one(
            {"name": name, "series": series, "path": path, "cfg": cfg.model_dump(), "model_name": model_name}
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

    def list_features(
        self, sae_name: str, sae_series: str | None, indices: list[int], with_samplings: bool = True
    ) -> list[FeatureRecord]:
        projection = {"analyses.samplings": 0} if not with_samplings else None

        features = self.feature_collection.find(
            {"sae_name": sae_name, "sae_series": sae_series, "index": {"$in": indices}}, projection=projection
        ).sort("index", pymongo.ASCENDING)

        if self.is_gridfs_enabled():
            features = [self._from_gridfs(feature) for feature in features]

        return [FeatureRecord.model_validate(feature) for feature in features]

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

    def get_sae_model_name(self, sae_name: str, sae_series: str) -> Optional[str]:
        sae = self.sae_collection.find_one({"name": sae_name, "series": sae_series})
        if sae is None:
            return None
        return sae["model_name"]

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
        for i, feature_analysis in enumerate(tqdm(analysis, desc="Adding feature analyses to MongoDB...")):
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
        for i, feature_update in enumerate(tqdm(update_data, desc="Updating features in MongoDB...")):
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
        """List bookmarks with optional filtering."""
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
        """Update an existing bookmark."""
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
        """Get the total count of bookmarks with optional filtering."""
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

    def add_sae_set(self, name: str, sae_series: str, sae_names: list[str]):
        self.sae_set_collection.insert_one({"name": name, "sae_series": sae_series, "sae_names": sae_names})

    def get_sae_set(self, name: str) -> Optional[SAESetRecord]:
        sae_set = self.sae_set_collection.find_one({"name": name})
        if sae_set is None:
            return None
        return SAESetRecord.model_validate(sae_set)

    def list_sae_sets(self) -> list[SAESetRecord]:
        return [SAESetRecord.model_validate(sae_set) for sae_set in self.sae_set_collection.find()]

    def create_circuit(
        self,
        sae_set_name: str,
        sae_series: str,
        prompt: str,
        input: CircuitInput,
        config: CircuitConfig,
        graph_data: dict[str, Any],
        name: Optional[str] = None,
    ) -> str:
        """Create a new circuit graph record."""
        circuit_data = {
            "name": name,
            "sae_set_name": sae_set_name,
            "sae_series": sae_series,
            "prompt": prompt,
            "input": input.model_dump(),
            "config": config.model_dump(),
            "graph_data": graph_data,
            "created_at": datetime.utcnow(),
        }
        result = self.circuit_collection.insert_one(circuit_data)
        return str(result.inserted_id)

    def get_circuit(self, circuit_id: str) -> Optional[CircuitRecord]:
        """Get a circuit by its ID."""
        try:
            circuit = self.circuit_collection.find_one({"_id": ObjectId(circuit_id)})
        except Exception:
            return None
        if circuit is None:
            return None
        circuit["id"] = str(circuit.pop("_id"))
        return CircuitRecord.model_validate(circuit)

    def list_circuits(
        self,
        sae_series: Optional[str] = None,
        limit: Optional[int] = None,
        skip: int = 0,
        with_graph_data: bool = False,
    ) -> list[dict[str, Any]]:
        """List circuits with optional filtering."""
        query: dict[str, Any] = {}
        if sae_series is not None:
            query["sae_series"] = sae_series

        projection = {"graph_data": 0} if not with_graph_data else None

        cursor = self.circuit_collection.find(query, projection=projection).sort("created_at", pymongo.DESCENDING)

        if skip > 0:
            cursor = cursor.skip(skip)
        if limit is not None:
            cursor = cursor.limit(limit)

        circuits = []
        for circuit in cursor:
            circuit["id"] = str(circuit.pop("_id"))
            circuits.append(circuit)
        return circuits

    def delete_circuit(self, circuit_id: str) -> bool:
        """Delete a circuit by its ID."""
        try:
            result = self.circuit_collection.delete_one({"_id": ObjectId(circuit_id)})
        except Exception:
            return False
        return result.deleted_count > 0
