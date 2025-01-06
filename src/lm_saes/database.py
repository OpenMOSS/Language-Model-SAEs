from typing import Any, Optional

import gridfs
import numpy as np
import pymongo
import pymongo.database
from bson import ObjectId
from pydantic import BaseModel
from typing_extensions import deprecated

from lm_saes.config import BaseSAEConfig, DatasetConfig, LanguageModelConfig, SAEConfig

from .utils.bytes import bytes_to_np, np_to_bytes


class DatasetRecord(BaseModel):
    name: str
    cfg: DatasetConfig


class FeatureAnalysisSampling(BaseModel):
    name: str
    feature_acts: list[float]
    dataset_ids: list[int]
    shard_ids: Optional[list[int]] = None
    n_shards: Optional[list[int]] = None
    context_ids: list[int]


class FeatureAnalysis(BaseModel):
    name: str
    act_times: int
    max_feature_act: int
    samplings: list[FeatureAnalysisSampling]


class FeatureRecord(BaseModel):
    sae_name: str
    sae_series: str
    index: int
    analyses: list[FeatureAnalysis] = []


class AnalysisRecord(BaseModel):
    name: str
    sae_name: str
    sae_series: str


class ModelRecord(BaseModel):
    name: str
    series: str
    cfg: LanguageModelConfig


class SAERecord(BaseModel):
    name: str
    series: str
    path: str
    cfg: SAEConfig  # TODO: add more variants of SAEConfig


class MongoClient:
    def __init__(self, mongo_uri: str, mongo_db: str):
        self.client: pymongo.MongoClient = pymongo.MongoClient(mongo_uri)
        self.db = self.client[mongo_db]
        self.fs: gridfs.GridFS | None = None
        self.feature_collection = self.db["features"]
        self.sae_collection = self.db["saes"]
        self.analysis_collection = self.db["analyses"]
        self.dataset_collection = self.db["datasets"]
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

    @deprecated("Not recommended for new code, where any single record can fit in 16MB size limit of BSON")
    def _init_fs(self):
        self.fs = gridfs.GridFS(self.db)

    @deprecated("Not recommended for new code, where any single record can fit in 16MB size limit of BSON")
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

    @deprecated("Not recommended for new code, where any single record can fit in 16MB size limit of BSON")
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

    @deprecated("Not recommended for new code, where any single record can fit in 16MB size limit of BSON")
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

    def list_saes(self, sae_series: str | None = None) -> list[str]:
        return [d["name"] for d in self.sae_collection.find({"series": sae_series} if sae_series is not None else {})]

    def get_dataset(self, name: str) -> Optional[DatasetRecord]:
        dataset = self.dataset_collection.find_one({"name": name})
        if dataset is None:
            return None
        return DatasetRecord.model_validate(dataset)

    def get_feature(self, sae_name: str, sae_series: str, index: int) -> Optional[FeatureRecord]:
        feature = self.feature_collection.find_one({"sae_name": sae_name, "sae_series": sae_series, "index": index})
        if feature is None:
            return None
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

    def get_random_alive_feature(self, sae_name: str, sae_series: str):
        feature = self.feature_collection.aggregate(
            [
                {"$match": {"sae_name": sae_name, "sae_series": sae_series, "max_feature_acts": {"$gt": 0}}},
                {"$sample": {"size": 1}},
            ]
        ).next()
        if feature is None:
            return None
        return FeatureRecord.model_validate(**feature)

    def get_alive_feature_count(self, sae_name: str, sae_series: str):
        return self.feature_collection.count_documents(
            {"sae_name": sae_name, "sae_series": sae_series, "max_feature_acts": {"$gt": 0}}
        )

    def get_max_feature_acts(self, sae_name: str, sae_series: str) -> dict[int, int] | None:
        pipeline = [
            {"$match": {"sae_name": sae_name, "sae_series": sae_series, "max_feature_acts": {"$gt": 0}}},
            {"$project": {"_id": 0, "index": 1, "max_feature_acts": 1}},
        ]
        return {f["index"]: f["max_feature_acts"] for f in self.feature_collection.aggregate(pipeline)}

    def get_feature_act_times(self, sae_name: str, sae_series: str):
        pipeline = [
            {"$match": {"sae_name": sae_name, "sae_series": sae_series, "max_feature_acts": {"$gt": 0}}},
            {"$project": {"_id": 0, "index": 1, "act_times": 1}},
        ]
        return {f["index"]: f["act_times"] for f in self.feature_collection.aggregate(pipeline)}

    def get_sae_path(self, sae_name: str, sae_series: str):
        sae = self.sae_collection.find_one({"name": sae_name, "series": sae_series})
        if sae is None:
            return None
        return sae["path"]
