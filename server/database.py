import os
from typing import Dict

import pymongo
import gridfs

from core.utils.bytes import bytes_to_np

mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_db = os.getenv("MONGO_DB", "mechinterp")
result_dir = os.getenv("RESULT_DIR", "results")

client = pymongo.MongoClient(mongo_uri)
db = client[mongo_db]
fs = gridfs.GridFS(db)
feature_collection = db["features"]
dictionary_collection = db["dictionaries"]
attn_head_collection = db["attn_heads"]
dictionary_collection.create_index("name", unique=True)
feature_collection.create_index(
    [("dictionary_id", pymongo.ASCENDING), ("index", pymongo.ASCENDING)], unique=True
)
attn_head_collection.create_index(
    [("dictionary_id", pymongo.ASCENDING), ("index", pymongo.ASCENDING)], unique=True
)


def list_dictionaries():
    return [d["name"] for d in dictionary_collection.find()]


def get_feature(dictionary_name: str, feature_index: int):
    dictionary = dictionary_collection.find_one({"name": dictionary_name})
    if dictionary is None:
        return None
    feature = feature_collection.find_one(
        {"dictionary_id": dictionary["_id"], "index": feature_index}
    )
    if feature is None:
        return None
    return {
        "index": feature["index"],
        "act_times": feature["act_times"],
        "max_feature_acts": feature["max_feature_acts"],
        "feature_acts_all": bytes_to_np(fs.get(feature["feature_acts_all"]).read()),
        "analysis": [
            {
                "name": analysis["name"],
                "feature_acts": bytes_to_np(fs.get(analysis["feature_acts"]).read()),
                "contexts": bytes_to_np(fs.get(analysis["contexts"]).read()),
            }
            for analysis in feature["analysis"]
        ],
        "interpretation": feature["interpretation"],
    }


def get_random_alive_feature(dictionary_name: str):
    dictionary = dictionary_collection.find_one({"name": dictionary_name})
    if dictionary is None:
        return None
    feature = feature_collection.aggregate(
        [
            {
                "$match": {
                    "dictionary_id": dictionary["_id"],
                    "max_feature_acts": {"$gt": 0},
                }
            },
            {"$sample": {"size": 1}},
        ]
    ).next()
    if feature is None:
        return None
    return {
        "index": feature["index"],
        "act_times": feature["act_times"],
        "max_feature_acts": feature["max_feature_acts"],
        "feature_acts_all": bytes_to_np(fs.get(feature["feature_acts_all"]).read()),
        "analysis": [
            {
                "name": analysis["name"],
                "feature_acts": bytes_to_np(fs.get(analysis["feature_acts"]).read()),
                "contexts": bytes_to_np(fs.get(analysis["contexts"]).read()),
            }
            for analysis in feature["analysis"]
        ],
        "interpretation": feature["interpretation"],
    }


def get_alive_feature_count(dictionary_name: str):
    dictionary = dictionary_collection.find_one({"name": dictionary_name})
    if dictionary is None:
        return None
    return feature_collection.count_documents(
        {"dictionary_id": dictionary["_id"], "max_feature_acts": {"$gt": 0}}
    )


def update_feature_interpretation(
    dictionary_name: str, feature_index: int, interpretation: Dict
):
    dictionary = dictionary_collection.find_one({"name": dictionary_name})
    assert dictionary is not None, f"Dictionary {dictionary_name} not found"
    feature = feature_collection.find_one(
        {"dictionary_id": dictionary["_id"], "index": feature_index}
    )
    assert (
        feature is not None
    ), f"Feature {feature_index} not found in dictionary {dictionary_name}"
    feature_collection.update_one(
        {"_id": feature["_id"]}, {"$set": {"interpretation": interpretation}}
    )


def get_attn_head(layer: int, head_index: int):
    if layer == 0:
        pipeline = [
            {
                "$lookup": {
                    "localField": "dictionary_id",
                    "from": "dictionaries",
                    "foreignField": "_id",
                    "as": "dictionary",
                }
            },
            {"$unwind": "$dictionary"},
            {
                "$match": {
                    "dictionary.name": f"L{layer}A-l1-0.00012-lr-0.001-32x",
                    "index": head_index,
                }
            },
            {
                "$project": {
                    "dictionary_id": 0,
                }
            }
        ]
    else:
        pipeline = [
            {
                "$lookup": {
                    "localField": "dictionary_id",
                    "from": "dictionaries",
                    "foreignField": "_id",
                    "as": "dictionary",
                }
            },
            {"$unwind": "$dictionary"},
            {"$match": {"dictionary.name": f"L{layer}A-l1-0.00012-lr-0.001-32x", "index": head_index}},
            {"$unwind": "$attn_scores"},
            {
                "$lookup": {
                    "localField": "attn_scores.dictionary1_id",
                    "from": "dictionaries",
                    "foreignField": "_id",
                    "as": "attn_scores.dictionary1",
                }
            },
            {"$unwind": "$attn_scores.dictionary1"},
            {
                "$lookup": {
                    "localField": "attn_scores.dictionary2_id",
                    "from": "dictionaries",
                    "foreignField": "_id",
                    "as": "attn_scores.dictionary2",
                }
            },
            {"$unwind": "$attn_scores.dictionary2"},
            {
                "$project": {
                    "attn_scores.dictionary1_id": 0,
                    "attn_scores.dictionary2_id": 0,
                }
            },
            {
                "$group": {
                    "_id": "$_id",
                    "attn_scores": {"$push": "$attn_scores"},
                    "dictionary": {"$first": "$dictionary"},
                    "index": {"$first": "$index"},
                }
            },
        ]

    return attn_head_collection.aggregate(pipeline).next()
