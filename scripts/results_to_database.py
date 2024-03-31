from functools import cmp_to_key
import os
import sys

sys.path.insert(0, os.getcwd())

import pymongo
import gridfs
import argparse

import numpy as np

from datasets import Dataset

from tqdm import tqdm

from pymongo import MongoClient

from core.utils.bytes import np_to_bytes

def connect_to_db(url, db_name):
    client = MongoClient(url)

    db = client[db_name]

    return db

def find_one(collection, condition):
    return collection.find_one(condition)

def update_one(collection, condition, set):
    collection.update_one(condition, {"$set": set})

def main():
    parser = argparse.ArgumentParser(description='Insert results into MongoDB')
    parser.add_argument('--uri', type=str, default='mongodb://localhost:27017', help='MongoDB URI')
    parser.add_argument('--database', type=str, default='mechinterp', help='MongoDB database')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory containing results')
    args = parser.parse_args()

    client = pymongo.MongoClient(args.uri)
    db = client[args.database]
    fs = gridfs.GridFS(db)
    feature_collection = db['features']
    dictionary_collection = db['dictionaries']
    dictionary_collection.create_index('name', unique=True)
    feature_collection.create_index([('dictionary_id', pymongo.ASCENDING), ('index', pymongo.ASCENDING)], unique=True)

    for dict_name in os.listdir(args.result_dir):
        dict_path = os.path.join(args.result_dir, dict_name)
        if not os.path.isdir(dict_path):
            continue

        analysis_dir = os.path.join(dict_path, 'analysis')
        if not os.path.isdir(analysis_dir):
            continue
        
        dictionary = dictionary_collection.find_one({'name': dict_name})
        dict_id = dictionary['_id'] if dictionary is not None else dictionary_collection.insert_one({'name': dict_name}).inserted_id

        analysis_names = os.listdir(os.path.join(dict_path, 'analysis'))
        analysis_names = sorted(analysis_names, key=cmp_to_key(lambda a, b: -1 if a == "top_activations" else 1 if b == "top_activations" else 0))
        for analysis_name in analysis_names:
            analysis_path = os.path.join(dict_path, 'analysis', analysis_name)
            if not os.path.isdir(analysis_path):
                continue

            feature_activations = Dataset.load_from_disk(analysis_path)
            in_db_indices = [feature['index'] for feature in feature_collection.find({'dictionary_id': dict_id, 'analysis.name': analysis_name})]
            # print(f'{dict_name}/{analysis_name}: {len(in_db_indices)} features already in database')
            if len(in_db_indices) == len(feature_activations):
                print(f'{dict_name}/{analysis_name} already in database')
                continue

            for feature_activation in tqdm(feature_activations, desc=f'{dict_name}/{analysis_name}'):
                if feature_activation['index'] in in_db_indices:
                    continue
                feature = feature_collection.find_one({'dictionary_id': dict_id, 'index': feature_activation['index']})
                if feature is None:
                    feature = {
                        'dictionary_id': dict_id,
                        'index': feature_activation['index'],
                        'act_times': feature_activation['act_times'],
                        'max_feature_acts': feature_activation['max_feature_acts'],
                        'feature_acts_all': fs.put(np_to_bytes(np.array(feature_activation['feature_acts_all']))),
                        'analysis': [{
                            'name': analysis_name,
                            'feature_acts': fs.put(np_to_bytes(np.array(feature_activation['feature_acts']))),
                            'contexts': fs.put(np_to_bytes(np.array(feature_activation['contexts'])))
                        }]
                    }
                    feature_collection.insert_one(feature)
                else:
                    assert 'analysis' in feature, f'Feature {feature["_id"]} does not have analysis field'
                    if any(a['name'] == analysis_name for a in feature['analysis']):
                        raise ValueError(f'Feature {feature["_id"]} already has analysis {analysis_name}')
                    feature['analysis'].append({
                        'name': analysis_name,
                        'feature_acts': fs.put(np_to_bytes(np.array(feature_activation['feature_acts']))),
                        'contexts': fs.put(np_to_bytes(np.array(feature_activation['contexts'])))
                    })
                    feature_collection.update_one({'_id': feature['_id']}, {'$set': {'analysis': feature['analysis']}})
            
if __name__ == '__main__':
    main()