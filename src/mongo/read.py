
from src.lm_saes.config import MongoDBConfig
from src.lm_saes.database import MongoClient, FeatureRecord, SAERecord, DatasetRecord, ModelRecord
from src.lm_saes.resource_loaders import load_dataset_shard
from typing import Optional, Dict, List, Tuple, Any
from datasets import load_from_disk
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

DATASET_PATH = "/inspire/hdd/global_user/hezhengfu-240208120186/data/rlin_data/Chess/chess_master_data"

def get_fen_from_context_idx(context_idx: int, dataset_path: str = DATASET_PATH,
                           shard_idx: int = 0, n_shards: int = 1, mongo_client=None, dataset_name: str = "master") -> Optional[str]:
    try:
        if mongo_client:
            cfg = mongo_client.get_dataset_cfg(dataset_name)
            if cfg:
                dataset = load_dataset_shard(cfg, shard_idx, n_shards)
                if context_idx < len(dataset):
                    fen_data = dataset[context_idx]
                    if hasattr(fen_data, 'get') and callable(getattr(fen_data, 'get', None)):
                        fen = fen_data.get('fen')
                    elif isinstance(fen_data, dict):
                        fen = fen_data.get('fen')
                    else:
                        fen = getattr(fen_data, 'fen', None)
                    return fen
        else:
            if not os.path.exists(dataset_path):
                return None
            dataset = load_from_disk(dataset_path)
            if context_idx >= len(dataset):
                return None
            fen_data = dataset[context_idx]
            if hasattr(fen_data, 'get') and callable(getattr(fen_data, 'get', None)):
                fen = fen_data.get('fen')
            elif isinstance(fen_data, dict):
                fen = fen_data.get('fen')
            else:
                fen = getattr(fen_data, 'fen', None)
            if isinstance(fen, str):
                return fen
    except Exception:
        return None

def get_feature_top_activation(mongo_client, layer: int, feature_id: int, feature_type: str, sae_series: str = "BT4-exp128", analysis_name: str = "default") -> Optional[Dict]:
    """
    get top activation information for a specified feature.

    Args:
        mongo_client: MongoDB client
        layer: layer index
        feature_id: feature index
        feature_type: feature type ("transcoder" or "lorsa")
        sae_series: SAE series, default is "BT4-exp128"
        analysis_name: analysis name, default is "default"

    Returns:
        dictionary containing top activation information, if not found return None.
        The dictionary contains the following fields:
        - activation_value: activation value
        - context_idx: context index
        - position: position index (if available)
        - dataset_name: dataset name
        - shard_idx: shard index (if available)
        - n_shards: shard total (if available)
        - fen: FEN string (if available)
    """
    # build SAE name
    sae_name = (
        f"BT4_lorsa_L{layer}A_k30_e16" if feature_type == "lorsa"
        else f"BT4_tc_L{layer}M_k30_e16" if feature_type == "transcoder"
        else None
    )
    if sae_name is None:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    # get feature record
    fr = mongo_client.get_feature(sae_name=sae_name, sae_series=sae_series, index=feature_id)
    if not fr or not fr.analyses:
        return None

    # get specified analysis
    analysis = next((a for a in fr.analyses if a.name == analysis_name), fr.analyses[0])
    if not analysis.samplings:
        return None

    # use first sampling (usually contains top activations)
    sampling = analysis.samplings[0]
    feature_values = np.asarray(sampling.feature_acts_values)
    
    if len(feature_values) == 0:
        return None

    # find maximum activation value (top activation)
    max_idx = np.argmax(np.abs(feature_values))
    activation_value = float(feature_values[max_idx])

    # get corresponding context_idx
    context_indices = np.asarray(sampling.context_idx)
    dataset_names = sampling.dataset_name
    shard_idx = getattr(sampling, 'shard_idx', None)
    n_shards = getattr(sampling, 'n_shards', None)
    positions = getattr(sampling, 'feature_acts_indices', None)

    # parse position information
    context_idx_idx = max_idx
    position = None
    if positions is not None:
        if isinstance(positions, np.ndarray) and positions.ndim == 2:
            context_idx_idx = int(positions[0, max_idx])
            position = int(positions[1, max_idx])
        elif isinstance(positions, (list, tuple)) and len(positions) >= 2:
            context_idx_idx = int(positions[0][max_idx])
            position = int(positions[1][max_idx])

    # get dataset information
    context_idx = int(context_indices[context_idx_idx])
    dataset_name = str(dataset_names[context_idx_idx]) if context_idx_idx < len(dataset_names) else "master"
    shard_idx_val = (
        int(shard_idx[context_idx_idx])
        if shard_idx is not None
        and isinstance(shard_idx, (np.ndarray, list, tuple))
        and context_idx_idx < len(shard_idx)
        else None
    )
    n_shards_val = (
        int(n_shards[context_idx_idx])
        if n_shards is not None
        and isinstance(n_shards, (np.ndarray, list, tuple))
        and context_idx_idx < len(n_shards)
        else None
    )

    # try to get FEN
    fen = None
    try:
        if shard_idx_val is not None and n_shards_val is not None:
            cfg = mongo_client.get_dataset_cfg(dataset_name)
            if cfg:
                dataset = load_dataset_shard(cfg, shard_idx_val, n_shards_val)
                if context_idx < len(dataset):
                    fen_data = dataset[context_idx]
                    if hasattr(fen_data, 'get') and callable(getattr(fen_data, 'get', None)):
                        fen = fen_data.get('fen')
                    elif isinstance(fen_data, dict):
                        fen = fen_data.get('fen')
                    else:
                        fen = getattr(fen_data, 'fen', None)
        else:
            fen = get_fen_from_context_idx(context_idx, DATASET_PATH, mongo_client=mongo_client)
    except Exception:
        pass

    return {
        "activation_value": activation_value,
        "context_idx": context_idx,
        "position": position,
        "dataset_name": dataset_name,
        "shard_idx": shard_idx_val,
        "n_shards": n_shards_val,
        "fen": fen,
    }

def get_feature_top_activation_value(mongo_client, layer: int, feature_id: int, feature_type: str, sae_series: str = "BT4-exp128", analysis_name: str = "default") -> Optional[float]:
    """
    get top activation value for a specified feature.

    Args:
        mongo_client: MongoDB client
        layer: layer index
        feature_id: feature index
        feature_type: feature type ("transcoder" or "lorsa")
        sae_series: SAE series, default is "BT4-exp128"
        analysis_name: analysis name, default is "default"

    Returns:
        maximum activation value (float), if not found return None.
    """
    # build SAE name
    sae_name = (
        f"BT4_lorsa_L{layer}A_k30_e16" if feature_type == "lorsa"
        else f"BT4_tc_L{layer}M_k30_e16" if feature_type == "transcoder"
        else None
    )
    if sae_name is None:
        raise ValueError(f"Unknown feature_type: {feature_type}")
    
    # get feature record
    fr = mongo_client.get_feature(sae_name=sae_name, sae_series=sae_series, index=feature_id)
    if not fr or not fr.analyses:
        return None
    
    # get specified analysis
    analysis = next((a for a in fr.analyses if a.name == analysis_name), fr.analyses[0])
    if not analysis.samplings:
        return None
    
    # use first sampling
    sampling = analysis.samplings[0]
    feature_values = np.asarray(sampling.feature_acts_values)
    
    if len(feature_values) == 0:
        return None
    
    # find maximum activation value (top activation)
    max_idx = np.argmax(np.abs(feature_values))
    activation_value = float(feature_values[max_idx])
    
    return activation_value


def get_feature_top_activations(mongo_client, layer_or_name, feature_id_or_index, feature_type=None, sae_series: str = "BT4-exp128", analysis_name: str = "default") -> List[Dict]:
    pass


def get_feature_top_fen(mongo_client, layer: int, feature_id: int, feature_type: str, sae_series: str = "BT4-exp128", analysis_name: str = "default") -> List[str]:
    sae_name = f"BT4_lorsa_L{layer}A_k30_e16" if feature_type == "lorsa" else f"BT4_tc_L{layer}M_k30_e16" if feature_type == "transcoder" else None
    if sae_name is None:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    fr = mongo_client.get_feature(sae_name=sae_name, sae_series=sae_series, index=feature_id)
    if not fr or not fr.analyses:
        return []
    analysis = next((a for a in fr.analyses if a.name == analysis_name), fr.analyses[0])
    if not analysis.samplings:
        return []

    unique_fens = set()
    context_idx_to_fen = {}

    for sampling in analysis.samplings:
        context_indices = sampling.context_idx
        shard_idx = getattr(sampling, 'shard_idx', None)
        n_shards = getattr(sampling, 'n_shards', None)
        dataset_names = getattr(sampling, 'dataset_name', None)

        use_sharding = (shard_idx is not None and n_shards is not None and
                       hasattr(shard_idx, '__len__') and len(shard_idx) > 0)

        if use_sharding:
            shard_groups = {}
            for i, context_idx in enumerate(context_indices):
                context_idx = int(context_idx)
                current_shard_idx = shard_idx[i] if i < len(shard_idx) else shard_idx[0]
                current_n_shards = n_shards[i] if i < len(n_shards) else n_shards[0]
                current_dataset_name = dataset_names[i] if dataset_names and i < len(dataset_names) else "master"

                key = (current_shard_idx, current_n_shards, current_dataset_name)
                if key not in shard_groups:
                    shard_groups[key] = []
                shard_groups[key].append(context_idx)

            for (shard_idx_val, n_shards_val, dataset_name_val), context_indices_group in shard_groups.items():
                try:
                    cfg = mongo_client.get_dataset_cfg(dataset_name_val)
                    if cfg:
                        dataset = load_dataset_shard(cfg, shard_idx_val, n_shards_val)
                        for context_idx in context_indices_group:
                            if context_idx not in context_idx_to_fen:
                                if context_idx < len(dataset):
                                    fen_data = dataset[context_idx]
                                    if hasattr(fen_data, 'get') and callable(getattr(fen_data, 'get', None)):
                                        fen = fen_data.get('fen')
                                    elif isinstance(fen_data, dict):
                                        fen = fen_data.get('fen')
                                    else:
                                        fen = getattr(fen_data, 'fen', None)
                                    context_idx_to_fen[context_idx] = fen
                                else:
                                    context_idx_to_fen[context_idx] = None
                except Exception:
                    for context_idx in context_indices_group:
                        if context_idx not in context_idx_to_fen:
                            context_idx_to_fen[context_idx] = None
        else:
            for context_idx in context_indices:
                context_idx = int(context_idx)
                if context_idx not in context_idx_to_fen:
                    fen = get_fen_from_context_idx(context_idx, DATASET_PATH, mongo_client=mongo_client)
                    context_idx_to_fen[context_idx] = fen

        for context_idx in context_indices:
            context_idx = int(context_idx)
            fen = context_idx_to_fen.get(context_idx)
            if fen is not None:
                unique_fens.add(fen)

    return list(unique_fens)


def get_feature_top_fen_batch(mongo_client, features_list, sae_series: str = "BT4-exp128", analysis_name: str = "default") -> Dict[Tuple[str, int, int], List[str]]:
    """
    batch get top FEN list for multiple features.

    Args:
        mongo_client: MongoDB client
        features_list: [(feature_type, layer, feature_id), ...]
        sae_series: SAE series
        analysis_name: analysis name

    Returns:
        Dict[(feature_type, layer, feature_id), List[str]]: feature to FEN list mapping
    """
    result = {}

    # group by layer and feature_type for batch query
    features_by_sae = {}
    for feature_type, layer, feature_id in features_list:
        sae_name = f"BT4_lorsa_L{layer}A_k30_e16" if feature_type == "lorsa" else f"BT4_tc_L{layer}M_k30_e16" if feature_type == "transcoder" else None
        if sae_name is None:
            print(f"Unknown feature_type: {feature_type}, skip")
            continue
        if sae_name not in features_by_sae:
            features_by_sae[sae_name] = []
        features_by_sae[sae_name].append((feature_type, layer, feature_id, feature_id))

    print(f"Group by SAE completed:")
    for sae_name, feature_list in features_by_sae.items():
        print(f"   {sae_name}: {len(feature_list)} features")

    # batch query features
    print(f"Start batch processing {len(features_by_sae)} SAEs...")
    for sae_name, feature_info_list in tqdm(features_by_sae.items(), desc="Processing SAE"):
        print(f"Processing SAE: {sae_name}, features number: {len(feature_info_list)}")
        try:
            # parallel query all features of the SAE
            sae_features = {}
            print(f"   parallel query {len(feature_info_list)} features...")

            def query_single_feature(args):
                feature_type, layer, feature_id, index = args
                try:
                    fr = mongo_client.get_feature(sae_name=sae_name, sae_series=sae_series, index=index)
                    if fr and fr.analyses:
                        analysis = next((a for a in fr.analyses if a.name == analysis_name), fr.analyses[0])
                        if analysis.samplings:
                            return (feature_type, layer, feature_id), analysis
                except Exception:
                    pass
                return None

            # use thread pool to parallel query
            max_workers = min(20, len(feature_info_list))
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(query_single_feature, info) for info in feature_info_list]
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"Query {sae_name}"):
                        query_result = future.result()
                        if query_result:
                            key, analysis = query_result
                            sae_features[key] = analysis
            except Exception as e:
                print(f"Parallel query failed, fall back to sequential query: {e}")
                # fall back to sequential query
                for info in tqdm(feature_info_list, desc=f"Sequential query {sae_name}"):
                    query_result = query_single_feature(info)
                    if query_result:
                        key, analysis = query_result
                        sae_features[key] = analysis

            print(f"  Successfully get {len(sae_features)} features for SAE {sae_name}")

            # process all features of the SAE
            if sae_features:
                _process_sae_features_batch(mongo_client, sae_features, result)

        except Exception as e:
            print(f"Failed to process SAE {sae_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"get_feature_top_fen_batch completed")
    print(f"    return result contains {len(result)} features")
    total_fens = sum(len(fen_list) for fen_list in result.values())
    print(f"    total collected {total_fens} unique FENs")
    if result:
        avg_fens = total_fens / len(result)
        print(f"    average {avg_fens:.1f} FENs per feature")

    return result


def _process_sae_features_batch(mongo_client, sae_features, result):
    """process all features of a SAE for batch FEN extraction"""
    print(f"Start _process_sae_features_batch")
    print(f"   number of sae_features: {len(sae_features)}")
    print(f"   current size of result: {len(result)}")

    if not sae_features:
        print("sae_features is empty, return")
        return

    # collect all needed context_idx
    all_context_indices = set()
    feature_to_contexts = {}

    print("Collect context indices...")
    for (feature_type, layer, feature_id), analysis in sae_features.items():
        context_indices = set()
        for sampling in analysis.samplings:
            for context_idx in sampling.context_idx:
                context_indices.add(int(context_idx))

        feature_to_contexts[(feature_type, layer, feature_id)] = context_indices
        all_context_indices.update(context_indices)

    print(f"    collected {len(feature_to_contexts)} features, {len(all_context_indices)} unique context indices")
    print(f"   context indices range: {min(all_context_indices)} - {max(all_context_indices)}")

    if not all_context_indices:
        print("No context indices, return")
        return

    # group context_idx by shard
    shard_groups = {}
    context_to_shard_info = {}

    # for each feature, traverse all samplings, collect shard information
    for (feature_type, layer, feature_id), analysis in sae_features.items():
        for sampling in analysis.samplings:
            context_indices = sampling.context_idx
            shard_idx = getattr(sampling, 'shard_idx', None)
            n_shards = getattr(sampling, 'n_shards', None)
            dataset_names = getattr(sampling, 'dataset_name', None)

            use_sharding = (shard_idx is not None and n_shards is not None and
                           hasattr(shard_idx, '__len__') and len(shard_idx) > 0)

            if use_sharding:
                for i, context_idx in enumerate(context_indices):
                    context_idx = int(context_idx)
                    if context_idx in all_context_indices:
                        current_shard_idx = shard_idx[i] if i < len(shard_idx) else shard_idx[0]
                        current_n_shards = n_shards[i] if i < len(n_shards) else n_shards[0]
                        current_dataset_name = dataset_names[i] if dataset_names and i < len(dataset_names) else "master"

                        key = (current_shard_idx, current_n_shards, current_dataset_name)
                        if key not in shard_groups:
                            shard_groups[key] = []
                        if context_idx not in shard_groups[key]:
                            shard_groups[key].append(context_idx)
                        context_to_shard_info[context_idx] = key

    print(f"Group by shard completed: {len(shard_groups)} shard groups")
    for (shard_idx, n_shards, dataset_name), indices in list(shard_groups.items())[:3]:
        print(f"    shard ({shard_idx}/{n_shards}, {dataset_name}): {len(indices)} context indices")

    # batch load FEN (multi-thread version)
    context_idx_to_fen = {}

    def load_single_shard(shard_info):
        """load FEN data of a single shard"""
        (shard_idx_val, n_shards_val, dataset_name_val), context_indices_group = shard_info
        shard_results = {}

        try:
            print(f"load shard {shard_idx_val}/{n_shards_val} ({dataset_name_val}), contains {len(context_indices_group)} context indices...")
            cfg = mongo_client.get_dataset_cfg(dataset_name_val)
            if cfg:
                dataset = load_dataset_shard(cfg, shard_idx_val, n_shards_val)
                print(f"dataset size: {len(dataset)}")
                for context_idx in context_indices_group:
                    if context_idx < len(dataset):
                        fen_data = dataset[context_idx]
                        if hasattr(fen_data, 'get') and callable(getattr(fen_data, 'get', None)):
                            fen = fen_data.get('fen')
                        elif isinstance(fen_data, dict):
                            fen = fen_data.get('fen')
                        else:
                            fen = getattr(fen_data, 'fen', None)
                        shard_results[context_idx] = fen
                    else:
                        shard_results[context_idx] = None
            else:
                print(f"cannot get dataset configuration: {dataset_name_val}")
                for context_idx in context_indices_group:
                    shard_results[context_idx] = None
        except Exception as e:
            print(f"load shard failed: {e}")
            for context_idx in context_indices_group:
                shard_results[context_idx] = None
        return shard_results

    print(f"multi-thread load {len(shard_groups)} shards...")
    max_workers = min(8, len(shard_groups))
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(load_single_shard, shard_info) for shard_info in shard_groups.items()]
            for future in tqdm(as_completed(futures), total=len(futures), desc="load shards"):
                shard_results = future.result()
                context_idx_to_fen.update(shard_results)
    except Exception as e:
        print(f"multi-thread load failed, fall back to sequential load: {e}")
        # 回退到顺序加载
        for shard_info in tqdm(shard_groups.items(), desc="sequential load shards"):
            shard_results = load_single_shard(shard_info)
            context_idx_to_fen.update(shard_results)

    print(f"shard load completed: loaded {len(context_idx_to_fen)} context indices")

    # process non-shard cases (multi-thread version)
    non_shard_contexts = [ctx for ctx in all_context_indices if ctx not in context_idx_to_fen]
    if non_shard_contexts:
        print(f"found {len(non_shard_contexts)} non-shard context indices")
        print(f"total {len(all_context_indices)} context indices")
        print(f"shard loaded {len(context_idx_to_fen)} context indices")

        # examine why these context_idx are not covered by shards
        print(f"   examine shard coverage...")
        covered_by_shards = set()
        for context_indices_group in shard_groups.values():
            covered_by_shards.update(context_indices_group)
        print(f"   shard groups cover {len(covered_by_shards)} context indices")
        not_covered = all_context_indices - covered_by_shards
        if not_covered:
            print(f"   context indices not covered by shards: {sorted(list(not_covered))[:10]}")
            print(f"   these may come from samplings without shard information")

        def load_single_non_shard_context(context_idx):
            """load FEN of a single non-shard context"""
            try:
                fen = get_fen_from_context_idx(context_idx, DATASET_PATH, mongo_client=mongo_client)
                return context_idx, fen
            except Exception as e:
                print(f"     get context_idx {context_idx} failed: {e}")
                return context_idx, None

        max_workers = min(16, len(non_shard_contexts))
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(load_single_non_shard_context, ctx) for ctx in non_shard_contexts]
                for future in tqdm(as_completed(futures), total=len(futures), desc="load non-shard data"):
                    context_idx, fen = future.result()
                    context_idx_to_fen[context_idx] = fen
        except Exception as e:
            print(f"multi-thread load non-shard data failed, fall back to sequential load: {e}")
            # fall back to sequential load
            for context_idx in tqdm(non_shard_contexts, desc="sequential load non-shard data"):
                try:
                    fen = get_fen_from_context_idx(context_idx, DATASET_PATH, mongo_client=mongo_client)
                    context_idx_to_fen[context_idx] = fen
                except Exception as e:
                    print(f"     get context_idx {context_idx} failed: {e}")
                    context_idx_to_fen[context_idx] = None

    print(f"all context indices loaded: total {len(context_idx_to_fen)}")
    fen_count = sum(1 for fen in context_idx_to_fen.values() if fen is not None)
    print(f"    successfully loaded FEN: {fen_count}/{len(context_idx_to_fen)} ({fen_count/len(context_idx_to_fen)*100:.1f}%)")

    # generate FEN list for each feature (multi-thread version)
    print(f"generate FEN list for {len(feature_to_contexts)} features...")

    def generate_fen_list(feature_info):
        """generate FEN list for a single feature"""
        key, context_indices = feature_info
        unique_fens = set()
        for context_idx in context_indices:
            fen = context_idx_to_fen.get(context_idx)
            if fen is not None:
                unique_fens.add(fen)
        return key, list(unique_fens)

    max_workers = min(32, len(feature_to_contexts))
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(generate_fen_list, feature_info) for feature_info in feature_to_contexts.items()]
            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="生成FEN列表")):
                key, fen_list = future.result()
                result[key] = fen_list

                if (i + 1) % 100 == 0:
                    print(f"processed {i + 1}/{len(feature_to_contexts)} features")

    except Exception as e:
        print(f"multi-thread generate FEN list failed, fall back to sequential generate: {e}")
        # fall back to sequential generate
        for i, (key, context_indices) in enumerate(tqdm(feature_to_contexts.items(), desc="sequential generate FEN list")):
            unique_fens = set()
            for context_idx in context_indices:
                fen = context_idx_to_fen.get(context_idx)
                if fen is not None:
                    unique_fens.add(fen)
            result[key] = list(unique_fens)

            if (i + 1) % 50 == 0:
                print(f"processed {i + 1}/{len(feature_to_contexts)} features")

    print(f"generate FEN list completed")
    print(f"processed {len(result)} features")
    total_fens = sum(len(fen_list) for fen_list in result.values())
    print(f"total generated {total_fens} unique FENs")
    avg_fens_per_feature = total_fens / len(result) if result else 0
    print(f"average {avg_fens_per_feature:.1f} FENs per feature")

