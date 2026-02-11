
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
    获取指定 feature 的 top activation（最大激活值）信息。

    Args:
        mongo_client: MongoDB客户端
        layer: 层索引
        feature_id: feature索引
        feature_type: feature类型 ("transcoder" 或 "lorsa")
        sae_series: SAE系列，默认为 "BT4-exp128"
        analysis_name: 分析名称，默认为 "default"

    Returns:
        包含 top activation 信息的字典，如果未找到则返回 None。
        字典包含以下字段：
        - activation_value: 激活值
        - context_idx: context索引
        - position: 位置索引（如果有）
        - dataset_name: 数据集名称
        - shard_idx: 分片索引（如果有）
        - n_shards: 分片总数（如果有）
        - fen: FEN字符串（如果可获取）
    """
    # 构建 SAE 名称
    sae_name = (
        f"BT4_lorsa_L{layer}A_k30_e16" if feature_type == "lorsa"
        else f"BT4_tc_L{layer}M_k30_e16" if feature_type == "transcoder"
        else None
    )
    if sae_name is None:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    # 获取 feature record
    fr = mongo_client.get_feature(sae_name=sae_name, sae_series=sae_series, index=feature_id)
    if not fr or not fr.analyses:
        return None

    # 获取指定的 analysis
    analysis = next((a for a in fr.analyses if a.name == analysis_name), fr.analyses[0])
    if not analysis.samplings:
        return None

    # 使用第一个 sampling（通常包含 top activations）
    sampling = analysis.samplings[0]
    feature_values = np.asarray(sampling.feature_acts_values)
    
    if len(feature_values) == 0:
        return None

    # 找到最大激活值（top activation）
    max_idx = np.argmax(np.abs(feature_values))
    activation_value = float(feature_values[max_idx])

    # 获取对应的 context_idx
    context_indices = np.asarray(sampling.context_idx)
    dataset_names = sampling.dataset_name
    shard_idx = getattr(sampling, 'shard_idx', None)
    n_shards = getattr(sampling, 'n_shards', None)
    positions = getattr(sampling, 'feature_acts_indices', None)

    # 解析位置信息
    context_idx_idx = max_idx
    position = None
    if positions is not None:
        if isinstance(positions, np.ndarray) and positions.ndim == 2:
            context_idx_idx = int(positions[0, max_idx])
            position = int(positions[1, max_idx])
        elif isinstance(positions, (list, tuple)) and len(positions) >= 2:
            context_idx_idx = int(positions[0][max_idx])
            position = int(positions[1][max_idx])

    # 获取数据集信息
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

    # 尝试获取 FEN
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
    获取指定 feature 的 top activation 值（最大激活值）。

    Args:
        mongo_client: MongoDB客户端
        layer: 层索引
        feature_id: feature索引
        feature_type: feature类型 ("transcoder" 或 "lorsa")
        sae_series: SAE系列，默认为 "BT4-exp128"
        analysis_name: 分析名称，默认为 "default"

    Returns:
        最大的激活值（float），如果未找到则返回 None。
    """
    # 构建 SAE 名称
    sae_name = (
        f"BT4_lorsa_L{layer}A_k30_e16" if feature_type == "lorsa"
        else f"BT4_tc_L{layer}M_k30_e16" if feature_type == "transcoder"
        else None
    )
    if sae_name is None:
        raise ValueError(f"Unknown feature_type: {feature_type}")
    
    # 获取 feature record
    fr = mongo_client.get_feature(sae_name=sae_name, sae_series=sae_series, index=feature_id)
    if not fr or not fr.analyses:
        return None
    
    # 获取指定的 analysis
    analysis = next((a for a in fr.analyses if a.name == analysis_name), fr.analyses[0])
    if not analysis.samplings:
        return None
    
    # 使用第一个 sampling（通常包含 top activations）
    sampling = analysis.samplings[0]
    feature_values = np.asarray(sampling.feature_acts_values)
    
    if len(feature_values) == 0:
        return None
    
    # 找到最大激活值（top activation）
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
    批量获取多个features的top FEN列表

    Args:
        mongo_client: MongoDB客户端
        features_list: [(feature_type, layer, feature_id), ...]
        sae_series: SAE系列
        analysis_name: 分析名称

    Returns:
        Dict[(feature_type, layer, feature_id), List[str]]: feature到FEN列表的映射
    """
    print(f"🔍 get_feature_top_fen_batch 开始执行")
    print(f"   输入参数: sae_series='{sae_series}', analysis_name='{analysis_name}'")
    print(f"   features_list 长度: {len(features_list)}")
    if features_list:
        print(f"   features_list 示例: {features_list[:3]}")

    if not features_list:
        print("⚠️ features_list 为空，直接返回空字典")
        return {}

    result = {}

    # 按layer和feature_type分组，便于批量查询
    features_by_sae = {}
    for feature_type, layer, feature_id in features_list:
        sae_name = f"BT4_lorsa_L{layer}A_k30_e16" if feature_type == "lorsa" else f"BT4_tc_L{layer}M_k30_e16" if feature_type == "transcoder" else None
        if sae_name is None:
            print(f"⚠️ 未知的feature_type: {feature_type}，跳过")
            continue
        if sae_name not in features_by_sae:
            features_by_sae[sae_name] = []
        features_by_sae[sae_name].append((feature_type, layer, feature_id, feature_id))

    print(f"📊 按SAE分组完成:")
    for sae_name, feature_list in features_by_sae.items():
        print(f"   {sae_name}: {len(feature_list)} 个features")

    # 批量查询features
    print(f"🚀 开始批量处理 {len(features_by_sae)} 个SAE...")
    for sae_name, feature_info_list in tqdm(features_by_sae.items(), desc="处理SAE"):
        print(f"处理SAE: {sae_name}, features数量: {len(feature_info_list)}")
        try:
            # 并行查询SAE的所有features
            sae_features = {}
            print(f"  并行查询 {len(feature_info_list)} 个features...")

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

            # 使用线程池并行查询
            max_workers = min(20, len(feature_info_list))  # 最多20个并发线程
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(query_single_feature, info) for info in feature_info_list]
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"查询{sae_name}"):
                        query_result = future.result()
                        if query_result:
                            key, analysis = query_result
                            sae_features[key] = analysis
            except Exception as e:
                print(f"并行查询失败，回退到顺序查询: {e}")
                # 回退到顺序查询
                for info in tqdm(feature_info_list, desc=f"顺序查询{sae_name}"):
                    query_result = query_single_feature(info)
                    if query_result:
                        key, analysis = query_result
                        sae_features[key] = analysis

            print(f"  SAE {sae_name} 成功获取 {len(sae_features)} 个features")

            # 处理这个SAE的所有features
            if sae_features:
                _process_sae_features_batch(mongo_client, sae_features, result)

        except Exception as e:
            print(f"处理SAE {sae_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"🎉 get_feature_top_fen_batch 执行完成")
    print(f"   返回结果包含 {len(result)} 个features")
    total_fens = sum(len(fen_list) for fen_list in result.values())
    print(f"   总共收集了 {total_fens} 个唯一FEN")
    if result:
        avg_fens = total_fens / len(result)
        print(f"   平均每个feature {avg_fens:.1f} 个FEN")

    return result


def _process_sae_features_batch(mongo_client, sae_features, result):
    """处理单个SAE的所有features的批量FEN提取"""
    print(f"🔧 _process_sae_features_batch 开始执行")
    print(f"   sae_features 数量: {len(sae_features)}")
    print(f"   result 当前大小: {len(result)}")

    if not sae_features:
        print("⚠️ sae_features 为空，直接返回")
        return

    # 收集所有需要的context_idx
    all_context_indices = set()
    feature_to_contexts = {}

    print("📋 收集context indices...")
    for (feature_type, layer, feature_id), analysis in sae_features.items():
        context_indices = set()
        for sampling in analysis.samplings:
            for context_idx in sampling.context_idx:
                context_indices.add(int(context_idx))

        feature_to_contexts[(feature_type, layer, feature_id)] = context_indices
        all_context_indices.update(context_indices)

    print(f"   收集完成: {len(feature_to_contexts)} 个features, {len(all_context_indices)} 个唯一context indices")
    print(f"   context indices 范围: {min(all_context_indices)} - {max(all_context_indices)}")

    if not all_context_indices:
        print("⚠️ 没有context indices，直接返回")
        return

    # 按分片分组context_idx
    shard_groups = {}
    context_to_shard_info = {}

    # 为每个feature遍历其所有samplings，收集分片信息
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

    print(f"📦 分片分组完成: {len(shard_groups)} 个分片groups")
    for (shard_idx, n_shards, dataset_name), indices in list(shard_groups.items())[:3]:  # 只显示前3个
        print(f"   分片 ({shard_idx}/{n_shards}, {dataset_name}): {len(indices)} 个context indices")

    # 批量加载FEN (多线程版本)
    context_idx_to_fen = {}

    def load_single_shard(shard_info):
        """加载单个分片的FEN数据"""
        (shard_idx_val, n_shards_val, dataset_name_val), context_indices_group = shard_info
        shard_results = {}

        try:
            print(f"  加载分片 {shard_idx_val}/{n_shards_val} ({dataset_name_val}), 包含 {len(context_indices_group)} 个context indices...")
            cfg = mongo_client.get_dataset_cfg(dataset_name_val)
            if cfg:
                dataset = load_dataset_shard(cfg, shard_idx_val, n_shards_val)
                print(f"    数据集大小: {len(dataset)}")
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
                print(f"    无法获取数据集配置: {dataset_name_val}")
                for context_idx in context_indices_group:
                    shard_results[context_idx] = None
        except Exception as e:
            print(f"    加载分片失败: {e}")
            for context_idx in context_indices_group:
                shard_results[context_idx] = None
        return shard_results

    print(f"多线程加载 {len(shard_groups)} 个分片的数据...")
    max_workers = min(8, len(shard_groups))  # 最多8个并发线程加载分片
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(load_single_shard, shard_info) for shard_info in shard_groups.items()]
            for future in tqdm(as_completed(futures), total=len(futures), desc="加载分片"):
                shard_results = future.result()
                context_idx_to_fen.update(shard_results)
    except Exception as e:
        print(f"❌ 多线程加载失败，回退到顺序加载: {e}")
        # 回退到顺序加载
        for shard_info in tqdm(shard_groups.items(), desc="顺序加载分片"):
            shard_results = load_single_shard(shard_info)
            context_idx_to_fen.update(shard_results)

    print(f"✅ 分片加载完成: 加载了 {len(context_idx_to_fen)} 个context indices")

    # 处理非分片的情况 (多线程版本)
    non_shard_contexts = [ctx for ctx in all_context_indices if ctx not in context_idx_to_fen]
    if non_shard_contexts:
        print(f"发现 {len(non_shard_contexts)} 个非分片context indices")
        print(f"  总共有 {len(all_context_indices)} 个context indices")
        print(f"  分片加载了 {len(context_idx_to_fen)} 个context indices")
        print(f"  非分片indices示例: {sorted(non_shard_contexts)[:5]}")

        # 检查为什么这些context_idx没有被分片加载
        print(f"  检查分片覆盖情况...")
        covered_by_shards = set()
        for context_indices_group in shard_groups.values():
            covered_by_shards.update(context_indices_group)
        print(f"  分片groups覆盖了 {len(covered_by_shards)} 个context indices")
        not_covered = all_context_indices - covered_by_shards
        if not_covered:
            print(f"  未被分片覆盖的context indices: {sorted(list(not_covered))[:10]}")
            print(f"  这些可能来自没有分片信息的samplings")

        def load_single_non_shard_context(context_idx):
            """加载单个非分片context的FEN"""
            try:
                fen = get_fen_from_context_idx(context_idx, DATASET_PATH, mongo_client=mongo_client)
                return context_idx, fen
            except Exception as e:
                print(f"    获取context_idx {context_idx} 失败: {e}")
                return context_idx, None

        max_workers = min(16, len(non_shard_contexts))  # 最多16个并发线程
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(load_single_non_shard_context, ctx) for ctx in non_shard_contexts]
                for future in tqdm(as_completed(futures), total=len(futures), desc="加载非分片数据"):
                    context_idx, fen = future.result()
                    context_idx_to_fen[context_idx] = fen
        except Exception as e:
            print(f"多线程加载非分片数据失败，回退到顺序加载: {e}")
            # 回退到顺序加载
            for context_idx in tqdm(non_shard_contexts, desc="顺序加载非分片数据"):
                try:
                    fen = get_fen_from_context_idx(context_idx, DATASET_PATH, mongo_client=mongo_client)
                    context_idx_to_fen[context_idx] = fen
                except Exception as e:
                    print(f"    获取context_idx {context_idx} 失败: {e}")
                    context_idx_to_fen[context_idx] = None

    print(f"📈 所有context indices加载完成: 总共 {len(context_idx_to_fen)} 个")
    fen_count = sum(1 for fen in context_idx_to_fen.values() if fen is not None)
    print(f"   成功加载FEN: {fen_count}/{len(context_idx_to_fen)} ({fen_count/len(context_idx_to_fen)*100:.1f}%)")

    # 为每个feature生成FEN列表 (多线程版本)
    print(f"为 {len(feature_to_contexts)} 个features生成FEN列表...")

    def generate_fen_list(feature_info):
        """为单个feature生成FEN列表"""
        key, context_indices = feature_info
        unique_fens = set()
        for context_idx in context_indices:
            fen = context_idx_to_fen.get(context_idx)
            if fen is not None:
                unique_fens.add(fen)
        return key, list(unique_fens)

    max_workers = min(32, len(feature_to_contexts))  # 最多32个并发线程
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(generate_fen_list, feature_info) for feature_info in feature_to_contexts.items()]
            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="生成FEN列表")):
                key, fen_list = future.result()
                result[key] = fen_list

                # 每处理一定数量就报告进度
                if (i + 1) % 100 == 0:
                    print(f"    已处理 {i + 1}/{len(feature_to_contexts)} 个features")

    except Exception as e:
        print(f"多线程生成FEN列表失败，回退到顺序生成: {e}")
        # 回退到顺序生成
        for i, (key, context_indices) in enumerate(tqdm(feature_to_contexts.items(), desc="顺序生成FEN列表")):
            unique_fens = set()
            for context_idx in context_indices:
                fen = context_idx_to_fen.get(context_idx)
                if fen is not None:
                    unique_fens.add(fen)
            result[key] = list(unique_fens)

            if (i + 1) % 50 == 0:
                print(f"    已处理 {i + 1}/{len(feature_to_contexts)} 个features")

    print(f"✅ _process_sae_features_batch 完成")
    print(f"   处理了 {len(result)} 个features")
    total_fens = sum(len(fen_list) for fen_list in result.values())
    print(f"   总共生成了 {total_fens} 个唯一FEN")
    avg_fens_per_feature = total_fens / len(result) if result else 0
    print(f"   平均每个feature {avg_fens_per_feature:.1f} 个FEN")

