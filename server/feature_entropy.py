from typing import List, Dict, Set, Any, Optional
import numpy as np
import chess
from datasets import load_from_disk, load_dataset
from lm_saes.circuit.leela_board import LeelaBoard
from lm_saes.database import MongoClient


class FeatureEntropyCalculator:
    
    def __init__(self, mongo_client: MongoClient, sae_series: str = "BT4-exp128"):
        self.mongo_client = mongo_client
        self.sae_series = sae_series
        self._dataset_cache: Dict[str, Any] = {}
    
    def get_tc_sae_name(self, layer: int) -> str:
        """Get TC SAE name for a given layer."""
        return f"BT4_tc_L{layer}M"
    
    def get_dataset_by_name(self, dataset_name: str):
        """Get dataset by name with caching."""
        if dataset_name in self._dataset_cache:
            return self._dataset_cache[dataset_name]
        dataset_record = self.mongo_client.get_dataset(dataset_name)
        if dataset_record is None:
            raise ValueError(f"Dataset not found: {dataset_name}")
        ds_cfg = dataset_record.cfg
        dataset = load_from_disk(ds_cfg.dataset_name_or_path) if ds_cfg.is_dataset_on_disk else load_dataset(ds_cfg.dataset_name_or_path)
        self._dataset_cache[dataset_name] = dataset
        return dataset
    
    def get_piece_type_category(self, pos: int, fen: str) -> Optional[str]:
        """
        Get piece type category based on current side to move.
        Returns one of: myr, myn, myb, myq, myk, myp, space, opponentr, opponentn, opponentb, opponentq, opponentk, opponentp
        """
        try:
            lboard = LeelaBoard.from_fen(fen, history_synthesis=True)
            square = lboard.idx2sq(pos)
            piece = lboard.pc_board.piece_at(chess.parse_square(square))
            
            if piece is None:
                return "space"
            
            piece_symbol = piece.symbol()
            is_white = piece.color == chess.WHITE
            is_current_turn = lboard.turn == chess.WHITE
            
            # Map piece symbol to type
            piece_type_map = {
                'R': 'r', 'r': 'r',
                'N': 'n', 'n': 'n',
                'B': 'b', 'b': 'b',
                'Q': 'q', 'q': 'q',
                'K': 'k', 'k': 'k',
                'P': 'p', 'p': 'p'
            }
            piece_type = piece_type_map.get(piece_symbol, None)
            if piece_type is None:
                return "space"
            
            # Determine if piece belongs to current side
            if is_white == is_current_turn:
                return f"my{piece_type}"
            else:
                return f"opponent{piece_type}"
        except Exception:
            return None
    
    def fetch_tc_top_activations(self, layer: int, feature_idx: int, top_k: int = 16, analysis_name: str = "default"):
        """Fetch top activations for a TC feature."""
        sae_name = self.get_tc_sae_name(layer)
        feature_record = self.mongo_client.get_feature(sae_name=sae_name, sae_series=self.sae_series, index=feature_idx)
        
        if not feature_record or not feature_record.analyses:
            return []

        analysis = next((a for a in feature_record.analyses if a.name == analysis_name), feature_record.analyses[0])
        if not analysis.samplings:
            return []

        sampling = analysis.samplings[0]
        feature_values = np.asarray(sampling.feature_acts_values)
        dataset_names = sampling.dataset_name
        context_indices = sampling.context_idx
        shard_indices = getattr(sampling, 'shard_idx', None)
        n_shards = getattr(sampling, 'n_shards', None)
        positions = getattr(sampling, 'feature_acts_indices', None)

        top_samples = []
        for rank in range(min(top_k, len(feature_values))):
            activation_value = float(feature_values[rank])
            
            # Parse indices robustly: positions may be np.ndarray or nested list/tuple
            context_idx_idx, pos = rank, None
            if positions is not None:
                if isinstance(positions, np.ndarray) and positions.ndim == 2:
                    context_idx_idx = int(positions[0, rank])
                    pos = int(positions[1, rank])
                elif isinstance(positions, (list, tuple)) and len(positions) >= 2:
                    context_idx_idx = int(positions[0][rank])
                    pos = int(positions[1][rank])
            
            # Get dataset info using context_idx_idx
            dataset_name = str(dataset_names[context_idx_idx])
            context_idx = int(context_indices[context_idx_idx])
            shard_idx = int(shard_indices[context_idx_idx]) if shard_indices is not None else None
            n_shard = int(n_shards[context_idx_idx]) if n_shards is not None else None

            # Load dataset and get sample
            dataset = self.get_dataset_by_name(dataset_name)
            if shard_idx is not None and n_shard is not None:
                try:
                    dataset_shard = dataset.shard(n_shard, shard_idx) if hasattr(dataset, 'shard') else dataset
                    sample = dataset_shard[int(context_idx)]
                except Exception:
                    sample = dataset[int(context_idx)]
            else:
                sample = dataset[int(context_idx)]
            
            prompt = sample.get("prompt") or sample.get("text") or sample.get("fen") or sample
            # try multiple ways to get FEN
            if isinstance(prompt, str):
                prompt_stripped = prompt.strip()
                # check if it is a valid FEN format (at least 4 spaces separated parts)
                if len(prompt_stripped.split()) >= 4:
                    fen = prompt_stripped
                else:
                    fen = sample.get("fen") if isinstance(sample, dict) else None
            elif isinstance(sample, dict):
                fen = sample.get("fen")
            else:
                fen = None
            
            # get piece_type_category
            piece_type_category = None
            if pos is not None and fen is not None:
                try:
                    piece_type_category = self.get_piece_type_category(pos, fen)
                except Exception:
                    pass
            # if cannot get from pos and fen, try to get from sample directly
            if piece_type_category is None and isinstance(sample, dict):
                piece_type_category = sample.get("piece_type_category")

            top_samples.append({
                "rank": rank + 1,
                "activation": activation_value,
                "piece_type_category": piece_type_category,
            })

        return top_samples
    
    def compute_feature_entropy(self, samples: List[Dict]) -> float:
        """
        Compute entropy from activation samples.
        Returns entropy value.
        """
        if not samples:
            return 0.0
        
        category_counts = {}
        valid_samples = 0
        
        for sample in samples:
            category = sample.get('piece_type_category')
            if category is not None:
                category_counts[category] = category_counts.get(category, 0) + 1
                valid_samples += 1
        
        if valid_samples == 0:
            return 0.0
        
        category_probs = {cat: count / valid_samples for cat, count in category_counts.items()}
        entropy = -sum(p * np.log2(p) for p in category_probs.values() if p > 0)
        
        return float(entropy)
    
    def extract_tc_features_from_graph(self, graph_data: Dict) -> List[Dict[str, Any]]:
        """
        Extract TC feature list from graph data
        Returns format: [{"layer": 0, "features": [3026, 3113, ...]}, ...]
        """
        nodes = graph_data.get("nodes", [])
        layer_features: Dict[int, Set[int]] = {}
        
        for node in nodes:
            if node.get("feature_type") == "cross layer transcoder":
                node_id = node.get("node_id", "")
                if node_id:
                    parts = node_id.split("_")
                    if len(parts) >= 2:
                        try:
                            layer = int(parts[0]) // 2  # original layer divided by 2
                            feature_num = int(parts[1])
                            
                            if layer not in layer_features:
                                layer_features[layer] = set()
                            layer_features[layer].add(feature_num)
                        except (ValueError, IndexError):
                            continue
        
        # convert to list format
        result = []
        for layer in sorted(layer_features.keys()):
            result.append({
                "layer": layer,
                "features": sorted(list(layer_features[layer]))
            })
        
        return result
    
    def calculate_layer_average_entropy(
        self, 
        graph_data: Dict,
        top_k: int = 16,
        analysis_name: str = "default"
    ) -> Dict[str, Any]:
        """
        Calculate the average entropy for each layer of TC features
        
        Args:
            graph_data: graph data
            top_k: top k activation samples for each feature
            analysis_name: analysis name
        
        Returns:
            {
                "layer_stats": [
                    {
                        "layer": 0,
                        "n_features": 16,
                        "avg_entropy": 1.2345,
                        "min_entropy": 0.0,
                        "max_entropy": 2.5,
                        "std_entropy": 0.5
                    },
                    ...
                ],
                "overall_avg_entropy": 1.5,
                "total_features": 150
            }
        """
        # extract TC features
        tc_feature_layers = self.extract_tc_features_from_graph(graph_data)
        
        layer_stats = []
        all_entropies = []
        total_features = 0
        
        for layer_info in tc_feature_layers:
            layer = layer_info["layer"]
            features = layer_info["features"]
            
            # calculate the entropy for each feature in this layer
            layer_entropies = []
            for feature_idx in features:
                try:
                    top_samples = self.fetch_tc_top_activations(layer, feature_idx, top_k=top_k, analysis_name=analysis_name)
                    if top_samples:
                        entropy = self.compute_feature_entropy(top_samples)
                        layer_entropies.append(entropy)
                except Exception as e:
                    print(f"Error computing entropy for L{layer} F{feature_idx}: {e}")
                    continue
            
            # calculate the statistics for this layer
            if layer_entropies:
                avg_entropy = float(np.mean(layer_entropies))
                min_entropy = float(np.min(layer_entropies))
                max_entropy = float(np.max(layer_entropies))
                std_entropy = float(np.std(layer_entropies))
                
                layer_stats.append({
                    "layer": layer,
                    "n_features": len(features),
                    "avg_entropy": avg_entropy,
                    "min_entropy": min_entropy,
                    "max_entropy": max_entropy,
                    "std_entropy": std_entropy
                })
                
                all_entropies.extend(layer_entropies)
                total_features += len(features)
        
        # calculate the overall statistics
        overall_avg_entropy = float(np.mean(all_entropies)) if all_entropies else 0.0
        
        return {
            "layer_stats": layer_stats,
            "overall_avg_entropy": overall_avg_entropy,
            "total_features": total_features
        }