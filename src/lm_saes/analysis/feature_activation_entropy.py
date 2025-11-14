"""Feature Entropy Analyzer for computing entropy of feature activations over piece type categories."""

from typing import Any, Dict, List, Optional, cast, TYPE_CHECKING
import warnings

import torch
import numpy as np
import chess
from tqdm import tqdm 
from functools import partial
from einops import repeat
from torch.distributed.device_mesh import DeviceMesh

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.lorsa import LowRankSparseAttention
from lm_saes.config import FeatureAnalyzerConfig
from lm_saes.activation.factory import ActivationFactory
from lm_saes.utils.misc import is_primary_rank

# Avoid circular import by importing LeelaBoard only when type checking or at runtime
if TYPE_CHECKING:
    from lm_saes.circuit.leela_board import LeelaBoard


class FeatureEntropyAnalyzer:
    """Analyzes feature activation entropy over piece type categories.
    
    This class computes entropy metrics for SAE features by analyzing how
    feature activations distribute across different chess piece categories
    (e.g., "myr", "opponentq", "space", etc.).
    """

    def __init__(self, cfg: FeatureAnalyzerConfig):
        """Initialize the feature entropy analyzer.
        
        Args:
            cfg: Analysis configuration specifying parameters like sample sizes and thresholds
        """
        self.cfg = cfg

    @staticmethod
    def get_piece_type_category(pos: int, fen: str) -> Optional[str]:
        """Get the piece type category at a given position.
        
        Args:
            pos: Position index (0-63)
            fen: FEN string representing the board state
            
        Returns:
            Category string like "myr", "opponentq", "space", or None if error
        """
        try:
            # Import LeelaBoard here to avoid circular import
            from lm_saes.circuit.leela_board import LeelaBoard
            
            lboard = LeelaBoard.from_fen(fen, history_synthesis=True)
            square = lboard.idx2sq(pos)
            piece = lboard.pc_board.piece_at(chess.parse_square(square))
            if piece is None:
                return "space"

            is_white = piece.color == chess.WHITE
            is_current_turn = lboard.turn == chess.WHITE

            symbol = piece.symbol()
            symbol_map = {
                "R": "r", "r": "r",
                "N": "n", "n": "n",
                "B": "b", "b": "b",
                "Q": "q", "q": "q",
                "K": "k", "k": "k",
                "P": "p", "p": "p",
            }
            ptype = symbol_map.get(symbol)
            if ptype is None:
                return "space"
            return f"my{ptype}" if is_white == is_current_turn else f"opponent{ptype}"
        except Exception:
            return None

    @staticmethod
    def get_piece_type_categories_batch(
        fens: List[Optional[str]], 
        seq_len: int
    ) -> List[List[Optional[str]]]:
        """Get piece type categories for a batch of FENs.
        
        Args:
            fens: List of FEN strings (one per sample in batch)
            seq_len: Sequence length (number of positions per FEN)
            
        Returns:
            List of lists: [batch_size][seq_len] containing category strings
        """
        from lm_saes.circuit.leela_board import LeelaBoard
        
        batch_size = len(fens)
        categories = [[None for _ in range(seq_len)] for _ in range(batch_size)]
        
        for b, fen in enumerate(fens):
            if fen is None:
                continue
            
            try:
                lboard = LeelaBoard.from_fen(fen, history_synthesis=True)
                is_current_turn = lboard.turn == chess.WHITE
                
                for pos in range(seq_len):
                    try:
                        square = lboard.idx2sq(pos)
                        piece = lboard.pc_board.piece_at(chess.parse_square(square))
                        
                        if piece is None:
                            categories[b][pos] = "space"
                            continue
                        
                        is_white = piece.color == chess.WHITE
                        symbol = piece.symbol()
                        symbol_map = {
                            "R": "r", "r": "r",
                            "N": "n", "n": "n",
                            "B": "b", "b": "b",
                            "Q": "q", "q": "q",
                            "K": "k", "k": "k",
                            "P": "p", "p": "p",
                        }
                        ptype = symbol_map.get(symbol)
                        if ptype is None:
                            categories[b][pos] = "space"
                        else:
                            categories[b][pos] = f"my{ptype}" if is_white == is_current_turn else f"opponent{ptype}"
                    except Exception:
                        categories[b][pos] = None
            except Exception:
                continue
        
        return categories

    @staticmethod
    def compute_entropy_from_counts(counts: Dict[str, int]) -> Dict[str, Any]:
        """Compute entropy from category counts.
        
        Args:
            counts: Dictionary mapping category to count
            
        Returns:
            Dictionary containing entropy, counts, probabilities, and total
        """
        if not counts:
            return {
                "entropy": 0.0,
                "category_counts": {},
                "category_probs": {},
                "total_activations": 0
            }
        
        total = sum(counts.values())
        if total == 0:
            return {
                "entropy": 0.0,
                "category_counts": counts,
                "category_probs": {},
                "total_activations": 0
            }
        
        # Calculate probabilities
        probs = {k: v / total for k, v in counts.items()}
        
        # Calculate entropy: H = -sum(p * log2(p))
        entropy = -sum(p * np.log2(p) for p in probs.values() if p > 0)
        
        return {
            "entropy": float(entropy),
            "category_counts": counts,
            "category_probs": probs,
            "total_activations": total
        }

    @staticmethod
    def update_feature_counts_batch(
        feature_acts: torch.Tensor,  # [batch_size, seq_len, d_sae]
        categories: List[List[Optional[str]]],  # [batch_size][seq_len]
        ignore_masks: torch.Tensor,  # [batch_size, seq_len]
        feature_category_counts: List[Dict[str, int]],  # [d_sae]
    ) -> None:
        """Update feature category counts from a batch of activations.
        
        Args:
            feature_acts: Feature activations [batch_size, seq_len, d_sae]
            categories: Piece type categories [batch_size][seq_len]
            ignore_masks: Boolean mask for valid positions [batch_size, seq_len]
            feature_category_counts: List of category count dicts to update (in-place)
        """
        batch_size, seq_len, d_sae = feature_acts.shape
        
        # Convert to numpy for faster iteration
        feature_acts_np = feature_acts.cpu().numpy()
        ignore_masks_np = ignore_masks.cpu().numpy()
        
        for b in range(batch_size):
            for pos in range(seq_len):
                if not ignore_masks_np[b, pos]:
                    continue
                
                piece_cat = categories[b][pos]
                if piece_cat is None:
                    continue
                
                # Update counts for all activated features at this position
                for feat_idx in range(d_sae):
                    act_value = feature_acts_np[b, pos, feat_idx]
                    if act_value > 0:
                        if piece_cat not in feature_category_counts[feat_idx]:
                            feature_category_counts[feat_idx][piece_cat] = 0
                        feature_category_counts[feat_idx][piece_cat] += 1

    def compute_ignore_token_masks(
        self, tokens: torch.Tensor, ignore_token_ids: Optional[list[int]] = None
    ) -> torch.Tensor:
        """Compute ignore token masks for the given tokens.
        
        Args:
            tokens: The tokens to compute the ignore token masks for
            ignore_token_ids: The token IDs to ignore
            
        Returns:
            Boolean mask tensor
        """
        if ignore_token_ids is None:
            warnings.warn(
                "ignore_token_ids are not provided. No tokens (including pad tokens) will be filtered out.",
                UserWarning,
                stacklevel=2,
            )
            ignore_token_ids = []
        mask = torch.ones_like(tokens, dtype=torch.bool)
        for token_id in ignore_token_ids:
            mask &= tokens != token_id
        return mask

    def _get_fens_from_dataset_batch(
        self,
        dataset_names: List[str],
        context_indices: List[int],
        shard_indices: Optional[List[int]] = None,
        n_shards_list: Optional[List[int]] = None,
        dataset_path: Optional[str] = None,
    ) -> List[Optional[str]]:
        """Get FEN strings from dataset in batch using context_idx and shard_idx.
        
        Args:
            dataset_names: List of dataset names for each sample
            context_indices: List of indices in the dataset
            shard_indices: List of shard indices (optional)
            n_shards_list: List of number of shards (optional)
            dataset_path: Path to the dataset directory (if None, uses dataset_name directly)
            
        Returns:
            List of FEN strings (or None if not found)
        """
        try:
            from datasets import load_from_disk
            
            # Get dataset from cache or load it
            if not hasattr(self, '_dataset_cache'):
                self._dataset_cache: Dict[str, Any] = {}
            
            batch_size = len(dataset_names)
            fens = []
            
            # Group samples by dataset and shard for efficient batch loading
            dataset_groups: Dict[tuple, List[tuple]] = {}  # (dataset_name, shard_idx, n_shards) -> [(batch_idx, context_idx)]
            
            for b in range(batch_size):
                dataset_name = dataset_names[b]
                context_idx = context_indices[b]
                shard_idx = shard_indices[b] if shard_indices is not None else None
                n_shards = n_shards_list[b] if n_shards_list is not None else None
                
                key = (dataset_name, shard_idx, n_shards)
                if key not in dataset_groups:
                    dataset_groups[key] = []
                dataset_groups[key].append((b, context_idx))
            
            # Initialize result list
            fens = [None] * batch_size
            
            # Process each dataset group
            for (dataset_name, shard_idx, n_shards), indices in dataset_groups.items():
                # Load dataset if not cached
                if dataset_name not in self._dataset_cache:
                    # Use provided dataset_path or dataset_name as path
                    path_to_load = dataset_path if dataset_path is not None else dataset_name
                    try:
                        dataset = load_from_disk(path_to_load)
                        self._dataset_cache[dataset_name] = dataset
                    except Exception as e:
                        warnings.warn(f"Failed to load dataset from {path_to_load}: {e}", UserWarning)
                        continue
                
                dataset = self._dataset_cache[dataset_name]
                
                # Get the appropriate dataset shard
                if shard_idx is not None and n_shards is not None:
                    try:
                        dataset_to_use = dataset.shard(n_shards, shard_idx) if hasattr(dataset, "shard") else dataset
                    except Exception:
                        dataset_to_use = dataset
                else:
                    dataset_to_use = dataset
                
                # Batch load samples from this dataset/shard
                context_idx_list = [ctx_idx for _, ctx_idx in indices]
                try:
                    # Try to use select for batch loading (more efficient)
                    if hasattr(dataset_to_use, "select"):
                        samples = dataset_to_use.select(context_idx_list)
                        for i, (batch_idx, _) in enumerate(indices):
                            sample = samples[i]
                            prompt = sample.get("prompt") or sample.get("text") or sample.get("fen") or sample
                            fen = prompt.strip() if isinstance(prompt, str) and len(prompt.strip().split()) >= 4 else (sample.get("fen") if isinstance(sample, dict) else None)
                            fens[batch_idx] = fen
                    else:
                        # Fallback to individual access
                        for batch_idx, context_idx in indices:
                            sample = dataset_to_use[int(context_idx)]
                            prompt = sample.get("prompt") or sample.get("text") or sample.get("fen") or sample
                            fen = prompt.strip() if isinstance(prompt, str) and len(prompt.strip().split()) >= 4 else (sample.get("fen") if isinstance(sample, dict) else None)
                            fens[batch_idx] = fen
                except Exception as e:
                    # Fallback to individual access on error
                    for batch_idx, context_idx in indices:
                        try:
                            sample = dataset_to_use[int(context_idx)]
                            prompt = sample.get("prompt") or sample.get("text") or sample.get("fen") or sample
                            fen = prompt.strip() if isinstance(prompt, str) and len(prompt.strip().split()) >= 4 else (sample.get("fen") if isinstance(sample, dict) else None)
                            fens[batch_idx] = fen
                        except Exception:
                            fens[batch_idx] = None
            
            return fens
        except Exception as e:
            warnings.warn(f"Failed to get FENs from dataset batch: {e}", UserWarning)
            return [None] * len(dataset_names)

    @torch.no_grad()
    def analyze_entropy(
        self,
        activation_factory: ActivationFactory,
        sae: AbstractSparseAutoEncoder,
        device_mesh: DeviceMesh | None = None,
        activation_factory_process_kwargs: dict[str, Any] = {},
        dataset_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze feature activation entropy over the dataset.
        
        This method processes activations in batches, collects statistics about
        which piece types activate each feature, and computes entropy metrics.
        
        Args:
            activation_factory: The activation factory to use
            sae: The sparse autoencoder model
            device_mesh: The device mesh to use for distributed processing
            activation_factory_process_kwargs: Keyword arguments to pass to activation factory
            dataset_path: Path to the dataset directory (for loading FENs)
            
        Returns:
            Dictionary containing:
            - per_feature_entropy: List of entropy values per feature
            - mean_entropy: Average entropy across all features
            - feature_details: Detailed statistics per feature (optional)
        """
        activation_stream = activation_factory.process(**activation_factory_process_kwargs)
        n_tokens = 0

        # Progress tracking
        pbar = tqdm(
            total=self.cfg.total_analyzing_tokens,
            desc="Computing Feature Entropy",
            smoothing=0.01,
            disable=not is_primary_rank(device_mesh),
        )

        if device_mesh is not None and device_mesh.mesh_dim_names is not None and "model" in device_mesh.mesh_dim_names:
            d_sae_local = sae.cfg.d_sae // device_mesh["model"].size()
        else:
            d_sae_local = sae.cfg.d_sae

        # Initialize storage for each feature: Dict[str, int] containing category counts
        # feature_category_counts[feat_idx][category] = count
        feature_category_counts: List[Dict[str, int]] = [{} for _ in range(d_sae_local)]

        if sae.cfg.sae_type == "clt":
            sae.encode = partial(sae.encode_single_layer, layer=self.cfg.clt_layer)
            sae.prepare_input = partial(sae.prepare_input_single_layer, layer=self.cfg.clt_layer)
            sae.keep_only_decoders_for_layer_from(self.cfg.clt_layer)
            torch.cuda.empty_cache()

        # Process activation batches
        for batch in activation_stream:
            # Get metadata (context_idx, shard_idx, etc.)
            meta = {k: [m[k] for m in batch["meta"]] for k in batch["meta"][0].keys()}

            # Get feature activations from SAE
            x, kwargs = sae.prepare_input(batch)
            feature_acts: torch.Tensor = sae.encode(x, **kwargs)

            if isinstance(feature_acts, torch.distributed.tensor.DTensor):
                feature_acts = feature_acts.redistribute(
                    placements=[torch.distributed.tensor.Replicate()] * len(feature_acts.placements)
                ).to_local()

            if isinstance(sae, LowRankSparseAttention) and sae.cfg.skip_bos:
                feature_acts[:, 0, :] = 0

            assert feature_acts.shape == (
                batch["tokens"].shape[0],
                batch["tokens"].shape[1],
                d_sae_local,
            ), f"feature_acts.shape: {feature_acts.shape}"

            # Compute ignore token masks
            ignore_token_masks = self.compute_ignore_token_masks(
                batch["tokens"], self.cfg.ignore_token_ids
            )
            feature_acts *= repeat(ignore_token_masks, "batch_size n_ctx -> batch_size n_ctx 1")

            # Collect activation data with piece type categories
            batch_size, seq_len, _ = feature_acts.shape

            # Get FENs for each sample in the batch (batch processing for efficiency)
            if "fen" in meta:
                # FENs are directly available in metadata
                fens = meta["fen"]
            elif "context_idx" in meta:
                # Need to fetch FENs from dataset using context_idx and shard_idx
                dataset_names = [meta.get("dataset_name", ["master"] * batch_size)[b] if "dataset_name" in meta else "master" for b in range(batch_size)]
                context_indices = [int(meta["context_idx"][b]) for b in range(batch_size)]
                shard_indices = [int(meta["shard_idx"][b]) for b in range(batch_size)] if "shard_idx" in meta else None
                n_shards_list = [int(meta["n_shards"][b]) for b in range(batch_size)] if "n_shards" in meta else None
                
                # Batch fetch FENs
                fens = self._get_fens_from_dataset_batch(
                    dataset_names, 
                    context_indices, 
                    shard_indices, 
                    n_shards_list, 
                    dataset_path
                )
                
            else:
                # No FEN information available
                fens = [None] * batch_size
            # print(f'{fens = }')
            # Get piece type categories for all positions in the batch
            categories = self.get_piece_type_categories_batch(fens, seq_len)
            
            # Update feature counts using batch processing
            self.update_feature_counts_batch(
                feature_acts,
                categories,
                ignore_token_masks,
                feature_category_counts
            )
            print("update_feature_counts_batch")
            # Update progress
            n_tokens_current = batch["tokens"].numel()
            n_tokens += n_tokens_current
            pbar.update(n_tokens_current)

            if n_tokens >= self.cfg.total_analyzing_tokens:
                break

        pbar.close()

        # Compute entropy for each feature from the collected category counts
        per_feature_entropy = []
        feature_details = []

        for feat_idx in range(d_sae_local):
            counts = feature_category_counts[feat_idx]
            
            # Use the unified entropy computation function
            entropy_info = self.compute_entropy_from_counts(counts)
            
            per_feature_entropy.append(entropy_info["entropy"])
            feature_details.append({
                "feature_idx": feat_idx,
                **entropy_info
            })

        # Compute mean entropy (only over features that have activations)
        valid_entropies = [e for e in per_feature_entropy if e > 0]
        mean_entropy = np.mean(valid_entropies) if valid_entropies else 0.0

        return {
            "per_feature_entropy": per_feature_entropy,
            "mean_entropy": float(mean_entropy),
            "feature_details": feature_details,
            "n_features": d_sae_local,
        }
