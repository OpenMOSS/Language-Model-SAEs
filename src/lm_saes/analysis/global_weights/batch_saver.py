"""Utility module for saving activation data to disk using safetensors (one file per sentence)."""

from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file, load_file
from jaxtyping import Float


def sparse_to_dict(sparse_tensor: torch.sparse.Tensor) -> dict[str, torch.Tensor]:
    """Convert a sparse tensor to a dictionary of indices and values.
    
    Args:
        sparse_tensor: Sparse COO tensor to convert
        
    Returns:
        Dictionary with keys:
            - 'indices': (ndim, nnz) tensor of indices
            - 'values': (nnz,) tensor of values
            - 'size': (ndim,) tensor of original tensor size
            - 'dtype': string representation of dtype
    """
    sparse_tensor = sparse_tensor.coalesce()
    return {
        "indices": sparse_tensor.indices().cpu(),
        "values": sparse_tensor.values().cpu(),
        "size": torch.tensor(sparse_tensor.shape, dtype=torch.long),
        "dtype": str(sparse_tensor.dtype),
    }


def dict_to_sparse(data: dict[str, torch.Tensor], device: Optional[torch.device] = None) -> torch.sparse.Tensor:
    """Convert a dictionary back to a sparse tensor.
    
    Args:
        data: Dictionary with 'indices', 'values', 'size', and 'dtype' keys
        device: Device to place the tensor on
        
    Returns:
        Sparse COO tensor
    """
    dtype = getattr(torch, data["dtype"].split(".")[-1])
    indices = data["indices"]
    values = data["values"]
    size = tuple(data["size"].tolist())
    
    if device is not None:
        indices = indices.to(device)
        values = values.to(device)
    
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values.to(dtype),
        size=size,
        device=device if device is not None else indices.device,
    )


def extract_attention_patterns_coo(
    attention_pattern: Float[torch.Tensor, "n_layer n_heads q_pos k_pos"],
    threshold: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Extract attention patterns as list of list of 2D COO sparse tensors with thresholding.
    
    For each (layer, head) pair, creates a 2D sparse COO tensor of shape (q_pos, k_pos).
    Values below the threshold are set to 0 and excluded from the sparse representation.
    
    Args:
        attention_pattern: Attention pattern tensor with shape (n_layer, n_heads, q_pos, k_pos)
        threshold: Threshold value below which values are set to 0 (default: 0.01)
        
    Returns:
        Dictionary containing:
            - Keys like 'lorsa_attention_pattern.{layer}.{head}.indices', 
              'lorsa_attention_pattern.{layer}.{head}.values', and
              'lorsa_attention_pattern.{layer}.{head}.size' for each layer-head pair
            - 'lorsa_attention_pattern.shape': Original shape (n_layer, n_heads, q_pos, k_pos)
    """
    n_layer, n_heads, q_pos, k_pos = attention_pattern.shape
    result = {}
    
    # Save shape (dtype will be inferred from values tensors when loading)
    result["lorsa_attention_pattern.shape"] = torch.tensor([n_layer, n_heads, q_pos, k_pos], dtype=torch.long)
    
    # Process each layer-head pair
    for layer in range(n_layer):
        for head in range(n_heads):
            # Get 2D pattern for this layer-head
            pattern_2d = attention_pattern[layer, head]  # (q_pos, k_pos)
            
            # Threshold: set values < threshold to 0
            pattern_2d = pattern_2d.clone()
            pattern_2d[pattern_2d < threshold] = 0.0
            
            # Convert to sparse COO tensor
            sparse_2d = pattern_2d.to_sparse_coo().coalesce()
            
            # Save indices and values
            key_prefix = f"lorsa_attention_pattern.{layer}.{head}"
            result[f"{key_prefix}.indices"] = sparse_2d.indices().cpu()
            result[f"{key_prefix}.values"] = sparse_2d.values().cpu()
            result[f"{key_prefix}.size"] = torch.tensor([q_pos, k_pos], dtype=torch.long)
    
    return result


def reconstruct_attention_patterns_coo(
    data: dict[str, torch.Tensor],
    device: Optional[torch.device] = None,
) -> Float[torch.Tensor, "n_layer n_heads q_pos k_pos"]:
    """Reconstruct attention pattern from list of list of 2D COO sparse tensors.
    
    Args:
        data: Dictionary with keys like 'lorsa_attention_pattern.{layer}.{head}.indices',
              'lorsa_attention_pattern.{layer}.{head}.values', 
              'lorsa_attention_pattern.{layer}.{head}.size', and
              'lorsa_attention_pattern.shape'
        device: Device to place the tensor on
        
    Returns:
        Reconstructed attention pattern tensor of shape (n_layer, n_heads, q_pos, k_pos)
    """
    # Get shape information
    shape = data["lorsa_attention_pattern.shape"].tolist()
    n_layer, n_heads, q_pos, k_pos = shape
    
    # Determine dtype from first available values tensor
    dtype_str = None
    for layer in range(n_layer):
        for head in range(n_heads):
            key = f"lorsa_attention_pattern.{layer}.{head}.values"
            if key in data:
                dtype_str = str(data[key].dtype)
                break
        if dtype_str is not None:
            break
    
    if dtype_str is None:
        raise ValueError("No attention pattern data found")
    
    dtype = getattr(torch, dtype_str.split(".")[-1])
    
    # Reconstruct each layer-head pattern
    patterns = []
    for layer in range(n_layer):
        layer_patterns = []
        for head in range(n_heads):
            key_prefix = f"lorsa_attention_pattern.{layer}.{head}"
            indices_key = f"{key_prefix}.indices"
            values_key = f"{key_prefix}.values"
            size_key = f"{key_prefix}.size"
            
            if indices_key in data and values_key in data:
                indices = data[indices_key]
                values = data[values_key]
                size = tuple(data[size_key].tolist())
                
                if device is not None:
                    indices = indices.to(device)
                    values = values.to(device)
                
                # Reconstruct sparse tensor
                sparse_2d = torch.sparse_coo_tensor(
                    indices=indices,
                    values=values.to(dtype),
                    size=size,
                    device=device if device is not None else indices.device,
                )
                # Convert to dense
                pattern_2d = sparse_2d.to_dense()
            else:
                # If data is missing, create zero tensor
                pattern_2d = torch.zeros(q_pos, k_pos, dtype=dtype, device=device)
            
            layer_patterns.append(pattern_2d)
        patterns.append(torch.stack(layer_patterns))
    
    return torch.stack(patterns).to(dtype)


class SampleSaver:
    """Saves activation data for a single sentence to a safetensors file."""
    
    def __init__(
        self,
        output_dir: str | Path,
    ):
        """Initialize the sample saver.
        
        Args:
            output_dir: Directory to save files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_idx = 0
    
    def save_sample(
        self,
        lorsa_activation_matrix: Optional[torch.sparse.Tensor],
        lorsa_attention_pattern: Optional[Float[torch.Tensor, "n_layer n_heads q_pos k_pos"]],
        clt_activation_matrix: torch.sparse.Tensor,
        attn_hook_scales: Float[torch.Tensor, "n_layer pos"],
        mlp_hook_scales: Float[torch.Tensor, "n_layer pos"],
        input_ids: torch.Tensor,
    ) -> str:
        """Save a single sample to disk.
        
        Args:
            lorsa_activation_matrix: Sparse lorsa activation matrix (can be None)
            lorsa_attention_pattern: Attention pattern tensor (can be None)
            clt_activation_matrix: Sparse CLT activation matrix
            attn_hook_scales: Attention hook scales
            mlp_hook_scales: MLP hook scales
            input_ids: Token IDs tensor
            
        Returns:
            Path to saved file
        """
        sample_data = {}
        
        # Handle lorsa_activation_matrix
        if lorsa_activation_matrix is not None:
            lorsa_data = sparse_to_dict(lorsa_activation_matrix)
            sample_data["lorsa_activation_matrix.indices"] = lorsa_data["indices"]
            sample_data["lorsa_activation_matrix.values"] = lorsa_data["values"]
            sample_data["lorsa_activation_matrix.size"] = lorsa_data["size"]
        
        # Handle lorsa_attention_pattern
        if lorsa_attention_pattern is not None:
            pattern_data = extract_attention_patterns_coo(lorsa_attention_pattern, threshold=0.01)
            # Merge pattern_data into sample_data (includes shape and all layer-head tensors)
            sample_data.update(pattern_data)
        
        # Handle clt_activation_matrix
        clt_data = sparse_to_dict(clt_activation_matrix)
        sample_data["clt_activation_matrix.indices"] = clt_data["indices"]
        sample_data["clt_activation_matrix.values"] = clt_data["values"]
        sample_data["clt_activation_matrix.size"] = clt_data["size"]
        
        # Handle hook scales
        sample_data["attn_hook_scales"] = attn_hook_scales.cpu()
        sample_data["mlp_hook_scales"] = mlp_hook_scales.cpu()
        
        # Handle input_ids
        sample_data["input_ids"] = input_ids.cpu()
        
        # Save to file
        file_path = self.output_dir / f"sample_{self.sample_idx:08d}.safetensors"
        save_file(sample_data, file_path)
        
        self.sample_idx += 1
        return str(file_path)


def load_sample(file_path: str | Path, device: Optional[torch.device] = None) -> dict:
    """Load a sample file from disk.
    
    Args:
        file_path: Path to the safetensors file
        device: Device to load tensors onto
        
    Returns:
        Dictionary with keys:
            - 'lorsa_activation_matrix': Sparse tensor (or None)
            - 'lorsa_attention_pattern': Attention pattern tensor (or None)
            - 'clt_activation_matrix': Sparse tensor
            - 'attn_hook_scales': Tensor of shape (n_layer, pos)
            - 'mlp_hook_scales': Tensor of shape (n_layer, pos)
            - 'input_ids': Tensor of token IDs
    """
    data = load_file(str(file_path))
    
    result = {}
    
    # Reconstruct lorsa_activation_matrix if present
    if "lorsa_activation_matrix.indices" in data:
        sample_data = {
            "indices": data["lorsa_activation_matrix.indices"],
            "values": data["lorsa_activation_matrix.values"],
            "size": data["lorsa_activation_matrix.size"],
            "dtype": str(data["lorsa_activation_matrix.values"].dtype),
        }
        result["lorsa_activation_matrix"] = dict_to_sparse(sample_data, device=device)
    else:
        result["lorsa_activation_matrix"] = None
    
    # Reconstruct lorsa_attention_pattern if present
    if "lorsa_attention_pattern.shape" in data:
        result["lorsa_attention_pattern"] = reconstruct_attention_patterns_coo(data, device=device)
    else:
        result["lorsa_attention_pattern"] = None
    
    # Reconstruct clt_activation_matrix
    clt_data = {
        "indices": data["clt_activation_matrix.indices"],
        "values": data["clt_activation_matrix.values"],
        "size": data["clt_activation_matrix.size"],
        "dtype": str(data["clt_activation_matrix.values"].dtype),
    }
    result["clt_activation_matrix"] = dict_to_sparse(clt_data, device=device)
    
    # Load hook scales
    result["attn_hook_scales"] = data["attn_hook_scales"].to(device) if device is not None else data["attn_hook_scales"]
    result["mlp_hook_scales"] = data["mlp_hook_scales"].to(device) if device is not None else data["mlp_hook_scales"]
    
    # Load input_ids
    result["input_ids"] = data["input_ids"].to(device) if device is not None else data["input_ids"]
    
    return result


class SampleDataset(Dataset):
    """PyTorch Dataset for loading activation samples from safetensors files.
    
    Scans a directory for sample files matching the pattern 'sample_*.safetensors'
    and provides efficient loading with multiprocessing support.
    """
    
    def __init__(
        self,
        data_dir: str | Path,
        device: Optional[torch.device] = None,
        pattern: str = "sample_*.safetensors",
    ):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing sample safetensors files
            device: Device to load tensors onto (None for CPU)
            pattern: Glob pattern to match sample files (default: "sample_*.safetensors")
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        self.device = device
        # Find all matching files and sort them for consistent ordering
        self.file_paths = sorted(self.data_dir.glob(pattern))
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No files found matching pattern '{pattern}' in {self.data_dir}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> dict:
        """Load and return a single sample.
        
        Args:
            idx: Index of the sample to load
            
        Returns:
            Dictionary with keys:
                - 'lorsa_activation_matrix': Sparse tensor (or None)
                - 'lorsa_attention_pattern': Attention pattern tensor (or None)
                - 'clt_activation_matrix': Sparse tensor
                - 'attn_hook_scales': Tensor of shape (n_layer, pos)
                - 'mlp_hook_scales': Tensor of shape (n_layer, pos)
                - 'input_ids': Tensor of token IDs
        """
        file_path = self.file_paths[idx]
        return load_sample(file_path, device=self.device)


def collate_single_sample(batch: Sequence[dict]) -> dict:
    """Custom collate function for single sample loading with sparse tensor support.
    
    Since batch_size=1, this simply returns the first (and only) sample.
    This avoids the default collate function's attempt to stack sparse tensors.
    
    Args:
        batch: List containing a single sample dictionary
        
    Returns:
        The single sample dictionary (not wrapped in a batch)
    """
    # With batch_size=1, batch will always contain exactly one element
    return batch[0]


def create_dataloader(
    data_dir: str | Path,
    shuffle: bool = False,
    num_workers: int = 0,
    prefetch_factor: Optional[int] = None,
    device: Optional[torch.device] = 'cuda',
    pin_memory: bool = False,
    pattern: str = "sample_*.safetensors",
    **dataloader_kwargs,
) -> DataLoader:
    """Create a DataLoader for loading activation samples one at a time with multiprocessing support.
    
    Args:
        data_dir: Directory containing sample safetensors files
        shuffle: Whether to shuffle the dataset (default: False)
        num_workers: Number of worker processes for data loading (default: 0, use main process)
                     Set to > 0 to enable multiprocessing
        prefetch_factor: Number of samples to prefetch per worker (default: 2 if num_workers > 0, else None)
        device: Device to load tensors onto (None for CPU). Note: if using multiprocessing,
                tensors will be loaded on CPU in workers and moved to device in the main process.
        pin_memory: Whether to pin memory for faster GPU transfer (default: False)
        pattern: Glob pattern to match sample files (default: "sample_*.safetensors")
        **dataloader_kwargs: Additional arguments to pass to DataLoader
        
    Returns:
        DataLoader instance configured for loading samples one at a time
        
    Note:
        When using multiprocessing (num_workers > 0), it's recommended to:
        - Set device=None in the Dataset (loads on CPU in workers)
        - Set pin_memory=True if using GPU (faster CPU->GPU transfer)
        - Move tensors to device in the training loop after getting samples
        
    Example:
        >>> # Single process loading
        >>> loader = create_dataloader("/path/to/data", device="cuda")
        >>> 
        >>> # Multiprocessing with prefetching
        >>> loader = create_dataloader(
        ...     "/path/to/data",
        ...     num_workers=4,
        ...     prefetch_factor=2,
        ...     pin_memory=True,
        ... )
        >>> for sample in loader:
        ...     # Move to device if using multiprocessing
        ...     sample = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v 
        ...              for k, v in sample.items()}
        ...     # Process sample...
    """
    dataset = SampleDataset(data_dir, device=None if num_workers > 0 else device, pattern=pattern)
    
    # Set default prefetch_factor for multiprocessing
    if prefetch_factor is None and num_workers > 0:
        prefetch_factor = 2
    
    return DataLoader(
        dataset,
        batch_size=1,  # Always load one sample at a time
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        collate_fn=collate_single_sample,  # Handle sparse tensors correctly
        **dataloader_kwargs,
    )
