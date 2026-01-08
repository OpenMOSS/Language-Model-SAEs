import base64
import io
import json
import os
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Generator, Optional
from urllib.parse import urlparse, parse_qs

import msgpack
import numpy as np
import plotly.graph_objects as go
import torch
import yaml
from datasets import Dataset
from fastapi import Body, FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import sys
sys.path.append("pytorch-grad-cam")

try:
    from torchvision import transforms
except ImportError:
    transforms = None
    print("WARNING: torchvision not found, image processing will be disabled")

from lm_saes.backend import LanguageModel
from lm_saes.config import MongoDBConfig, SAEConfig
from lm_saes.database import MongoClient
from lm_saes.resource_loaders import load_dataset_shard, load_model
from lm_saes.sae import SparseAutoEncoder
from torchvision.transforms import v2
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    LayerCAM,
    ScoreCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from lm_saes.xai.sae_feature_classifier import SAEFeatureClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Feature Visualize (CNNSAE + diffusion posterior sampling) support ---
# Note: These are optional dependencies and may not be available on all servers
ROOT_DIR = Path(__file__).resolve().parents[1]
DIFF_DIR = ROOT_DIR / "diffusion-posterior-sampling"

CNNSAEFeatureMaxConfig = None
generate_with_point_supervision = None

if DIFF_DIR.exists() and (DIFF_DIR / "cnnsae_feature_max.py").exists():
    try:
        # Use importlib to load the module with proper package structure
        import importlib.util
        import types
        
        print(f"Attempting to load diffusion modules from {DIFF_DIR}")
        
        # Add DIFF_DIR to sys.path so that 'util' and 'motionblur' modules can be found
        if str(DIFF_DIR) not in sys.path:
            sys.path.insert(0, str(DIFF_DIR))
            print(f"  Added {DIFF_DIR} to sys.path")
        
        # Create package structure for relative imports
        pkg_name = "diffusion_posterior_sampling"
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(DIFF_DIR)]  # Set __path__ for package recognition
            sys.modules[pkg_name] = pkg
        
        # Load guided_diffusion subpackage
        guided_pkg_name = f"{pkg_name}.guided_diffusion"
        if guided_pkg_name not in sys.modules:
            guided_pkg = types.ModuleType(guided_pkg_name)
            guided_pkg.__path__ = [str(DIFF_DIR / "guided_diffusion")]
            setattr(sys.modules[pkg_name], "guided_diffusion", guided_pkg)
            sys.modules[guided_pkg_name] = guided_pkg
        
        # Load required submodules first (for relative imports)
        # Order matters: load dependencies first
        submodules = ["unet", "gaussian_diffusion", "condition_methods", "measurements"]
        for submod_name in submodules:
            submod_path = DIFF_DIR / "guided_diffusion" / f"{submod_name}.py"
            if submod_path.exists():
                full_name = f"{guided_pkg_name}.{submod_name}"
                if full_name not in sys.modules:
                    spec = importlib.util.spec_from_file_location(full_name, submod_path)
                    if spec is not None and spec.loader is not None:
                        submod = importlib.util.module_from_spec(spec)
                        submod.__package__ = guided_pkg_name
                        submod.__name__ = full_name
                        submod.__file__ = str(submod_path)
                        sys.modules[full_name] = submod
                        try:
                            spec.loader.exec_module(submod)
                            print(f"  ✓ Loaded {submod_name}")
                        except Exception as e:
                            print(f"  ✗ Failed to load {submod_name}: {e}")
                            import traceback
                            traceback.print_exc()
            else:
                print(f"  ⚠ {submod_name}.py not found at {submod_path}")
        
        # Now load the main module
        main_module_path = DIFF_DIR / "cnnsae_feature_max.py"
        main_module_name = f"{pkg_name}.cnnsae_feature_max"
        spec = importlib.util.spec_from_file_location(main_module_name, main_module_path)
        if spec is not None and spec.loader is not None:
            main_module = importlib.util.module_from_spec(spec)
            # Set necessary attributes for relative imports
            main_module.__package__ = pkg_name
            main_module.__name__ = main_module_name
            main_module.__file__ = str(main_module_path)
            sys.modules[main_module_name] = main_module
            
            try:
                spec.loader.exec_module(main_module)
                CNNSAEFeatureMaxConfig = main_module.CNNSAEFeatureMaxConfig
                generate_with_point_supervision = main_module.generate_with_point_supervision
                print("✓ Successfully loaded diffusion feature visualize modules")
            except Exception as e:
                print(f"✗ Failed to execute cnnsae_feature_max module: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"✗ Failed to create module spec for cnnsae_feature_max")
    except Exception as e:  # pragma: no cover - optional dependency
        print(f"WARNING: failed to import diffusion feature visualize modules: {e}")
        import traceback
        traceback.print_exc()
else:
    if not DIFF_DIR.exists():
        print(f"WARNING: diffusion-posterior-sampling directory not found at {DIFF_DIR}")
    else:
        print(f"WARNING: cnnsae_feature_max.py not found in {DIFF_DIR}")

DEFAULT_FEATURE_VIZ_TASK_CFG = DIFF_DIR / "configs" / "cnnsae_feature_max_config.yaml" if DIFF_DIR.exists() else None
DEFAULT_FEATURE_VIZ_MODEL_CFG = DIFF_DIR / "configs" / "imagenet_model_config.yaml" if DIFF_DIR.exists() else None
DEFAULT_FEATURE_VIZ_DIFFUSION_CFG = DIFF_DIR / "configs" / "diffusion_config.yaml" if DIFF_DIR.exists() else None

# Feature visualize config directories
FEATURE_VIZ_CONFIG_DIR = ROOT_DIR / "feature_visualize_config"
FEATURE_VIZ_DIFFUSION_DIR = FEATURE_VIZ_CONFIG_DIR / "Diffusion" if FEATURE_VIZ_CONFIG_DIR.exists() else None
FEATURE_VIZ_SAE_DIR = FEATURE_VIZ_CONFIG_DIR / "SAE" if FEATURE_VIZ_CONFIG_DIR.exists() else None


def _build_feature_viz_config(
    *,
    seed: Optional[int] = None,
    image_size: Optional[int] = None,
    batch_size: int = 1,
    device_override: Optional[str] = None,
    diffusion_config: Optional[str] = None,
    sae_config: Optional[str] = None,
) -> Optional["CNNSAEFeatureMaxConfig"]:
    """
    Create a CNNSAEFeatureMaxConfig using config files.
    If diffusion_config and sae_config are provided, use them from feature_visualize_config.
    Otherwise, use default configs from diffusion-posterior-sampling.
    """
    if CNNSAEFeatureMaxConfig is None:
        return None
    
    try:
        import yaml
        import tempfile
        
        # Determine which configs to use
        if diffusion_config and sae_config and FEATURE_VIZ_DIFFUSION_DIR and FEATURE_VIZ_SAE_DIR:
            # Use custom configs from feature_visualize_config
            diffusion_config_path = FEATURE_VIZ_DIFFUSION_DIR / diffusion_config
            sae_config_path = FEATURE_VIZ_SAE_DIR / sae_config
            
            if not diffusion_config_path.exists():
                print(f"WARNING: Diffusion config not found: {diffusion_config_path}")
                return None
            if not sae_config_path.exists():
                print(f"WARNING: SAE config not found: {sae_config_path}")
                return None
            
            # Load diffusion config to get model path
            with open(diffusion_config_path, 'r') as f:
                diffusion_cfg = yaml.safe_load(f)
            model_path = diffusion_cfg.get("path", "")
            if not model_path:
                print(f"WARNING: No model path in diffusion config")
                return None
            
            # Convert relative path to absolute if needed
            model_path_obj = Path(model_path)
            if not model_path_obj.is_absolute():
                # Try relative to DIFF_DIR first
                abs_model_path = DIFF_DIR / model_path_obj
                if not abs_model_path.exists():
                    # Try relative to ROOT_DIR
                    abs_model_path = ROOT_DIR / model_path_obj
                model_path = str(abs_model_path) if abs_model_path.exists() else model_path
            
            # Create a temporary model config file
            model_cfg = {
                "image_size": image_size or 256,
                "num_channels": 256,
                "num_res_blocks": 2,
                "channel_mult": "",
                "learn_sigma": True,
                "class_cond": False,
                "use_checkpoint": False,
                "attention_resolutions": "32,16,8",
                "num_heads": 4,
                "num_head_channels": 64,
                "num_heads_upsample": -1,
                "use_scale_shift_norm": True,
                "dropout": 0.0,
                "resblock_updown": True,
                "use_fp16": False,
                "use_new_attention_order": False,
                "model_path": model_path,
            }
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
                yaml.dump(model_cfg, tmp, default_flow_style=False)
                model_config_path = tmp.name
            
            # Use default diffusion config (sampling parameters)
            diffusion_config_path = str(DEFAULT_FEATURE_VIZ_DIFFUSION_CFG) if DEFAULT_FEATURE_VIZ_DIFFUSION_CFG and DEFAULT_FEATURE_VIZ_DIFFUSION_CFG.exists() else str(DIFF_DIR / "configs" / "diffusion_config.yaml")
            task_config_path = str(sae_config_path)
        else:
            # Use default configs
            if (
                DEFAULT_FEATURE_VIZ_TASK_CFG is None
                or DEFAULT_FEATURE_VIZ_MODEL_CFG is None
                or DEFAULT_FEATURE_VIZ_DIFFUSION_CFG is None
                or not DEFAULT_FEATURE_VIZ_TASK_CFG.exists()
                or not DEFAULT_FEATURE_VIZ_MODEL_CFG.exists()
                or not DEFAULT_FEATURE_VIZ_DIFFUSION_CFG.exists()
            ):
                print("WARNING: Feature visualize config files not found")
                return None
            
            # Load model config and fix model_path if it's relative
            with open(DEFAULT_FEATURE_VIZ_MODEL_CFG, 'r') as f:
                model_cfg = yaml.safe_load(f)
            
            model_config_path = str(DEFAULT_FEATURE_VIZ_MODEL_CFG)
            if 'model_path' in model_cfg:
                model_path = Path(model_cfg['model_path'])
                if not model_path.is_absolute():
                    # Convert relative path to absolute path relative to DIFF_DIR
                    abs_model_path = DIFF_DIR / model_path
                    if abs_model_path.exists():
                        model_cfg['model_path'] = str(abs_model_path)
                        # Create a temporary file with the modified config
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
                            yaml.dump(model_cfg, tmp, default_flow_style=False)
                            model_config_path = tmp.name
                        print(f"  Fixed model_path: {model_path} -> {abs_model_path}")
                    else:
                        print(f"  WARNING: Model file not found at {abs_model_path}, using original path")
            
            diffusion_config_path = str(DEFAULT_FEATURE_VIZ_DIFFUSION_CFG)
            task_config_path = str(DEFAULT_FEATURE_VIZ_TASK_CFG)
        
        cfg = CNNSAEFeatureMaxConfig.from_yaml(
            model_config_path=model_config_path,
            diffusion_config_path=diffusion_config_path,
            task_config_path=task_config_path,
            device=device_override or device,
            seed=seed,
        )
        # override dynamic fields (dataclass is frozen -> use replace)
        cfg = replace(cfg, batch_size=batch_size)
        if image_size is not None:
            cfg = replace(cfg, image_size=int(image_size))
        return cfg
    except Exception as e:
        print(f"WARNING: Failed to build feature visualize config: {e}")
        import traceback
        traceback.print_exc()
        return None


def _tensor_to_base64_image(img: torch.Tensor) -> str:
    """
    Convert a tensor in [-1,1] range with shape [3,H,W] to PNG base64 string.
    """
    img = img.detach().cpu().clamp(-1, 1)
    img = (img + 1) * 0.5  # [0,1]
    img = (img * 255).byte()
    img = img.permute(1, 2, 0).numpy()  # HWC
    from PIL import Image

    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=1000)

print(os.environ.get("MONGO_URI", "mongodb://localhost:27017/"))
print(os.environ.get("MONGO_DB", "mechinterp"))
client = MongoClient(MongoDBConfig())
sae_series = os.environ.get("SAE_SERIES", "default")
tokenizer_only = os.environ.get("TOKENIZER_ONLY", "false").lower() == "true"
if tokenizer_only:
    print("WARNING: Tokenizer only mode is enabled, some features may not be available")

# Remove global caches in favor of LRU cache
# sae_cache: dict[str, SparseAutoEncoder] = {}
# lm_cache: dict[str, LanguageModel] = {}
# dataset_cache: dict[tuple[str, int, int], Dataset] = {}


@lru_cache(maxsize=8)
def get_model(name: str) -> LanguageModel:
    """Load and cache a language model.

    Args:
        name: Name of the model to load

    Returns:
        LanguageModel: The loaded model

    Raises:
        ValueError: If the model is not found
    """
    cfg = client.get_model_cfg(name)
    if cfg is None:
        raise ValueError(f"Model {name} not found")
    cfg.tokenizer_only = tokenizer_only
    return load_model(cfg)


@lru_cache(maxsize=16)
def get_dataset(name: str, shard_idx: int = 0, n_shards: int = 1) -> Dataset:
    """Load and cache a dataset shard.

    Args:
        name: Name of the dataset
        shard_idx: Index of the shard to load
        n_shards: Total number of shards

    Returns:
        Dataset: The loaded dataset shard

    Raises:
        AssertionError: If the dataset is not found
    """
    print(f"get_dataset: {name}")
    cfg = client.get_dataset_cfg(name)
    print("get_dataset ok")
    print(f"dataset_cfg: {cfg}")
    assert cfg is not None, f"Dataset {name} not found"
    return load_dataset_shard(cfg, shard_idx, n_shards)


@lru_cache(maxsize=8)
def get_sae(name: str) -> SparseAutoEncoder:
    """Load and cache a sparse autoencoder.

    Args:
        name: Name of the SAE to load

    Returns:
        SparseAutoEncoder: The loaded SAE

    Raises:
        AssertionError: If the SAE is not found
    """
    path = client.get_sae_path(name, sae_series)
    assert path is not None, f"SAE {name} not found"
    cfg = SAEConfig.from_pretrained(path)
    sae = SparseAutoEncoder.from_config(cfg)
    sae.eval()
    return sae


def make_serializable(obj):
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def trim_minimum(
    origins: list[dict[str, Any] | None],
    feature_acts_indices: np.ndarray,
    feature_acts_values: np.ndarray,
) -> tuple[list[dict[str, Any] | None], np.ndarray, np.ndarray]:
    """Trim multiple arrays to the length of the shortest non-None array.

    Args:
        origins: Origins
        feature_acts_indices: Feature acts indices
        feature_acts_values: Feature acts values

    Returns:
        list: List of trimmed arrays
    """

    min_length = min(len(origins), feature_acts_indices[-1] + 10)
    feature_acts_indices_mask = feature_acts_indices <= min_length
    return (
        origins[: int(min_length)],
        feature_acts_indices[feature_acts_indices_mask],
        feature_acts_values[feature_acts_indices_mask],
    )


@app.exception_handler(AssertionError)
async def assertion_error_handler(request, exc):
    return Response(content=str(exc), status_code=400)


@app.exception_handler(torch.cuda.OutOfMemoryError)
async def oom_error_handler(request, exc):
    print("CUDA Out of memory. Clearing cache.")
    # Clear LRU caches
    get_model.cache_clear()
    get_dataset.cache_clear()
    get_sae.cache_clear()
    # Clear GradCAM model cache
    _clear_gradcam_cache()
    # Force CUDA cache cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return Response(content="CUDA Out of memory", status_code=500)


@app.get("/dictionaries")
def list_dictionaries():
    return client.list_saes(sae_series=sae_series, has_analyses=True)


@app.get("/images/{dataset_name}")
def get_image(dataset_name: str, context_idx: int, image_idx: int, shard_idx: int = 0, n_shards: int = 1):
    assert transforms is not None, "torchvision not found, image processing will be disabled"
    dataset = get_dataset(dataset_name, shard_idx, n_shards)
    data = dataset[int(context_idx)]

    image_key = next((key for key in ["image", "images"] if key in data), None)
    if image_key is None:
        return Response(content="Image not found", status_code=404)

    if len(data[image_key]) <= image_idx:
        return Response(content="Image not found", status_code=404)

    image_tensor = data[image_key][image_idx]

    # Convert tensor to PIL Image and then to bytes
    image = transforms.ToPILImage()(image_tensor.to(torch.uint8))
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")

    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

@app.get("/images_single/{dataset_name}")
def get_image_single(dataset_name: str, context_idx: int,  shard_idx: int = 0, n_shards: int = 1):
    assert transforms is not None, "torchvision not found, image processing will be disabled"
    dataset = get_dataset(dataset_name, shard_idx, n_shards)
    data = dataset[int(context_idx)]

    image = data['image']
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")

    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

@app.get("/dictionaries/{name}/metrics")
def get_available_metrics(name: str):
    """Get available metrics for a dictionary.

    Args:
        name: Name of the dictionary/SAE

    Returns:
        List of available metric names
    """
    metrics = client.get_available_metrics(name, sae_series=sae_series)
    return {"metrics": metrics}


@app.get("/dictionaries/{name}/features/count")
def count_features_with_filters(
    name: str,
    feature_analysis_name: str | None = None,
    metric_filters: str | None = None,
    act_times_min: float | None = None,
    act_times_max: float | None = None,
):
    """Count features that match the given filters.

    Args:
        name: Name of the dictionary/SAE
        feature_analysis_name: Optional analysis name
        metric_filters: Optional JSON string of metric filters

    Returns:
        Count of features matching the filters
    """
    # Parse metric filters if provided
    parsed_metric_filters = None
    if metric_filters:
        try:
            parsed_metric_filters = json.loads(metric_filters)
        except (json.JSONDecodeError, TypeError):
            return Response(
                content=f"Invalid metric_filters format: {metric_filters}",
                status_code=400,
            )

    count = client.count_features_with_filters(
        sae_name=name,
        sae_series=sae_series,
        name=feature_analysis_name,
        metric_filters=parsed_metric_filters,
        act_times_min=act_times_min,
        act_times_max=act_times_max,
    )

    return {"count": count}


@app.get("/dictionaries/{name}/features/{feature_index}")
def get_feature(
    name: str,
    feature_index: str | int,
    feature_analysis_name: str | None = None,
    metric_filters: str | None = None,
    act_times_min: float | None = None,
    act_times_max: float | None = None,
):
    # Parse feature_index if it's a string
    if isinstance(feature_index, str) and feature_index != "random":
        try:
            feature_index = int(feature_index)
        except ValueError:
            return Response(
                content=f"Feature index {feature_index} is not a valid integer",
                status_code=400,
            )

    # Parse metric filters if provided
    parsed_metric_filters = None
    if metric_filters:
        try:
            parsed_metric_filters = json.loads(metric_filters)
        except (json.JSONDecodeError, TypeError):
            return Response(
                content=f"Invalid metric_filters format: {metric_filters}",
                status_code=400,
            )

    # Get feature data
    feature = (
        client.get_random_alive_feature(
            sae_name=name,
            sae_series=sae_series,
            name=feature_analysis_name,
            metric_filters=parsed_metric_filters,
            act_times_min=act_times_min,
            act_times_max=act_times_max,
        )
        if feature_index == "random"
        else client.get_feature(sae_name=name, sae_series=sae_series, index=feature_index)
    )

    if feature is None:
        return Response(
            content=f"Feature {feature_index} not found in SAE {name}",
            status_code=404,
        )

    analysis = next(
        (a for a in feature.analyses if a.name == feature_analysis_name or feature_analysis_name is None),
        None,
    )
    if analysis is None:
        return Response(
            content=f"Feature analysis {feature_analysis_name} not found in SAE {name}"
            if feature_analysis_name is not None
            else f"No feature analysis found in SAE {name}",
            status_code=404,
        )

    def process_sample(
        *,
        sparse_feature_acts,
        context_idx,
        dataset_name,
        model_name,
        shard_idx=None,
        n_shards=None,
        reliance=None,
    ):
        """Process a sample to extract and format feature data.

        Args:
            sparse_feature_acts: Sparse feature activations,
                optional z pattern activations
            decoder_norms: Decoder norms
            context_idx: Context index in the dataset
            dataset_name: Name of the dataset
            model_name: Name of the model
            shard_idx: Index of the dataset shard, defaults to 0
            n_shards: Total number of shards, defaults to 1

        Returns:
            dict: Processed sample data
        """  # Get model and dataset
        # print(f"{sparse_feature_acts=}")
        model = get_model(model_name)
        # model = None
        # print("get_data")
        data = get_dataset(dataset_name, shard_idx, n_shards)[context_idx.item()]
        # print("get_data ok")
        # print(f"{data=}")

        (
            feature_acts_indices,
            feature_acts_values,
            z_pattern_indices,
            z_pattern_values,
        ) = sparse_feature_acts
        # print(f"{type(feature_acts_indices)=}")
        # Process image data if present
        if dataset_name != "imagenet":
            # Get origins for the features
            origins = model.trace({k: [v] for k, v in data.items()})[0]
            image_key = next(
                (key for key in ["image", "images"] if key in data),
                None,
            )
            if image_key is not None:
                image_urls = [
                    f"/images/{dataset_name}?context_idx={context_idx}&"
                    f"shard_idx={shard_idx}&n_shards={n_shards}&"
                    f"image_idx={img_idx}"
                    for img_idx in range(len(data[image_key]))
                ]
                del data[image_key]
                data["images"] = image_urls

            # Trim to matching lengths
            origins, feature_acts_indices, feature_acts_values = trim_minimum(
                origins,
                feature_acts_indices,
                feature_acts_values,
            )
            assert origins is not None and feature_acts_indices is not None and feature_acts_values is not None, (
                "Origins and feature acts must not be None"
            )

            # Process text data if present
            if "text" in data:
                text_ranges = [origin["range"] for origin in origins if origin is not None and origin["key"] == "text"]
                if text_ranges:
                    max_text_origin = max(text_ranges, key=lambda x: x[1])
                    data["text"] = data["text"][: max_text_origin[1]]

            return {
                **data,
                "origins": origins,
                "feature_acts_indices": feature_acts_indices,
                "feature_acts_values": feature_acts_values,
                "z_pattern_indices": z_pattern_indices,
                "z_pattern_values": z_pattern_values,
                **({"reliance": reliance} if reliance is not None else {}),
            }
        else:
            image_urls = [
                f"/images_single/{dataset_name}?context_idx={context_idx}&shard_idx={shard_idx}&n_shards={n_shards}"
            ]
            data['images'] = image_urls
            del data['image']
            # transform = v2.Compose([
            #     v2.ToImage(),
            #     v2.CenterCrop((256, 256)),
            #     v2.ToDtype(torch.long, scale=False)
            # ])
            # data['image'] = transform(data['image'])
            # img = transform(img)
            return {
                **data,
                "origins": [],
                "feature_acts_indices": feature_acts_indices,
                "feature_acts_values": feature_acts_values,
                "z_pattern_indices": z_pattern_indices,
                "z_pattern_values": z_pattern_values,
                **({"reliance": reliance} if reliance is not None else {}),
            }

    def process_sparse_feature_acts(
        feature_acts_indices: np.ndarray,
        feature_acts_values: np.ndarray,
        z_pattern_indices: np.ndarray | None = None,
        z_pattern_values: np.ndarray | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None], Any, None]:
        """Process sparse feature acts.

        Args:
            feature_acts_indices: Feature acts indices
            feature_acts_values: Feature acts values
            z_pattern_indices: Z pattern indices
            z_pattern_values: Z pattern values

        TODO: This is really ugly, we should find a better way to do this.
        """
        _, feature_acts_counts = np.unique(
            feature_acts_indices[0],
            return_counts=True,
        )
        _, z_pattern_counts = (
            np.unique(z_pattern_indices[0], return_counts=True) if z_pattern_indices is not None else (None, None)
        )

        feature_acts_sample_ranges = np.concatenate([[0], np.cumsum(feature_acts_counts)])
        z_pattern_sample_ranges = (
            np.concatenate([[0], np.cumsum(z_pattern_counts)]) if z_pattern_counts is not None else None
        )

        feature_acts_sample_ranges = list(zip(feature_acts_sample_ranges[:-1], feature_acts_sample_ranges[1:]))
        z_pattern_sample_ranges = (
            list(zip(z_pattern_sample_ranges[:-1], z_pattern_sample_ranges[1:]))
            if z_pattern_sample_ranges is not None
            else [(None, None)] * len(feature_acts_sample_ranges)
        )
        # print(f"{z_pattern_sample_ranges=} {feature_acts_counts=}")
        # if z_pattern_sample_ranges[0][0] is not None:
        #     assert len(feature_acts_sample_ranges) == len(z_pattern_sample_ranges), (
        #         "Feature acts and z pattern must have the same number of samples"
        #     )

        for (feature_acts_start, feature_acts_end), (z_pattern_start, z_pattern_end) in zip(
            feature_acts_sample_ranges, z_pattern_sample_ranges
        ):
            feature_acts_indices_i = feature_acts_indices[1, feature_acts_start:feature_acts_end]
            feature_acts_values_i = feature_acts_values[feature_acts_start:feature_acts_end]
            z_pattern_indices_i = (
                z_pattern_indices[1:, z_pattern_start:z_pattern_end] if z_pattern_indices is not None else None
            )
            z_pattern_values_i = (
                z_pattern_values[z_pattern_start:z_pattern_end] if z_pattern_values is not None else None
            )
            yield feature_acts_indices_i, feature_acts_values_i, z_pattern_indices_i, z_pattern_values_i

    # Process all samples for each sampling
    sample_groups = []
    for sampling in analysis.samplings:
        # Using zip to process correlated data instead of indexing
        shard_iter = sampling.shard_idx if sampling.shard_idx is not None else [0] * len(sampling.feature_acts_indices)
        nshard_iter = sampling.n_shards if sampling.n_shards is not None else [1] * len(sampling.feature_acts_indices)

        samples = []
        for sample_i, (sparse_feature_acts, context_idx, dataset_name, model_name, shard_idx, n_shards) in enumerate(
            zip(
                process_sparse_feature_acts(
                    sampling.feature_acts_indices,
                    sampling.feature_acts_values,
                    sampling.z_pattern_indices,
                    sampling.z_pattern_values,
                ),
                sampling.context_idx,
                sampling.dataset_name,
                sampling.model_name,
                shard_iter,
                nshard_iter,
            )
        ):
            reliance = None
            if (
                sampling.sample_reliance_label is not None
                and sampling.sample_reliance_relative_changes is not None
                and sampling.sample_reliance_probabilities is not None
                and sample_i < len(sampling.sample_reliance_label)
            ):
                try:
                    reliance = {
                        "label": sampling.sample_reliance_label[sample_i],
                        "relative_changes": {
                            k: sampling.sample_reliance_relative_changes.get(k, [0.0])[sample_i]
                            for k in ["shape", "texture", "color"]
                        },
                        "probabilities": {
                            k: sampling.sample_reliance_probabilities.get(k, [0.0])[sample_i]
                            for k in ["shape", "texture", "color"]
                        },
                    }
                except Exception:
                    reliance = None

            samples.append(
                process_sample(
                    sparse_feature_acts=sparse_feature_acts,
                    context_idx=context_idx,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    shard_idx=shard_idx,
                    n_shards=n_shards,
                    reliance=reliance,
                )
            )

        sample_groups.append(
            {
                "analysis_name": sampling.name,
                "samples": samples,
            }
        )

    # Prepare response
    response_data = {
        "feature_index": feature.index,
        "analysis_name": analysis.name,
        "interpretation": feature.interpretation,
        "dictionary_name": feature.sae_name,
        "logits": feature.logits,
        "decoder_norms": analysis.decoder_norms,
        "decoder_similarity_matrix": analysis.decoder_similarity_matrices,
        "decoder_inner_product_matrix": analysis.decoder_inner_product_matrices,
        "act_times": analysis.act_times,
        "max_feature_act": analysis.max_feature_acts,
        "n_analyzed_tokens": analysis.n_analyzed_tokens,
        "sample_groups": sample_groups,
        "is_bookmarked": client.is_bookmarked(sae_name=name, sae_series=sae_series, feature_index=feature.index),
        "reliance_protocol": getattr(feature, "reliance_protocol", None),
    }
    # print(f"{response_data=}")
    return Response(
        content=msgpack.packb(make_serializable(response_data)),
        media_type="application/x-msgpack",
    )


@app.get("/dictionaries/{name}/reliance_summary")
def get_reliance_summary(name: str):
    """
    Aggregate feature-level reliance probabilities for a dictionary.

    We sum per-feature probabilities:
      prob_shape = p_shape / (p_shape+p_texture+p_color)
    and return the sums and simple normalized proportions.
    """
    pipeline = [
        {
            "$match": {
                "sae_name": name,
                "sae_series": sae_series,
                "reliance_protocol.probabilities": {"$exists": True},
            }
        },
        {
            "$group": {
                "_id": None,
                "n_features": {"$sum": 1},
                "shape": {"$sum": {"$ifNull": ["$reliance_protocol.probabilities.shape", 0.0]}},
                "texture": {"$sum": {"$ifNull": ["$reliance_protocol.probabilities.texture", 0.0]}},
                "color": {"$sum": {"$ifNull": ["$reliance_protocol.probabilities.color", 0.0]}},
            }
        },
    ]

    res = list(client.feature_collection.aggregate(pipeline, allowDiskUse=True))
    if not res:
        return {"n_features": 0, "prob_sums": {"shape": 0.0, "texture": 0.0, "color": 0.0}, "proportions": None}

    row = res[0]
    n = int(row.get("n_features", 0) or 0)
    sums = {
        "shape": float(row.get("shape", 0.0) or 0.0),
        "texture": float(row.get("texture", 0.0) or 0.0),
        "color": float(row.get("color", 0.0) or 0.0),
    }
    denom = sums["shape"] + sums["texture"] + sums["color"]
    proportions = None
    if denom > 0:
        proportions = {k: v / denom for k, v in sums.items()}

    return {"n_features": n, "prob_sums": sums, "proportions": proportions}


# @app.post("/dictionaries/{name}/cache_features")
# def cache_features(
#     name: str,
#     features: list[dict[str, Any]] = Body(..., embed=True),
#     output_dir: str = Body(...),
# ):
#     """Batch-fetch and persist feature payloads for offline reuse.

#     Args:
#         name: Dictionary/SAE name.
#         features: List of feature specs currently on screen. Each item should contain
#             - feature_id: int
#             - layer: int
#             - is_lorsa: bool
#             - analysis_name: Optional[str] (overrides auto selection)
#         output_dir: Directory on the server filesystem to write files into.

#     Returns:
#         Dict with count and directory path.
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     saved = 0
#     for f in features:
#         feature_id = int(f["feature_id"])  # may raise KeyError which FastAPI will surface
#         layer = int(f["layer"])  # required for formatting analysis name
#         is_lorsa = bool(f.get("is_lorsa", False))
#         analysis_name_override = f.get("analysis_name")

#         # Determine analysis name for this feature
#         formatted_analysis_name: str | None = None
#         if analysis_name_override is not None:
#             formatted_analysis_name = analysis_name_override
#         else:
#             try:
#                 base_name = (
#                     client.get_lorsa_analysis_name(name, sae_series)
#                     if is_lorsa
#                     else client.get_clt_analysis_name(name, sae_series)
#                 )
#             except AttributeError:
#                 base_name = None
#             if base_name is None:
#                 feat = client.get_random_alive_feature(sae_name=name, sae_series=sae_series)
#                 if feat is None:
#                     return Response(content=f"Dictionary {name} not found", status_code=404)
#                 available = [a.name for a in feat.analyses]
#                 preferred = [a for a in available if ("lorsa" in a) == is_lorsa]
#                 base_name = preferred[0] if preferred else available[0]
#             formatted_analysis_name = base_name.replace("{}", str(layer))

#         # Reuse existing single-feature endpoint logic. Align with frontend usage where
#         # the path 'name' is the formatted analysis name used by GET /dictionaries/{name}/features/{id}.
#         res = get_feature(name=formatted_analysis_name, feature_index=feature_id, feature_analysis_name=None)
#         if isinstance(res, Response) and res.status_code != 200:
#             # Skip but continue
#             continue

#         payload = res.body if isinstance(res, Response) else res
#         # Write as msgpack for fidelity and also a JSON alongside for convenience
#         base = os.path.join(output_dir, f"layer{layer}__feature{feature_id}__{formatted_analysis_name}.msgpack")
#         with open(base, "wb") as fbin:
#             fbin.write(payload)
#         try:
#             decoded = msgpack.unpackb(payload, raw=False)
#             json_path = base.replace(".msgpack", ".json")
#             # make_serializable handles tensors/np arrays
#             import json as _json

#             with open(json_path, "w") as fj:
#                 _json.dump(make_serializable(decoded), fj)
#         except Exception:
#             pass
#         saved += 1

#     return {"saved": saved, "output_dir": output_dir}


@app.get("/dictionaries/{name}")
def get_dictionary(name: str):
    # Get feature activation times
    feature_activation_times = client.get_feature_act_times(name, sae_series=sae_series)
    if feature_activation_times is None:
        return Response(content=f"Dictionary {name} not found", status_code=404)

    # Create histogram of log activation times
    log_act_times = np.log10(np.array(list(feature_activation_times.values())))
    feature_activation_times_histogram = go.Histogram(
        x=log_act_times,
        nbinsx=100,
        hovertemplate="Count: %{y}<br>Range: %{x}<extra></extra>",
        marker_color="#636EFA",
        showlegend=False,
    ).to_plotly_json()

    # Get alive feature count
    # If no analysis exists under the default name, backend returns 0 (don't crash the endpoint).
    alive_feature_count = client.get_alive_feature_count(name, sae_series=sae_series)

    # Prepare and return response
    response_data = {
        "dictionary_name": name,
        "feature_activation_times_histogram": [feature_activation_times_histogram],
        "alive_feature_count": alive_feature_count,
    }

    return Response(
        content=msgpack.packb(make_serializable(response_data)),
        media_type="application/x-msgpack",
    )


@app.get("/dictionaries/{name}/analyses")
def get_analyses(name: str):
    """Get all available analyses for a dictionary.

    Args:
        name: Name of the dictionary/SAE

    Returns:
        List of analysis names
    """
    # Get a random feature to check its available analyses
    feature = client.get_random_alive_feature(sae_name=name, sae_series=sae_series)
    if feature is None:
        return Response(content=f"Dictionary {name} not found", status_code=404)

    # Extract unique analysis names from feature
    analyses = list(set(analysis.name for analysis in feature.analyses))
    return analyses


@app.post("/dictionaries/{name}/features/{feature_index}/bookmark")
def add_bookmark(name: str, feature_index: int):
    """Add a bookmark for a feature.

    Args:
        name: Name of the dictionary/SAE
        feature_index: Index of the feature to bookmark

    Returns:
        Success response or error
    """
    try:
        success = client.add_bookmark(sae_name=name, sae_series=sae_series, feature_index=feature_index)
        if success:
            return {"message": "Bookmark added successfully"}
        else:
            return Response(content="Feature is already bookmarked", status_code=409)
    except ValueError as e:
        return Response(content=str(e), status_code=404)


@app.delete("/dictionaries/{name}/features/{feature_index}/bookmark")
def remove_bookmark(name: str, feature_index: int):
    """Remove a bookmark for a feature.

    Args:
        name: Name of the dictionary/SAE
        feature_index: Index of the feature to remove bookmark from

    Returns:
        Success response or error
    """
    success = client.remove_bookmark(sae_name=name, sae_series=sae_series, feature_index=feature_index)
    if success:
        return {"message": "Bookmark removed successfully"}
    else:
        return Response(content="Bookmark not found", status_code=404)


@app.get("/dictionaries/{name}/features/{feature_index}/bookmark")
def check_bookmark(name: str, feature_index: int):
    """Check if a feature is bookmarked.

    Args:
        name: Name of the dictionary/SAE
        feature_index: Index of the feature

    Returns:
        Bookmark status
    """
    is_bookmarked = client.is_bookmarked(sae_name=name, sae_series=sae_series, feature_index=feature_index)
    return {"is_bookmarked": is_bookmarked}


@app.get("/bookmarks")
def list_bookmarks(sae_name: Optional[str] = None, sae_series: Optional[str] = None, limit: int = 100, skip: int = 0):
    """List bookmarks with optional filtering.

    Args:
        sae_name: Optional SAE name filter
        sae_series: Optional SAE series filter
        limit: Maximum number of bookmarks to return
        skip: Number of bookmarks to skip (for pagination)

    Returns:
        List of bookmarks
    """
    bookmarks = client.list_bookmarks(sae_name=sae_name, sae_series=sae_series, limit=limit, skip=skip)

    # Convert to dict for JSON serialization
    bookmark_data = []
    for bookmark in bookmarks:
        bookmark_dict = bookmark.model_dump()
        # Convert datetime to ISO string for JSON
        bookmark_dict["created_at"] = bookmark.created_at.isoformat()
        bookmark_data.append(bookmark_dict)

    return {
        "bookmarks": bookmark_data,
        "total_count": client.get_bookmark_count(sae_name=sae_name, sae_series=sae_series),
    }


@app.put("/dictionaries/{name}/features/{feature_index}/bookmark")
def update_bookmark(name: str, feature_index: int, tags: Optional[list[str]] = None, notes: Optional[str] = None):
    """Update a bookmark with new tags or notes.

    Args:
        name: Name of the dictionary/SAE
        feature_index: Index of the feature
        tags: Optional new tags for the bookmark
        notes: Optional new notes for the bookmark

    Returns:
        Success response or error
    """
    success = client.update_bookmark(
        sae_name=name, sae_series=sae_series, feature_index=feature_index, tags=tags, notes=notes
    )
    if success:
        return {"message": "Bookmark updated successfully"}
    else:
        return Response(content="Bookmark not found", status_code=404)


@app.get("/feature_visualize/configs/diffusion")
def list_diffusion_configs():
    """List available diffusion model configurations."""
    if FEATURE_VIZ_DIFFUSION_DIR is None or not FEATURE_VIZ_DIFFUSION_DIR.exists():
        return {"configs": []}
    
    configs = []
    for config_file in sorted(FEATURE_VIZ_DIFFUSION_DIR.glob("*.yaml")):
        try:
            import yaml
            with open(config_file, 'r') as f:
                cfg = yaml.safe_load(f)
            configs.append({
                "name": cfg.get("diffusion_name", config_file.stem),
                "file": config_file.name,
                "path": cfg.get("path", ""),
            })
        except Exception as e:
            print(f"Error reading {config_file}: {e}")
    
    return {"configs": configs}


@app.get("/feature_visualize/configs/sae")
def list_sae_configs():
    """List available SAE configurations."""
    if FEATURE_VIZ_SAE_DIR is None or not FEATURE_VIZ_SAE_DIR.exists():
        return {"configs": []}
    
    configs = []
    for config_file in sorted(FEATURE_VIZ_SAE_DIR.glob("*.yaml")):
        try:
            import yaml
            with open(config_file, 'r') as f:
                cfg = yaml.safe_load(f)
            sae_name = cfg.get("sae_name", config_file.stem)
            configs.append({
                "name": sae_name,
                "file": config_file.name,
                "sae_name": sae_name,  # Include sae_name for exact matching
            })
        except Exception as e:
            print(f"Error reading {config_file}: {e}")
    
    return {"configs": configs}


@app.post("/feature_visualize/generate")
def feature_visualize_generate(
    payload: dict = Body(
        ...,
        example={
            "supervisions": [
                {"feature_id": 123, "pos": [[0, 0], [12, 18]], "cap_value": 15},
                {"feature_id": 456, "pos": [[3, 4]], "objective": "max"},
            ],
            "seed": 42,
            "image_size": 256,
            "batch_size": 1,
            "diffusion_config": "ddpm_imagenet.yaml",
            "sae_config": "layer20.yaml",
        },
    )
):
    """
    Generate an image via act_point conditioning (CNNSAE feature point supervision).
    """
    print(f"{generate_with_point_supervision=}")
    print(f"{CNNSAEFeatureMaxConfig=}")
    if generate_with_point_supervision is None or CNNSAEFeatureMaxConfig is None:
        return Response(
            content="diffusion feature visualize dependencies not available on server",
            status_code=500,
        )

    supervisions = payload.get("supervisions", None)
    if not isinstance(supervisions, list) or len(supervisions) == 0:
        return Response(content="`supervisions` must be a non-empty list", status_code=400)

    seed = payload.get("seed", None)
    image_size = payload.get("image_size", None)
    batch_size = int(payload.get("batch_size", 1) or 1)

    diffusion_config = payload.get("diffusion_config", None)
    sae_config = payload.get("sae_config", None)
    
    cfg = _build_feature_viz_config(
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        diffusion_config=diffusion_config,
        sae_config=sae_config,
    )
    if cfg is None:
        return Response(
            content="feature visualize config could not be constructed (missing dependencies)",
            status_code=500,
        )

    try:
        sample, _fmap = generate_with_point_supervision(cfg, supervisions, return_feature_maps=False)
        # only return first image
        img_base64 = _tensor_to_base64_image(sample[0])
        return {
            "image_base64": img_base64,
            "width": int(sample.shape[-1]),
            "height": int(sample.shape[-2]),
            "supervisions": supervisions,
        }
    except Exception as e:
        print(f"[feature_visualize_generate] failed: {e}")
        return Response(content=str(e), status_code=500)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Grad-CAM utilities
# --------------------------------------------------------------------------- #
CAM_METHODS = {
    "gradcam": GradCAM,
    "hirescam": HiResCAM,
    "gradcam++": GradCAMPlusPlus,
    "ablationcam": AblationCAM,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "layercam": LayerCAM,
    "scorecam": ScoreCAM,
    "fullgrad": FullGrad,
}

GRADCAM_CONFIG_DIR = ROOT_DIR / "feature_visualize_config" / "GradCAM"

# Cache for SAEFeatureClassifier models to avoid reloading on every request
_gradcam_model_cache: dict[str, SAEFeatureClassifier] = {}


def _get_gradcam_model(
    *,
    sae_name: str,
    sae_path: str,
    dino_ckpt_path: str,
    hook_point: str,
    device_override: str,
    reduce: str = "mean",
    center_frac: float = 1.0,
) -> SAEFeatureClassifier:
    """
    Get or create a cached SAEFeatureClassifier for GradCAM.
    Models are cached by sae_name since the same sae_name uses the same SAE and DINO.
    """
    cache_key = sae_name
    
    if cache_key in _gradcam_model_cache:
        return _gradcam_model_cache[cache_key]
    
    # Create new model
    model = SAEFeatureClassifier(
        sae_path=sae_path,
        dino_ckpt_path=dino_ckpt_path,
        hook_point=hook_point,
        device=torch.device(device_override),
        reduce=reduce,
        center_frac=center_frac,
    ).to(device_override)
    
    # Cache it
    _gradcam_model_cache[cache_key] = model
    print(f"[GradCAM] Cached model for sae_name={sae_name}")
    
    return model


def _clear_gradcam_cache():
    """Clear the GradCAM model cache and free GPU memory."""
    global _gradcam_model_cache
    for key in list(_gradcam_model_cache.keys()):
        del _gradcam_model_cache[key]
    _gradcam_model_cache = {}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[GradCAM] Cache cleared")


def _load_gradcam_config_from_file(sae_name: str) -> Optional[dict[str, Any]]:
    """
    Load Grad-CAM config from a YAML/JSON file located at:
      feature_visualize_config/GradCAM/{sae_name}.yaml|yml|json
    """
    if not GRADCAM_CONFIG_DIR.exists():
        return None
    for ext in ["yaml", "yml", "json"]:
        path = GRADCAM_CONFIG_DIR / f"{sae_name}.{ext}"
        if path.exists():
            try:
                with open(path, "r") as f:
                    if ext in ("yaml", "yml"):
                        return yaml.safe_load(f) or {}
                    return json.load(f)
            except Exception as e:  # pragma: no cover
                print(f"[gradcam_config] failed to load {path}: {e}")
                return None
    return None


def _first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for v in x:
            t = _first_tensor(v)
            if t is not None:
                return t
    if isinstance(x, dict):
        for v in x.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    return None


def _filter_connected_layers(
    *,
    model: torch.nn.Module,
    candidate_layers: list[torch.nn.Module],
    input_tensor: torch.Tensor,
    feature_idx: int,
) -> list[torch.nn.Module]:
    """
    Remove layers that are not connected to the selected feature output to avoid CAM crashes.
    """
    outs = {}
    handles = []
    for i, layer in enumerate(candidate_layers):
        def _mk_hook(key):
            def _hook(_m, _inp, out):
                outs[key] = _first_tensor(out)
            return _hook
        handles.append(layer.register_forward_hook(_mk_hook(i)))

    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        loss = logits[:, feature_idx].sum() if logits.ndim > 1 else logits[feature_idx]

        kept = []
        for i, layer in enumerate(candidate_layers):
            out = outs.get(i, None)
            if out is None or not getattr(out, "requires_grad", False):
                continue
            g = torch.autograd.grad(loss, out, retain_graph=True, allow_unused=True)[0]
            if g is None:
                continue
            kept.append(layer)
        return kept
    finally:
        for h in handles:
            h.remove()


def _pil_from_dataset(
    *,
    dataset_name: str,
    context_idx: int,
    image_idx: int = 0,
    shard_idx: int = 0,
    n_shards: int = 1,
):
    assert transforms is not None, "torchvision not found, image processing will be disabled"
    dataset = get_dataset(dataset_name, shard_idx, n_shards)
    data = dataset[int(context_idx)]

    image_key = next((key for key in ["image", "images"] if key in data), None)
    if image_key is None:
        raise ValueError("Image not found in dataset item")

    if image_key == "images":
        images = data[image_key]
        if image_idx >= len(images):
            raise ValueError("image_idx out of range for images list")
        img = images[image_idx]
    else:
        img = data[image_key]

    if hasattr(img, "save"):  # PIL Image
        return img
    # Assume torch tensor [C,H,W] uint8
    return transforms.ToPILImage()(img.to(torch.uint8))


def _pil_to_rgb_and_tensor(img, img_size: int):
    # PIL compatibility: Image.Resampling exists in newer versions.
    from PIL import Image
    bicubic = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    img_resized = img.convert("RGB").resize((img_size, img_size), resample=bicubic)
    rgb = np.array(img_resized).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0  # [-1, 1]
    return rgb, tensor


def _encode_np_image_to_base64(img_np: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_image_url(image_url: str) -> dict[str, Any]:
    """
    Parse an image URL generated by the backend endpoints (/images or /images_single)
    and return dataset_name/context_idx/image_idx/shard_idx/n_shards.
    """
    parsed = urlparse(image_url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError("image_url must include dataset name")
    endpoint = parts[0]
    dataset_name = parts[1]

    qs = parse_qs(parsed.query)
    def _get_int(name: str, default: int) -> int:
        try:
            return int(qs.get(name, [default])[0])
        except Exception:
            return default

    context_idx = _get_int("context_idx", 0)
    shard_idx = _get_int("shard_idx", 0)
    n_shards = _get_int("n_shards", 1)
    image_idx = _get_int("image_idx", 0)

    # /images_single has only one image per item
    if endpoint == "images_single":
        image_idx = 0

    return {
        "dataset_name": dataset_name,
        "context_idx": context_idx,
        "image_idx": image_idx,
        "shard_idx": shard_idx,
        "n_shards": n_shards,
    }


@app.get("/dictionaries/{name}/gradcam_config")
def get_gradcam_config(name: str):
    cfg = _load_gradcam_config_from_file(name)
    return {"gradcam_config": cfg}


@app.post("/gradcam/clear_cache")
def clear_gradcam_cache():
    """
    Manually clear the GradCAM model cache to free GPU memory.
    Call this endpoint when switching between different SAE models or to free memory.
    """
    _clear_gradcam_cache()
    return {"message": "GradCAM cache cleared successfully"}


@app.post("/gradcam/compute")
def compute_gradcam(payload: dict = Body(...)):
    """
    Compute Grad-CAM for a specific SAE feature on a selected image.

    Expected payload:
    {
        "sae_name": "cnn_sae",
        "image_url": "/images_single/imagenet?context_idx=0&shard_idx=0&n_shards=1",
        "feature_idx": 123,
        "cam_method": "gradcam"
    }
    """
    img_tensor = None
    grayscale_cam = None
    cam_image = None
    
    try:
        sae_name = payload.get("sae_name") or payload.get("saeName")
        image_url = payload.get("image_url") or payload.get("imageUrl")
        feature_idx = payload.get("feature_idx")
        cam_method = payload.get("cam_method", "gradcam")
        if sae_name is None or image_url is None or feature_idx is None:
            return Response(content="sae_name, image_url, and feature_idx are required", status_code=400)
        feature_idx = int(feature_idx)

        gradcam_cfg = _load_gradcam_config_from_file(sae_name)
        if gradcam_cfg is None:
            return Response(content="Grad-CAM config not found for SAE (file-based)", status_code=404)

        # Validate cam method
        available_methods = gradcam_cfg.get("available_methods")
        if available_methods and cam_method not in available_methods:
            return Response(
                content=f"cam_method {cam_method} not allowed. Available: {available_methods}",
                status_code=400,
            )
        if cam_method not in CAM_METHODS:
            return Response(content=f"Unknown cam_method {cam_method}", status_code=400)

        # Prefer file-based SAE path if provided; otherwise fallback to Mongo
        sae_path = gradcam_cfg.get("sae_path") or client.get_sae_path(sae_name, sae_series=sae_series)
        if sae_path is None:
            return Response(content="SAE path not found (config or Mongo)", status_code=404)

        hook_point = gradcam_cfg.get("hook_point")
        dino_ckpt_path = gradcam_cfg.get("dino_ckpt_path")
        if not hook_point or not dino_ckpt_path:
            return Response(content="Grad-CAM config missing hook_point or dino_ckpt_path", status_code=400)

        img_size = int(gradcam_cfg.get("img_size", 256))
        reduce = gradcam_cfg.get("reduce", "mean")
        center_frac = float(gradcam_cfg.get("center_frac", 1.0))
        device_override = gradcam_cfg.get("device") or device

        # Use cached model instead of creating new one each time
        model = _get_gradcam_model(
            sae_name=sae_name,
            sae_path=sae_path,
            dino_ckpt_path=dino_ckpt_path,
            hook_point=hook_point,
            device_override=device_override,
            reduce=reduce,
            center_frac=center_frac,
        )

        try:
            img_info = _parse_image_url(image_url)
            pil_img = _pil_from_dataset(**img_info)
        except Exception as e:
            return Response(content=f"Failed to load image: {e}", status_code=400)

        rgb_np, img_tensor = _pil_to_rgb_and_tensor(pil_img, img_size=img_size)
        img_tensor = img_tensor.to(device_override)
        img_tensor.requires_grad_(True)

        target_layers_names = gradcam_cfg.get("target_layers") or [hook_point]
        target_layers = model.resolve_target_layers(target_layers_names)

        if not gradcam_cfg.get("no_filter_layers", False):
            target_layers = _filter_connected_layers(
                model=model,
                candidate_layers=target_layers,
                input_tensor=img_tensor,
                feature_idx=feature_idx,
            )
            if len(target_layers) == 0:
                return Response(
                    content="No target layers have gradients to the selected feature. Choose earlier layers.",
                    status_code=400,
                )

        cam_cls = CAM_METHODS[cam_method]
        targets = [ClassifierOutputTarget(feature_idx)]

        # Clear any accumulated gradients from previous computations
        model.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(True):
            with cam_cls(model=model, target_layers=target_layers) as cam:
                grayscale_cam = cam(
                    input_tensor=img_tensor,
                    targets=targets,
                    aug_smooth=bool(gradcam_cfg.get("aug_smooth", False)),
                    eigen_smooth=bool(gradcam_cfg.get("eigen_smooth", False)),
                )
                grayscale_cam = grayscale_cam[0]

        cam_image = show_cam_on_image(
            rgb_np,
            grayscale_cam,
            use_rgb=True,
            image_weight=float(gradcam_cfg.get("image_weight", 0.5)),
        )

        # Encode result before cleanup
        result_base64 = _encode_np_image_to_base64(cam_image)
        result = {
            "image_base64": result_base64,
            "width": cam_image.shape[1],
            "height": cam_image.shape[0],
            "cam_method": cam_method,
        }
        
        return result
        
    except Exception as e:
        print(f"[gradcam/compute] failed: {e}")
        return Response(content=str(e), status_code=500)
    finally:
        # Cleanup: delete temporary tensors and clear CUDA cache
        if img_tensor is not None:
            del img_tensor
        if grayscale_cam is not None:
            del grayscale_cam
        if cam_image is not None:
            del cam_image
        # Clear intermediate gradients from the model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
