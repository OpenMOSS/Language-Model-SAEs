import torch
from typing import Dict, Any, Optional, List, Tuple
from fastapi import HTTPException

# add project root to Python path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.feature_and_steering.interact import analyze_node_interaction
except ImportError:
    # if import fails, try to import from current directory
    try:
        from feature_and_steering.interact import analyze_node_interaction
    except ImportError:
        analyze_node_interaction = None
        print("WARNING: analyze_node_interaction not found, node interaction functionality will be disabled")

# global constants
try:
    from .constants import BT4_MODEL_NAME
except ImportError:
    from constants import BT4_MODEL_NAME

try:
    from .circuits_service import get_cached_models
except ImportError:
    try:
        from circuits_service import get_cached_models
    except ImportError:
        get_cached_models = None
        print("WARNING: circuits_service not found, cached models unavailable")


def analyze_node_interaction_impl(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    implement the backend logic for node interaction analysis
    
    Args:
        request: request body, containing:
            - model_name: model name (optional, default is BT4)
            - sae_combo_id: SAE combo ID (required)
            - fen: FEN string (required)
            - steering_nodes: steering nodes list or single node (required)
                - single node: dict with {feature_type, layer, feature, pos}
                - list: list of dicts
            - target_nodes: target nodes list or single node (required)
                - single node: dict with {feature_type, layer, feature, pos}
                - list: list of dicts
            - steering_scale: steering scale factor (optional, default is 2.0)
    
    Returns:
        dictionary containing the analysis results:
            - steering_scale: 使用的steering scale
            - steering_nodes_count: number of steering nodes
            - steering_details: details about each steering node
            - target_nodes: list of results for each target node
    """
    if analyze_node_interaction is None:
        raise HTTPException(
            status_code=503,
            detail="Node interaction analysis not available. Please check if src/feature_and_steering/interact.py is available."
        )
    
    if get_cached_models is None:
        raise HTTPException(
            status_code=503,
            detail="Circuits service not available. Please check if circuits_service.py is available."
        )
    
    # extract parameters
    model_name = request.get("model_name", BT4_MODEL_NAME)
    sae_combo_id = request.get("sae_combo_id")
    fen = request.get("fen")
    steering_nodes = request.get("steering_nodes")
    target_nodes = request.get("target_nodes")
    steering_scale = request.get("steering_scale", 2.0)
    
    # validate required parameters
    if not sae_combo_id:
        raise HTTPException(status_code=400, detail="sae_combo_id is required")
    if not fen:
        raise HTTPException(status_code=400, detail="fen is required")
    if steering_nodes is None:
        raise HTTPException(status_code=400, detail="steering_nodes is required")
    if target_nodes is None:
        raise HTTPException(status_code=400, detail="target_nodes is required")
    if not isinstance(steering_scale, (int, float)):
        raise HTTPException(status_code=400, detail="steering_scale must be a number")
    
    # ensure steering_nodes is a list
    if not isinstance(steering_nodes, list):
        steering_nodes = [steering_nodes]
    
    # ensure target_nodes is a list
    if not isinstance(target_nodes, list):
        target_nodes = [target_nodes]
    
    if len(steering_nodes) == 0:
        raise HTTPException(status_code=400, detail="At least one steering node is required")
    if len(target_nodes) == 0:
        raise HTTPException(status_code=400, detail="At least one target node is required")
    
    # build cache_key
    cache_key = f"{model_name}::{sae_combo_id}"
    
    # get cached model, transcoders and lorsas
    try:
        cached_hooked_model, cached_transcoders, cached_lorsas, _ = get_cached_models(cache_key)
        
        if cached_hooked_model is None:
            raise HTTPException(
                status_code=503,
                detail=f"Model not loaded. Please call /circuit/preload_models first with combo_id={sae_combo_id}"
            )
        if cached_transcoders is None or cached_lorsas is None:
            raise HTTPException(
                status_code=503,
                detail=f"Transcoders/Lorsas not loaded. Please call /circuit/preload_models first with combo_id={sae_combo_id}"
            )
        
        # check if transcoders and lorsas are complete (should have 15 layers)
        if len(cached_transcoders) != 15 or len(cached_lorsas) != 15:
            raise HTTPException(
                status_code=503,
                detail=f"Transcoders/Lorsas incomplete. Expected 15 layers, got transcoders={len(cached_transcoders)}, lorsas={len(cached_lorsas)}"
            )
        
        model = cached_hooked_model
        transcoders = cached_transcoders
        lorsas = cached_lorsas
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to get cached models: {str(e)}. Please call /circuit/preload_models first."
        )
    
    # run model and get cache
    try:
        _, cache = model.run_with_cache(fen, prepend_bos=False)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run model and get cache: {str(e)}"
        )
    
    # call analyze_node_interaction function
    try:
        result = analyze_node_interaction(
            steering_nodes=steering_nodes,
            target_nodes=target_nodes,
            steering_scale=steering_scale,
            cache=cache,
            model=model,
            transcoders=transcoders,
            lorsas=lorsas,
        )
        return result
    except ValueError as e:
        # layer validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze node interaction: {str(e)}"
        )
    finally:
        # clean up hooks
        try:
            model.reset_hooks()
        except Exception:
            pass
