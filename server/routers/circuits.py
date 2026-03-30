import functools
import itertools
import logging
import re
import threading
import traceback
from datetime import datetime
from typing import Any, Optional

import torch
from fastapi import APIRouter, BackgroundTasks, Response
from pydantic import BaseModel

from lm_saes.backend.language_model import TransformerLensLanguageModel, prune_attribution
from lm_saes.database import CircuitConfig, CircuitInput, CircuitStatus
from lm_saes.models.lorsa import LowRankSparseAttention
from server.config import LRU_CACHE_SIZE_CIRCUITS, client, sae_series
from server.logic.loaders import get_model, get_sae
from server.logic.samples import list_feature_data
from server.utils.common import make_serializable, synchronized

logger = logging.getLogger(__name__)

router = APIRouter(tags=["circuits"])

# Lock for thread-safe circuit generation
_generation_lock = threading.Lock()


class PreviewRequest(BaseModel):
    input: CircuitInput
    sae_set_name: str


@router.post("/preview")
def preview(request: PreviewRequest):
    """Preview the prompt and predicted next tokens."""
    sae_set = client.get_sae_set(name=request.sae_set_name)
    assert sae_set is not None, f"SAE set {request.sae_set_name} not found"
    sae_names = sae_set.sae_names
    model_name = client.get_sae_model_name(sae_names[0], sae_set.sae_series)
    model = get_model(name=model_name)
    assert isinstance(model, TransformerLensLanguageModel) and model.model is not None, (
        f"Preview only supports TransformerLens backend, got {type(model)}"
    )

    if request.input.input_type == "plain_text":
        prompt = request.input.text
    elif request.input.input_type == "chat_template":
        prompt = model.tokenizer.apply_chat_template(
            request.input.messages, tokenize=False, add_generation_prompt=False, continue_final_message=True
        )
    else:
        raise ValueError(f"Invalid input type: {request.input.input_type}")

    # Generate next predicted tokens
    with torch.no_grad():
        logits = model(prompt)[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, 5)

        next_tokens = []
        for i, prob in zip(topk.indices.tolist(), topk.values.tolist()):
            token_str = model.tokenizer.decode([i])
            next_tokens.append({"token": token_str, "prob": prob})

    return {"prompt": prompt, "next_tokens": next_tokens}


class CreateSaeSetRequest(BaseModel):
    name: str
    sae_names: list[str]


@router.get("/sae-sets")
def list_sae_sets():
    """List all available SAE sets for the current series."""
    return [sae_set.name for sae_set in client.list_sae_sets() if sae_set.sae_series == sae_series]


@router.post("/sae-sets")
def create_sae_set(request: CreateSaeSetRequest):
    """Create a new SAE set in the current series."""
    try:
        client.add_sae_set(
            name=request.name,
            sae_series=sae_series,
            sae_names=request.sae_names,
        )
        return {"message": f"SAE set '{request.name}' created successfully"}
    except Exception as e:
        if "duplicate key" in str(e).lower():
            return Response(
                content=f"SAE set with name '{request.name}' already exists",
                status_code=409,
            )
        raise


class GenerateCircuitRequest(BaseModel):
    """Request to generate a new circuit graph.

    Note: node_threshold and edge_threshold are no longer part of generation.
    They are now query-time parameters for dynamic pruning.

    list_of_features: list of (layer, feature_idx, pos, is_lorsa) tuples
    """

    input: CircuitInput
    name: Optional[str] = None
    group: Optional[str] = None
    desired_logit_prob: float = 0.98
    max_feature_nodes: int = 256
    qk_tracing_topk: int = 10
    max_n_logits: int = 1
    list_of_features: Optional[list[tuple[int, int, int, bool]]] = None
    parent_id: Optional[str] = None


def concretize_graph_data(graph_data: dict[str, Any]):
    """Concretize a graph data by adding feature data. This will modify the graph data in place."""
    logger.info("Retrieving feature records for circuit")

    features = functools.reduce(
        lambda acc, x: acc | x,
        [
            list_feature_data(sae_name=sae_name, indices=[node["feature"] for node in nodes], with_samplings=False)
            for sae_name, nodes in itertools.groupby(
                sorted(
                    filter(lambda x: x.get("sae_name") is not None, graph_data["nodes"]), key=lambda x: x["sae_name"]
                ),
                key=lambda x: x["sae_name"],
            )
        ],
        {},
    )

    for node in graph_data["nodes"]:
        if node.get("sae_name") is not None:
            if (node["sae_name"], node["feature"]) not in features:
                return Response(
                    content=f"Feature {node['feature']} not found in SAE {node['sae_name']}", status_code=404
                )
            node["feature"] = features[(node["sae_name"], node["feature"])]


@synchronized
@functools.lru_cache(maxsize=LRU_CACHE_SIZE_CIRCUITS)
def load_circuit_graph(*, circuit_id: str, node_threshold: float, edge_threshold: float) -> dict[str, Any]:
    """Load, prune, and concretize a circuit graph. Cached for repeated access."""
    circuit = client.get_circuit(circuit_id)
    if circuit is None:
        raise ValueError(f"Circuit {circuit_id} not found")
    if circuit.status != CircuitStatus.COMPLETED:
        raise ValueError(f"Circuit {circuit_id} is not completed (status: {circuit.status})")

    ar = client.load_attribution(circuit_id)
    if ar is None:
        raise ValueError(f"Attribution data not found for circuit {circuit_id}")

    attribution = prune_attribution(
        ar.attribution,
        ar.probs,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )

    sae_set = client.get_sae_set(name=circuit.sae_set_name)
    if sae_set is None:
        raise ValueError(f"SAE set {circuit.sae_set_name} not found")

    sae_metadata: dict[str, dict[str, Any]] = {}
    for sae_name in sae_set.sae_names:
        sae = get_sae(name=sae_name)
        hook_point_out = getattr(sae.cfg, "hook_point_out", None)
        if hook_point_out is None:
            continue
        layer_match = re.search(r"blocks\.(\d+)\.", hook_point_out)
        layer_idx = int(layer_match.group(1)) if layer_match else 0
        sae_metadata[hook_point_out] = {
            "sae_name": sae_name,
            "is_lorsa": isinstance(sae, LowRankSparseAttention),
            "layer_idx": layer_idx,
        }

    def node_signature(node_key: Any, indices: torch.Tensor) -> tuple[str, tuple[int, ...]]:
        return (str(node_key), tuple(indices[0].tolist()))

    def parse_hook_layer(hook_key: str) -> int:
        match = re.search(r"blocks\.(\d+)\.", hook_key)
        return int(match.group(1)) if match else 0

    edge_weights, (targets, sources) = attribution.nonzero()
    nodes = (sources + targets).unique()

    activation_map: dict[tuple[str, tuple[int, ...]], float] = {}
    for node_info in nodes.node_infos:
        for single_node_info in node_info.unbind():
            sig = node_signature(single_node_info.key, single_node_info.indices)
            activation_value = ar.activations[[single_node_info]].data
            activation_map[sig] = float(activation_value.item())

    logit_token_map = {token_id: token for token_id, token in zip(ar.logit_token_ids, ar.logit_tokens)}
    logit_prob_map = {token_id: float(prob.item()) for token_id, prob in zip(ar.logit_token_ids, ar.probs)}
    target_vocab_index = int(targets.node_mappings["logits"].indices[0][0].item())
    n_layers = max((int(item["layer_idx"]) for item in sae_metadata.values()), default=-1) + 1

    def make_node(node_key: str, indices_tuple: tuple[int, ...]) -> dict[str, Any]:
        indices = list(indices_tuple)
        sig = (node_key, indices_tuple)

        if node_key == "hook_embed":
            pos = indices[0]
            token_id = int(ar.prompt_token_ids[pos]) if pos < len(ar.prompt_token_ids) else -1
            token = ar.prompt_tokens[pos] if pos < len(ar.prompt_tokens) else ""
            return {
                "feature_type": "embedding",
                "node_id": f"E_{token_id}_{pos}",
                "layer": -1,
                "ctx_idx": pos,
                "token": token,
                "is_target_logit": False,
                "is_from_qk_tracing": False,
            }

        if node_key == "logits":
            vocab_idx = indices[0]
            ctx_idx = max(len(ar.prompt_token_ids) - 1, 0)
            layer = 2 * n_layers
            return {
                "feature_type": "logit",
                "node_id": f"{layer}_{vocab_idx}_{ctx_idx}",
                "layer": layer,
                "ctx_idx": ctx_idx,
                "token_prob": logit_prob_map[vocab_idx],
                "token": logit_token_map[vocab_idx],
                "is_target_logit": vocab_idx == target_vocab_index,
                "is_from_qk_tracing": False,
            }

        if node_key.endswith(".error"):
            hook_point_out = node_key.removesuffix(".error")
            metadata = sae_metadata.get(hook_point_out, None)
            is_lorsa = bool(metadata["is_lorsa"]) if metadata is not None else False
            layer_base = int(metadata["layer_idx"]) if metadata is not None else parse_hook_layer(hook_point_out)
            layer = 2 * layer_base + int(not is_lorsa)
            pos = indices[0]
            return {
                "feature_type": "lorsa error" if is_lorsa else "mlp reconstruction error",
                "node_id": f"{layer}_error_{pos}",
                "layer": layer,
                "ctx_idx": pos,
                "is_target_logit": False,
                "is_from_qk_tracing": False,
            }

        if node_key.endswith(".sae.hook_feature_acts"):
            hook_point_out = node_key.removesuffix(".sae.hook_feature_acts")
            metadata = sae_metadata.get(hook_point_out, None)
            is_lorsa = bool(metadata["is_lorsa"]) if metadata is not None else False
            layer_base = int(metadata["layer_idx"]) if metadata is not None else parse_hook_layer(hook_point_out)
            layer = 2 * layer_base + int(not is_lorsa)
            pos = indices[0] if len(indices) > 0 else 0
            feature_idx = indices[1] if len(indices) > 1 else 0
            activation_value = activation_map.get(sig, 0.0)
            return {
                "feature_type": "lorsa" if is_lorsa else "cross layer transcoder",
                "node_id": f"{layer}_{feature_idx}_{pos}",
                "layer": layer,
                "ctx_idx": pos,
                "feature": feature_idx,
                "sae_name": metadata["sae_name"] if metadata is not None else None,
                "activation": activation_value,
                "qk_tracing_results": None,
                "is_target_logit": False,
                "is_from_qk_tracing": False,
            }

        fallback_id = f"{node_key}:{'_'.join(str(v) for v in indices)}"
        return {
            "feature_type": "bias",
            "node_id": fallback_id,
            "layer": 0,
            "ctx_idx": indices[0] if len(indices) > 0 else 0,
            "is_target_logit": False,
            "is_from_qk_tracing": False,
        }

    node_lookup = {
        sig: make_node(*sig)
        for node_info in nodes.node_infos
        for single_node_info in node_info.unbind()
        for sig in (node_signature(single_node_info.key, single_node_info.indices),)
    }

    links = [
        {
            "source": node_lookup[node_signature(source_ni.key, source_ni.indices)]["node_id"],
            "target": node_lookup[node_signature(target_ni.key, target_ni.indices)]["node_id"],
            "weight": float(weight),
        }
        for weight, target_ni, source_ni in zip(edge_weights.tolist(), targets, sources)
    ]

    graph_data: dict[str, Any] = {
        "metadata": {
            "prompt_tokens": ar.prompt_tokens,
            "prompt": circuit.prompt,
            "schema_version": 1,
        },
        "nodes": list(node_lookup.values()),
        "links": links,
    }
    concretize_graph_data(graph_data)
    return graph_data


def run_circuit_attribution(
    circuit_id: str,
    sae_set_name: str,
    prompt: str,
    request: GenerateCircuitRequest,
):
    """Background task to run circuit attribution and store the result.

    This function is designed to run in a background thread/task.
    """
    try:
        with _generation_lock:
            # Update status to running
            client.update_circuit_status(circuit_id, CircuitStatus.RUNNING)
            client.update_circuit_progress(circuit_id, 0.0, "Loading sparse dictionaries...")

            # Load SAE set and models
            sae_set = client.get_sae_set(name=sae_set_name)
            assert sae_set is not None, f"SAE set {sae_set_name} not found"
            sae_names = sae_set.sae_names
            saes = {sae_name: get_sae(name=sae_name) for sae_name in sae_names}

            model_name = client.get_sae_model_name(sae_names[0], sae_set.sae_series)
            model = get_model(name=model_name)
            assert isinstance(model, TransformerLensLanguageModel) and model.model is not None, (
                "Circuit tracing only supports exact model of TransformerLens backend"
            )

            client.update_circuit_progress(circuit_id, 15.0, "Computing attribution...")

            progress_checkpoints = [(15.0, 95.0, "Feature influence computation")]

            def progress_callback(current: float, total: float, phase: str):
                phase_start, phase_end = next(
                    (p_start, p_end) for p_start, p_end, p_phase in progress_checkpoints if p_phase == phase
                )
                overall_progress = phase_start + (current / total) * (phase_end - phase_start)
                client.update_circuit_progress(circuit_id, overall_progress, phase)

            attribution = model.attribute(
                inputs=prompt,
                replacement_modules=list(saes.values()),
                max_n_logits=request.max_n_logits,
                desired_logit_prob=request.desired_logit_prob,
                batch_size=128,
                max_features=request.max_feature_nodes,
            )

            client.update_circuit_progress(circuit_id, 90.0, "Storing attribution data...")
            client.store_attribution(circuit_id, attribution)

            client.update_circuit_progress(circuit_id, 100.0, "Completed")
            client.update_circuit_status(circuit_id, CircuitStatus.COMPLETED)

            logger.info(f"Circuit {circuit_id} attribution completed successfully")

    except Exception as e:
        error_msg = f"Attribution failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"Circuit {circuit_id} attribution failed: {error_msg}")
        client.update_circuit_status(circuit_id, CircuitStatus.FAILED, error_message=str(e))


@router.post("/circuits")
def create_circuit(sae_set_name: str, request: GenerateCircuitRequest, background_tasks: BackgroundTasks):
    """Start circuit graph generation as a background task.

    Returns immediately with the circuit_id and status='pending'.
    Use GET /circuits/{id}/status to check progress.
    """
    sae_set = client.get_sae_set(name=sae_set_name)
    if sae_set is None:
        return Response(content=f"SAE set {sae_set_name} not found", status_code=404)

    sae_names = sae_set.sae_names

    model_name = client.get_sae_model_name(sae_names[0], sae_set.sae_series)
    model = get_model(name=model_name)
    if not isinstance(model, TransformerLensLanguageModel) or model.model is None:
        return Response(
            content="Circuit tracing only supports TransformerLens backend",
            status_code=400,
        )

    # Determine prompt
    if request.input.input_type == "plain_text":
        prompt = request.input.text
    elif request.input.input_type == "chat_template":
        prompt = model.tokenizer.apply_chat_template(
            request.input.messages, tokenize=False, add_generation_prompt=False, continue_final_message=True
        )
    else:
        return Response(content=f"Invalid input type: {request.input.input_type}", status_code=400)

    # Create circuit config (without threshold params)
    config = CircuitConfig(
        desired_logit_prob=request.desired_logit_prob,
        max_feature_nodes=request.max_feature_nodes,
        qk_tracing_topk=request.qk_tracing_topk,
        max_n_logits=request.max_n_logits,
        list_of_features=request.list_of_features,
    )

    # Create circuit record with pending status
    circuit_id = client.create_circuit(
        sae_set_name=sae_set_name,
        sae_series=sae_series,
        prompt=prompt,
        input=request.input,
        config=config,
        name=request.name,
        group=request.group,
        parent_id=request.parent_id,
    )

    # Start background task
    background_tasks.add_task(
        run_circuit_attribution,
        circuit_id=circuit_id,
        sae_set_name=sae_set_name,
        prompt=prompt,
        request=request,
    )

    return {
        "circuit_id": circuit_id,
        "status": CircuitStatus.PENDING,
        "name": request.name,
        "group": request.group,
        "sae_set_name": sae_set_name,
        "prompt": prompt,
        "config": config.model_dump(),
        "input": request.input.model_dump(),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "parent_id": request.parent_id,
    }


@router.get("/circuits")
def list_circuits(limit: int = 100, skip: int = 0, group: Optional[str] = None):
    """List all circuits for the current SAE series."""
    circuits = client.list_circuits(sae_series=sae_series, limit=limit, skip=skip, group=group)

    results = []
    for circuit in circuits:
        circuit["created_at"] = circuit["created_at"].isoformat() + "Z"
        results.append(circuit)

    return results


@router.get("/circuits/{circuit_id}/status")
def get_circuit_status(circuit_id: str):
    """Get the current status of a circuit generation."""
    status = client.get_circuit_status(circuit_id)
    if status is None:
        return Response(content=f"Circuit {circuit_id} not found", status_code=404)

    return status


@router.get("/circuits/{circuit_id}")
def get_circuit(circuit_id: str, node_threshold: float = 0.6, edge_threshold: float = 0.8):
    """Get a circuit by its ID with dynamic pruning.

    The raw graph is pruned on-the-fly using the provided threshold parameters.

    Args:
        circuit_id: The circuit ID.
        node_threshold: Keep nodes contributing to this fraction of total influence (0-1).
        edge_threshold: Keep edges contributing to this fraction of total influence (0-1).
    """
    circuit = client.get_circuit(circuit_id)
    if circuit is None:
        return Response(content=f"Circuit {circuit_id} not found", status_code=404)

    # Check if circuit is completed
    if circuit.status != CircuitStatus.COMPLETED:
        return Response(
            content=f"Circuit {circuit_id} is not ready (status: {circuit.status})",
            status_code=202,
            headers={"X-Circuit-Status": circuit.status},
        )

    try:
        graph_data = load_circuit_graph(
            circuit_id=circuit_id, node_threshold=node_threshold, edge_threshold=edge_threshold
        )
    except ValueError as e:
        return Response(content=str(e), status_code=404)

    result = {
        "circuit_id": circuit.id,
        "name": circuit.name,
        "group": circuit.group,
        "input": circuit.input.model_dump(),
        "sae_set_name": circuit.sae_set_name,
        "prompt": circuit.prompt,
        "config": circuit.config.model_dump(),
        "graph_data": graph_data,
        "created_at": circuit.created_at.isoformat() + "Z",
        "parent_id": circuit.parent_id,
        "status": circuit.status,
        "node_threshold": node_threshold,
        "edge_threshold": edge_threshold,
    }

    return make_serializable(result)


@router.get("/circuits/{circuit_id}/qk/{node_id}")
def get_circuit_qk_node(circuit_id: str, node_id: str, node_threshold: float = 0.6, edge_threshold: float = 0.8):
    """Get QK tracing data for a specific node, returning only the target node and referenced nodes."""
    circuit = client.get_circuit(circuit_id)
    if circuit is None:
        return Response(content=f"Circuit {circuit_id} not found", status_code=404)

    if circuit.status != CircuitStatus.COMPLETED:
        return Response(
            content=f"Circuit {circuit_id} is not ready (status: {circuit.status})",
            status_code=202,
            headers={"X-Circuit-Status": circuit.status},
        )

    try:
        graph_data = load_circuit_graph(
            circuit_id=circuit_id, node_threshold=node_threshold, edge_threshold=edge_threshold
        )
    except ValueError as e:
        return Response(content=str(e), status_code=404)

    nodes_by_id = {node["node_id"]: node for node in graph_data["nodes"]}

    target_node = nodes_by_id.get(node_id)
    if target_node is None:
        return Response(content=f"Node {node_id} not found in circuit {circuit_id}", status_code=404)

    qk_results = target_node.get("qk_tracing_results")
    if qk_results is None:
        return Response(content=f"Node {node_id} has no QK tracing results", status_code=404)

    # Collect all node IDs referenced in QK tracing results
    referenced_ids: set[str] = set()
    for ref_id, _ in qk_results.get("top_q_marginal_contributors", []):
        referenced_ids.add(ref_id)
    for ref_id, _ in qk_results.get("top_k_marginal_contributors", []):
        referenced_ids.add(ref_id)
    for q_id, k_id, _ in qk_results.get("pair_wise_contributors", []):
        referenced_ids.add(q_id)
        referenced_ids.add(k_id)
    referenced_ids.discard(node_id)

    referenced_nodes = [nodes_by_id[nid] for nid in referenced_ids if nid in nodes_by_id]

    return make_serializable(
        {
            "target_node": target_node,
            "referenced_nodes": referenced_nodes,
        }
    )


@router.delete("/circuits/{circuit_id}")
def delete_circuit(circuit_id: str):
    """Delete a circuit by its ID."""
    success = client.delete_circuit(circuit_id)
    if success:
        return {"message": "Circuit deleted successfully"}
    else:
        return Response(content=f"Circuit {circuit_id} not found", status_code=404)
