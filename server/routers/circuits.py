import functools
import itertools
import logging
import threading
import traceback
from datetime import datetime
from typing import Any, Optional

import torch
from fastapi import APIRouter, BackgroundTasks, Response
from pydantic import BaseModel

from lm_saes.backend.language_model import TransformerLensLanguageModel
from lm_saes.circuit.attribution import attribute
from lm_saes.circuit.graph import Graph
from lm_saes.circuit.replacement_model import ReplacementModel
from lm_saes.circuit.utils.create_graph_files import serialize_graph
from lm_saes.circuit.utils.transcoder_set import TranscoderSet, TranscoderSetConfig
from lm_saes.database import CircuitConfig, CircuitInput, CircuitStatus
from lm_saes.lorsa import LowRankSparseAttention
from lm_saes.sae import SparseAutoEncoder
from server.config import client, sae_series
from server.logic.loaders import get_model, get_sae
from server.logic.samples import list_feature_data
from server.utils.common import make_serializable

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
                sorted(filter(lambda x: x["sae_name"] is not None, graph_data["nodes"]), key=lambda x: x["sae_name"]),
                key=lambda x: x["sae_name"],
            )
        ],
        {},
    )

    for node in graph_data["nodes"]:
        if node["sae_name"] is not None:
            if (node["sae_name"], node["feature"]) not in features:
                return Response(
                    content=f"Feature {node['feature']} not found in SAE {node['sae_name']}", status_code=404
                )
            node["feature"] = features[(node["sae_name"], node["feature"])]


def run_circuit_attribution(
    circuit_id: str,
    sae_set_name: str,
    prompt: str,
    request: GenerateCircuitRequest,
    use_lorsa: bool,
):
    """Background task to run circuit attribution and store the result.

    This function is designed to run in a background thread/task.
    """
    try:
        with _generation_lock:
            # Update status to running
            client.update_circuit_status(circuit_id, CircuitStatus.RUNNING)
            client.update_circuit_progress(circuit_id, 0.0, "Initializing models...")

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

            client.update_circuit_progress(circuit_id, 5.0, "Loading transcoders...")

            model_dtype = model.cfg.dtype
            lorsas = {
                sae_name: sae.to(model_dtype)
                for sae_name, sae in saes.items()
                if isinstance(sae, LowRankSparseAttention)
            }
            transcoders = {
                sae_name: sae.to(model_dtype) for sae_name, sae in saes.items() if isinstance(sae, SparseAutoEncoder)
            }

            client.update_circuit_progress(circuit_id, 10.0, "Building replacement model...")

            plt_set = TranscoderSet(
                TranscoderSetConfig(
                    n_layers=model.model.cfg.n_layers,
                    d_sae=list(transcoders.values())[0].cfg.d_sae,
                    feature_input_hook="ln2.hook_normalized",
                    feature_output_hook="hook_mlp_out",
                ),
                {i: transcoder for i, transcoder in enumerate(transcoders.values())},
            )

            replacement_model = ReplacementModel.from_pretrained(
                model.cfg, plt_set, list(lorsas.values()), use_lorsa=use_lorsa
            )

            client.update_circuit_progress(circuit_id, 15.0, "Computing attribution...")

            progress_checkpoints = [(15.0, 50.0, "Feature influence computation")]
            if use_lorsa:
                progress_checkpoints.append((50.0, 95.0, "Computing attention scores attribution"))

            def progress_callback(current: float, total: float, phase: str):
                phase_start, phase_end = next(
                    (p_start, p_end) for p_start, p_end, p_phase in progress_checkpoints if p_phase == phase
                )
                overall_progress = phase_start + (current / total) * (phase_end - phase_start)
                client.update_circuit_progress(circuit_id, overall_progress, phase)

            # Run attribution
            graph = attribute(
                prompt=prompt,
                model=replacement_model,
                max_n_logits=request.max_n_logits,
                desired_logit_prob=request.desired_logit_prob,
                batch_size=1,
                max_feature_nodes=request.max_feature_nodes,
                sae_series=sae_series,
                qk_tracing_topk=request.qk_tracing_topk,
                use_lorsa=use_lorsa,
                list_of_features=request.list_of_features,
                progress_callback=progress_callback,
            )
            graph.cfg.tokenizer_name = model.cfg.model_from_pretrained_path or model.cfg.model_name

            client.update_circuit_progress(circuit_id, 90.0, "Storing graph data...")

            # Store raw graph to GridFS
            graph_dict = graph.to_dict()
            success = client.store_raw_graph(circuit_id, graph_dict)
            if not success:
                raise RuntimeError("Failed to store raw graph to GridFS")

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
    saes = {sae_name: get_sae(name=sae_name) for sae_name in sae_names}

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

    # Determine CLT and LORSA names
    lorsas = {sae_name: sae for sae_name, sae in saes.items() if isinstance(sae, LowRankSparseAttention)}
    transcoders = {sae_name: sae for sae_name, sae in saes.items() if isinstance(sae, SparseAutoEncoder)}
    clt_names = list(transcoders.keys())
    lorsa_names = list(lorsas.keys())
    use_lorsa = len(lorsas) > 0

    if not use_lorsa and request.qk_tracing_topk > 0:
        return Response(content="QK tracing is only supported with Lorsas", status_code=400)

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
        clt_names=clt_names,
        lorsa_names=lorsa_names,
        use_lorsa=use_lorsa,
    )

    # Start background task
    background_tasks.add_task(
        run_circuit_attribution,
        circuit_id=circuit_id,
        sae_set_name=sae_set_name,
        prompt=prompt,
        request=request,
        use_lorsa=use_lorsa,
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
def get_circuit(circuit_id: str, node_threshold: float = 0.8, edge_threshold: float = 0.98):
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

    # Load raw graph from GridFS
    raw_graph_dict = client.load_raw_graph(circuit_id)
    if raw_graph_dict is None:
        return Response(content=f"Raw graph data not found for circuit {circuit_id}", status_code=404)

    # Reconstruct Graph object
    graph = Graph.from_dict(raw_graph_dict)

    # Serialize with dynamic thresholds
    graph_data = serialize_graph(
        graph=graph,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        clt_names=circuit.clt_names or [],
        lorsa_names=circuit.lorsa_names,
        use_lorsa=circuit.use_lorsa,
    )

    # Concretize feature data
    concretize_graph_data(graph_data)

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


@router.delete("/circuits/{circuit_id}")
def delete_circuit(circuit_id: str):
    """Delete a circuit by its ID."""
    success = client.delete_circuit(circuit_id)
    if success:
        return {"message": "Circuit deleted successfully"}
    else:
        return Response(content=f"Circuit {circuit_id} not found", status_code=404)
