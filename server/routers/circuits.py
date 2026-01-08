import functools
import itertools
import logging
from datetime import datetime
from typing import Any, Optional

import torch
from fastapi import APIRouter, Response
from pydantic import BaseModel

from lm_saes.backend.language_model import TransformerLensLanguageModel
from lm_saes.circuit.attribution import attribute
from lm_saes.circuit.replacement_model import ReplacementModel
from lm_saes.circuit.utils.create_graph_files import serialize_graph
from lm_saes.circuit.utils.transcoder_set import TranscoderSet, TranscoderSetConfig
from lm_saes.database import CircuitConfig, CircuitInput
from lm_saes.lorsa import LowRankSparseAttention
from lm_saes.sae import SparseAutoEncoder
from server.config import client, sae_series
from server.logic.loaders import get_model, get_sae
from server.logic.samples import list_feature_data
from server.utils.common import make_serializable

logger = logging.getLogger(__name__)

router = APIRouter(tags=["circuits"])


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
    input: CircuitInput
    name: Optional[str] = None
    desired_logit_prob: float = 0.98
    max_feature_nodes: int = 256
    qk_tracing_topk: int = 10
    node_threshold: float = 0.8
    edge_threshold: float = 0.98
    max_n_logits: int = 1


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


@router.post("/circuits")
def create_circuit(sae_set_name: str, request: GenerateCircuitRequest):
    """Generate and save a circuit graph for a given prompt and SAE set."""

    sae_set = client.get_sae_set(name=sae_set_name)
    assert sae_set is not None, f"SAE set {sae_set_name} not found"
    sae_names = sae_set.sae_names
    saes = {sae_name: get_sae(name=sae_name) for sae_name in sae_names}

    model_name = client.get_sae_model_name(sae_names[0], sae_set.sae_series)
    model = get_model(name=model_name)
    assert isinstance(model, TransformerLensLanguageModel) and model.model is not None, (
        "Circuit tracing only supports exact model of TransformerLens backend"
    )

    if request.input.input_type == "plain_text":
        prompt = request.input.text
    elif request.input.input_type == "chat_template":
        prompt = model.tokenizer.apply_chat_template(
            request.input.messages, tokenize=False, add_generation_prompt=False, continue_final_message=True
        )
    else:
        raise ValueError(f"Invalid input type: {request.input.input_type}")

    lorsas = {sae_name: sae for sae_name, sae in saes.items() if isinstance(sae, LowRankSparseAttention)}
    transcoders = {sae_name: sae for sae_name, sae in saes.items() if isinstance(sae, SparseAutoEncoder)}

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
        model.cfg, plt_set, list(lorsas.values()), use_lorsa=len(lorsas) > 0
    )

    if len(lorsas) == 0 and request.qk_tracing_topk > 0:
        return Response(content="QK tracing is only supported with Lorsas", status_code=400)

    graph = attribute(
        prompt=prompt,
        model=replacement_model,
        max_n_logits=request.max_n_logits,
        desired_logit_prob=request.desired_logit_prob,
        batch_size=1,
        max_feature_nodes=request.max_feature_nodes,
        sae_series=sae_series,
        qk_tracing_topk=request.qk_tracing_topk,
        use_lorsa=len(lorsas) > 0,
    )
    graph.cfg.tokenizer_name = model.cfg.model_from_pretrained_path or model.cfg.model_name
    graph_data = serialize_graph(
        graph=graph,
        node_threshold=request.node_threshold,
        edge_threshold=request.edge_threshold,
        clt_names=list(transcoders.keys()),
        lorsa_names=list(lorsas.keys()),
        use_lorsa=len(lorsas) > 0,
    )

    config = CircuitConfig(
        desired_logit_prob=request.desired_logit_prob,
        max_feature_nodes=request.max_feature_nodes,
        qk_tracing_topk=request.qk_tracing_topk,
        node_threshold=request.node_threshold,
        edge_threshold=request.edge_threshold,
        max_n_logits=request.max_n_logits,
    )
    circuit_id = client.create_circuit(
        sae_set_name=sae_set_name,
        sae_series=sae_series,
        prompt=prompt,
        input=request.input,
        config=config,
        graph_data=graph_data,
        name=request.name,
    )

    concretize_graph_data(graph_data)

    return make_serializable(
        {
            "circuit_id": circuit_id,
            "name": request.name,
            "sae_set_name": sae_set_name,
            "prompt": prompt,
            "config": config.model_dump(),
            "graph_data": graph_data,
            "input": request.input.model_dump(),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
    )


@router.get("/circuits")
def list_circuits(limit: int = 100, skip: int = 0):
    """List all circuits for the current SAE series."""
    circuits = client.list_circuits(sae_series=sae_series, limit=limit, skip=skip)

    results = []
    for circuit in circuits:
        circuit["created_at"] = circuit["created_at"].isoformat() + "Z"
        results.append(circuit)

    return results


@router.get("/circuits/{circuit_id}")
def get_circuit(circuit_id: str):
    """Get a circuit by its ID with concretized feature data."""
    circuit = client.get_circuit(circuit_id)
    if circuit is None:
        return Response(content=f"Circuit {circuit_id} not found", status_code=404)

    concretize_graph_data(circuit.graph_data)

    result = {
        "circuit_id": circuit.id,
        "name": circuit.name,
        "sae_set_name": circuit.sae_set_name,
        "prompt": circuit.prompt,
        "config": circuit.config.model_dump(),
        "graph_data": circuit.graph_data,
        "created_at": circuit.created_at.isoformat() + "Z",
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
