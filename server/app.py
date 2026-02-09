import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from server.logic.loaders import get_dataset, get_model, get_sae
from server.routers import admin, bookmarks, circuits, dictionaries
from server.routers.circuits import load_circuit_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_models = os.environ["PRELOAD_MODELS"].strip().split(",") if os.environ.get("PRELOAD_MODELS") else []
    for model in preload_models:
        get_model(name=model)

    preload_saes = os.environ["PRELOAD_SAES"].strip().split(",") if os.environ.get("PRELOAD_SAES") else []
    for sae in preload_saes:
        get_sae(name=sae)

    # Format: "circuit_id:node_threshold:edge_threshold,..."
    # Thresholds default to 0.6 and 0.8 if omitted.
    preload_circuits = os.environ["PRELOAD_CIRCUITS"].strip().split(",") if os.environ.get("PRELOAD_CIRCUITS") else []
    for entry in preload_circuits:
        parts = entry.strip().split(":")
        circuit_id = parts[0]
        node_threshold = float(parts[1]) if len(parts) > 1 else 0.6
        edge_threshold = float(parts[2]) if len(parts) > 2 else 0.8
        load_circuit_graph(circuit_id=circuit_id, node_threshold=node_threshold, edge_threshold=edge_threshold)

    yield

    get_model.cache_clear()
    get_dataset.cache_clear()
    get_sae.cache_clear()
    load_circuit_graph.cache_clear()


app = FastAPI(lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    return Response(content="CUDA Out of memory", status_code=500)


app.include_router(admin.router)
app.include_router(dictionaries.router)
app.include_router(circuits.router)
app.include_router(bookmarks.router)
