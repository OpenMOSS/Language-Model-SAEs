# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Language-Model-SAEs is a research codebase for training and analyzing Sparse Autoencoders (SAEs) on language models, developed by the OpenMOSS Mechanistic Interpretability Team. The project focuses on extracting interpretable features from language models using dictionary learning techniques and enables circuit-level analysis of transformer architectures.

## Core Architecture

### Main Package Structure (`src/lm_saes/`)

- **`sae.py`**: Standard Sparse Autoencoder implementation with support for GLU encoders
- **`clt.py`**: Cross-Layer Transcoder (CLT) - advanced architecture for multi-layer feature attribution
- **`crosscoder.py`**: CrossCoder implementation for cross-model analysis  
- **`activation/`**: Activation data processing, caching, and distributed processing pipelines
- **`analysis/`**: Feature analysis tools including DirectLogitAttributor and automated feature interpretation
- **`backend/`**: Language model backends (primarily HuggingFace integration)
- **`runners/`**: API entry points for training, analysis, generation, and evaluation workflows
- **`utils/`**: Utilities for distributed computing, logging, math operations, and tensor manipulation

### Key Research Components

- **Cross-Layer Transcoder (CLT)**: Implements circuit tracing methodology with L encoders and L(L+1)/2 decoders for linear attribution between features across layers
- **Feature Analysis Pipeline**: Automated feature interpretation using language models, logit attribution analysis
- **Distributed Training**: Full support for multi-GPU/multi-node training with tensor parallelism
- **Visualization System**: React/TypeScript UI with FastAPI backend for exploring learned features

## Development Commands

### Environment Setup

```bash
# Install dependencies using uv (replaces pdm)
uv sync

# Install frontend dependencies
cd ui && bun install
```

### Running Experiments

Experiments are typically launched using Python scripts that import the training functions directly. See examples in `exp/` and `examples/` directories:

```python
# Standard SAE Training
from lm_saes import (
    SAEConfig,
    TrainSAESettings, 
    TrainerConfig,
    InitializerConfig,
    WandbConfig,
    ActivationFactoryConfig,
    train_sae
)

settings = TrainSAESettings(
    sae=SAEConfig(
        hook_point_in="blocks.4.hook_resid_post",
        hook_point_out="blocks.4.hook_resid_post", 
        d_model=768,
        expansion_factor=8,
        act_fn="jumprelu",
        dtype=torch.float32,
        device="cuda",
    ),
    trainer=TrainerConfig(
        lr=4e-4,
        l1_coefficient=0.02,
        total_training_tokens=600_000_000,
    ),
    # ... other configurations
)
train_sae(settings)
```

```python
# CrossCoder Training
from lm_saes import (
    CrossCoderConfig,
    TrainCrossCoderSettings,
    train_crosscoder
)

settings = TrainCrossCoderSettings(
    sae=CrossCoderConfig(
        hook_points=[f"step{step}" for step in steps],
        d_model=768,
        expansion_factor=8,
        act_fn="jumprelu",
    ),
    # ... other configurations
)
train_crosscoder(settings)
```

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories  
pytest tests/unit/
pytest tests/integration/

# Run TransformerLens tests (from TransformerLens directory)
cd TransformerLens && make test
```

### Code Quality

```bash
# Format and lint with ruff
ruff format .
ruff check .

# Type checking
basedpyright
```

### Visualization System

```bash
# Start FastAPI backend (serves analysis results from MongoDB)
uvicorn server.app:app --port 24577 --env-file server/.env

# Start React frontend
cd ui && bun dev --port 24576
```

## Key Implementation Details

### Configuration System

Uses hierarchical configuration with Pydantic models. All configuration classes are imported directly from the main package. Common patterns:

- `SAEConfig` / `CrossCoderConfig` for model architecture
- `TrainerConfig` for training hyperparameters
- `ActivationFactoryConfig` for data loading
- `InitializerConfig` for parameter initialization
- `WandbConfig` for experiment tracking

### SAE Architectures

- **Standard SAE**: Basic sparse autoencoder with optional GLU encoder variants
- **Cross-Layer Transcoder**: Multi-layer attribution architecture implementing circuit tracing methodology  
- **CrossCoder**: Cross-model feature analysis capabilities

### Distributed Computing

Full distributed training support using PyTorch's DTensor with custom dimension mapping strategies. Supports both data and tensor parallelism across multiple GPUs/nodes. Parameters like `data_parallel_size` and `model_parallel_size` control distribution.

### Analysis Pipeline

- **Feature Interpretation**: Automated interpretation using language models (OpenAI API integration)
- **Direct Logit Attribution**: Analysis of how features influence model outputs
- **Circuit Analysis**: Tools for studying feature interactions across layers
- **MongoDB Integration**: Structured storage of analysis results with querying capabilities

### Data Processing

- **Real-time Processing**: On-the-fly activation extraction during training
- **Pre-computed Caching**: Efficient storage and loading of activation datasets via `ActivationFactoryActivationsSource`
- **Distributed Processing**: Scalable activation generation across multiple workers

## Important Notes

### Dependencies

- **Python 3.11** required
- **Package Manager**: Uses `uv` (not pdm) for dependency management
- **Optional Hardware**: NPU support for Huawei Ascend hardware (device="npu")
- **TransformerLens**: Included as local editable dependency for model analysis

### Database Requirements

Analysis results require MongoDB. Configure connection details in environment variables or configuration files:

- `MONGO_URI`: MongoDB connection string
- `MONGO_DB`: Database name (default: "mechinterp")

### Research Applications

This codebase enables research into:

- Mechanistic interpretability of transformer language models
- Sparse feature extraction and circuit analysis
- Cross-layer and cross-model feature studies
- Automated interpretation of learned representations
