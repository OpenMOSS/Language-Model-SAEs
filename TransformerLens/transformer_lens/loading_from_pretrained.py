"""Loading Pretrained Models Utilities.

This module contains functions for loading pretrained models from the Hugging Face Hub.
"""

import dataclasses
import logging
import os
import re
from typing import Dict, Optional, Union, cast

import einops
import torch
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModelForCausalLM, BertForPreTraining

import transformer_lens.utils as utils
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

OFFICIAL_MODEL_NAMES = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
    "facebook/opt-125m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    "facebook/opt-66b",
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-j-6B",
    "EleutherAI/gpt-neox-20b",
    "stanford-crfm/alias-gpt2-small-x21",
    "stanford-crfm/battlestar-gpt2-small-x49",
    "stanford-crfm/caprica-gpt2-small-x81",
    "stanford-crfm/darkmatter-gpt2-small-x343",
    "stanford-crfm/expanse-gpt2-small-x777",
    "stanford-crfm/arwen-gpt2-medium-x21",
    "stanford-crfm/beren-gpt2-medium-x49",
    "stanford-crfm/celebrimbor-gpt2-medium-x81",
    "stanford-crfm/durin-gpt2-medium-x343",
    "stanford-crfm/eowyn-gpt2-medium-x777",
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-31m",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1b-deduped",
    "EleutherAI/pythia-1.4b-deduped",
    "EleutherAI/pythia-2.8b-deduped",
    "EleutherAI/pythia-6.9b-deduped",
    "EleutherAI/pythia-12b-deduped",
    "EleutherAI/pythia-70m-v0",
    "EleutherAI/pythia-160m-v0",
    "EleutherAI/pythia-410m-v0",
    "EleutherAI/pythia-1b-v0",
    "EleutherAI/pythia-1.4b-v0",
    "EleutherAI/pythia-2.8b-v0",
    "EleutherAI/pythia-6.9b-v0",
    "EleutherAI/pythia-12b-v0",
    "EleutherAI/pythia-70m-deduped-v0",
    "EleutherAI/pythia-160m-deduped-v0",
    "EleutherAI/pythia-410m-deduped-v0",
    "EleutherAI/pythia-1b-deduped-v0",
    "EleutherAI/pythia-1.4b-deduped-v0",
    "EleutherAI/pythia-2.8b-deduped-v0",
    "EleutherAI/pythia-6.9b-deduped-v0",
    "EleutherAI/pythia-12b-deduped-v0",
    "EleutherAI/pythia-160m-seed1",
    "EleutherAI/pythia-160m-seed2",
    "EleutherAI/pythia-160m-seed3",
    "NeelNanda/SoLU_1L_v9_old",
    "NeelNanda/SoLU_2L_v10_old",
    "NeelNanda/SoLU_4L_v11_old",
    "NeelNanda/SoLU_6L_v13_old",
    "NeelNanda/SoLU_8L_v21_old",
    "NeelNanda/SoLU_10L_v22_old",
    "NeelNanda/SoLU_12L_v23_old",
    "NeelNanda/SoLU_1L512W_C4_Code",
    "NeelNanda/SoLU_2L512W_C4_Code",
    "NeelNanda/SoLU_3L512W_C4_Code",
    "NeelNanda/SoLU_4L512W_C4_Code",
    "NeelNanda/SoLU_6L768W_C4_Code",
    "NeelNanda/SoLU_8L1024W_C4_Code",
    "NeelNanda/SoLU_10L1280W_C4_Code",
    "NeelNanda/SoLU_12L1536W_C4_Code",
    "NeelNanda/GELU_1L512W_C4_Code",
    "NeelNanda/GELU_2L512W_C4_Code",
    "NeelNanda/GELU_3L512W_C4_Code",
    "NeelNanda/GELU_4L512W_C4_Code",
    "NeelNanda/Attn_Only_1L512W_C4_Code",
    "NeelNanda/Attn_Only_2L512W_C4_Code",
    "NeelNanda/Attn_Only_3L512W_C4_Code",
    "NeelNanda/Attn_Only_4L512W_C4_Code",
    "NeelNanda/Attn-Only-2L512W-Shortformer-6B-big-lr",
    "NeelNanda/SoLU_1L512W_Wiki_Finetune",
    "NeelNanda/SoLU_4L512W_Wiki_Finetune",
    "ArthurConmy/redwood_attn_2l",
    "llama-7b-hf",
    "llama-13b-hf",
    "llama-30b-hf",
    "llama-65b-hf",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "CodeLlama-7b-hf",
    "CodeLlama-7b-Python-hf",
    "CodeLlama-7b-Instruct-hf",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "Baidicoot/Othello-GPT-Transformer-Lens",
    "bert-base-cased",
    "roneneldan/TinyStories-1M",
    "roneneldan/TinyStories-3M",
    "roneneldan/TinyStories-8M",
    "roneneldan/TinyStories-28M",
    "roneneldan/TinyStories-33M",
    "roneneldan/TinyStories-Instruct-1M",
    "roneneldan/TinyStories-Instruct-3M",
    "roneneldan/TinyStories-Instruct-8M",
    "roneneldan/TinyStories-Instruct-28M",
    "roneneldan/TinyStories-Instruct-33M",
    "roneneldan/TinyStories-1Layer-21M",
    "roneneldan/TinyStories-2Layers-33M",
    "roneneldan/TinyStories-Instuct-1Layer-21M",
    "roneneldan/TinyStories-Instruct-2Layers-33M",
    "stabilityai/stablelm-base-alpha-3b",
    "stabilityai/stablelm-base-alpha-7b",
    "stabilityai/stablelm-tuned-alpha-3b",
    "stabilityai/stablelm-tuned-alpha-7b",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-3b",
    "bigscience/bloom-7b1",
    "bigcode/santacoder",
    "Qwen/Qwen-1_8B",
    "Qwen/Qwen-7B",
    "Qwen/Qwen-14B",
    "Qwen/Qwen-1_8B-Chat",
    "Qwen/Qwen-7B-Chat",
    "Qwen/Qwen-14B-Chat",
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen1.5-0.5B-Chat",
    "Qwen/Qwen1.5-1.8B",
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen1.5-4B",
    "Qwen/Qwen1.5-4B-Chat",
    "Qwen/Qwen1.5-7B",
    "Qwen/Qwen1.5-7B-Chat",
    "Qwen/Qwen1.5-14B",
    "Qwen/Qwen1.5-14B-Chat",
    "microsoft/phi-1",
    "microsoft/phi-1_5",
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
    "google/gemma-2b",
    "google/gemma-7b",
    "google/gemma-2b-it",
    "google/gemma-7b-it",
    "01-ai/Yi-6B",
    "01-ai/Yi-34B",
    "01-ai/Yi-6B-Chat",
    "01-ai/Yi-34B-Chat",
    "ai-forever/mGPT",
]
"""Official model names for models on HuggingFace."""

# Model Aliases:
MODEL_ALIASES = {
    "NeelNanda/SoLU_1L_v9_old": ["solu-1l-pile", "solu-1l-old"],
    "NeelNanda/SoLU_2L_v10_old": ["solu-2l-pile", "solu-2l-old"],
    "NeelNanda/SoLU_4L_v11_old": ["solu-4l-pile", "solu-4l-old"],
    "NeelNanda/SoLU_6L_v13_old": ["solu-6l-pile", "solu-6l-old"],
    "NeelNanda/SoLU_8L_v21_old": ["solu-8l-pile", "solu-8l-old"],
    "NeelNanda/SoLU_10L_v22_old": ["solu-10l-pile", "solu-10l-old"],
    "NeelNanda/SoLU_12L_v23_old": ["solu-12l-pile", "solu-12l-old"],
    "NeelNanda/SoLU_1L512W_C4_Code": ["solu-1l", "solu-1l-new", "solu-1l-c4-code"],
    "NeelNanda/SoLU_2L512W_C4_Code": ["solu-2l", "solu-2l-new", "solu-2l-c4-code"],
    "NeelNanda/SoLU_3L512W_C4_Code": ["solu-3l", "solu-3l-new", "solu-3l-c4-code"],
    "NeelNanda/SoLU_4L512W_C4_Code": ["solu-4l", "solu-4l-new", "solu-4l-c4-code"],
    "NeelNanda/GELU_1L512W_C4_Code": ["gelu-1l", "gelu-1l-new", "gelu-1l-c4-code"],
    "NeelNanda/GELU_2L512W_C4_Code": ["gelu-2l", "gelu-2l-new", "gelu-2l-c4-code"],
    "NeelNanda/GELU_3L512W_C4_Code": ["gelu-3l", "gelu-3l-new", "gelu-3l-c4-code"],
    "NeelNanda/GELU_4L512W_C4_Code": ["gelu-4l", "gelu-4l-new", "gelu-4l-c4-code"],
    "NeelNanda/Attn_Only_1L512W_C4_Code": [
        "attn-only-1l",
        "attn-only-1l-new",
        "attn-only-1l-c4-code",
    ],
    "NeelNanda/Attn_Only_2L512W_C4_Code": [
        "attn-only-2l",
        "attn-only-2l-new",
        "attn-only-2l-c4-code",
    ],
    "NeelNanda/Attn_Only_3L512W_C4_Code": [
        "attn-only-3l",
        "attn-only-3l-new",
        "attn-only-3l-c4-code",
    ],
    "NeelNanda/Attn_Only_4L512W_C4_Code": [
        "attn-only-4l",
        "attn-only-4l-new",
        "attn-only-4l-c4-code",
    ],
    "NeelNanda/SoLU_6L768W_C4_Code": ["solu-6l", "solu-6l-new", "solu-6l-c4-code"],
    "NeelNanda/SoLU_8L1024W_C4_Code": ["solu-8l", "solu-8l-new", "solu-8l-c4-code"],
    "NeelNanda/SoLU_10L1280W_C4_Code": ["solu-10l", "solu-10l-new", "solu-10l-c4-code"],
    "NeelNanda/SoLU_12L1536W_C4_Code": ["solu-12l", "solu-12l-new", "solu-12l-c4-code"],
    "NeelNanda/Attn-Only-2L512W-Shortformer-6B-big-lr": [
        "attn-only-2l-demo",
        "attn-only-2l-shortformer-6b-big-lr",
        "attn-only-2l-induction-demo",
        "attn-only-demo",
    ],
    "NeelNanda/SoLU_1L512W_Wiki_Finetune": [
        "solu-1l-wiki",
        "solu-1l-wiki-finetune",
        "solu-1l-finetune",
    ],
    "NeelNanda/SoLU_4L512W_Wiki_Finetune": [
        "solu-4l-wiki",
        "solu-4l-wiki-finetune",
        "solu-4l-finetune",
    ],
    "EleutherAI/pythia-14m": [
        "pythia-14m",
    ],
    "EleutherAI/pythia-31m": [
        "pythia-31m",
    ],
    "EleutherAI/pythia-70m": [
        "pythia-70m",
        "pythia",
        "EleutherAI/pythia-19m",
        "pythia-19m",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-160m": [
        "pythia-160m",
        "EleutherAI/pythia-125m",
        "pythia-125m",  # EleutherAI renamed this model"
    ],
    "EleutherAI/pythia-410m": [
        "pythia-410m",
        "EleutherAI/pythia-350m",
        "pythia-350m",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-1b": [
        "pythia-1b",
        "EleutherAI/pythia-800m",
        "pythia-800m",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-1.4b": [
        "pythia-1.4b",
        "EleutherAI/pythia-1.3b",
        "pythia-1.3b",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-2.8b": [
        "pythia-2.8b",
        "EleutherAI/pythia-2.7b",
        "pythia-2.7b",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-6.9b": [
        "pythia-6.9b",
        "EleutherAI/pythia-6.7b",
        "pythia-6.7b",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-12b": [
        "pythia-12b",
        "EleutherAI/pythia-13b",
        "pythia-13b",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-70m-deduped": [
        "pythia-70m-deduped",
        "EleutherAI/pythia-19m-deduped",  # EleutherAI renamed this model
        "pythia-19m-deduped",
    ],
    "EleutherAI/pythia-160m-deduped": [
        "pythia-160m-deduped",
        "EleutherAI/pythia-125m-deduped",  # EleutherAI renamed this model
        "pythia-125m-deduped",
    ],
    "EleutherAI/pythia-410m-deduped": [
        "pythia-410m-deduped",
        "EleutherAI/pythia-350m-deduped",  # EleutherAI renamed this model
        "pythia-350m-deduped",
    ],
    "EleutherAI/pythia-1b-deduped": [
        "pythia-1b-deduped",
        "EleutherAI/pythia-800m-deduped",  # EleutherAI renamed this model
        "pythia-800m-deduped",
    ],
    "EleutherAI/pythia-1.4b-deduped": [
        "pythia-1.4b-deduped",
        "EleutherAI/pythia-1.3b-deduped",  # EleutherAI renamed this model
        "pythia-1.3b-deduped",
    ],
    "EleutherAI/pythia-2.8b-deduped": [
        "pythia-2.8b-deduped",
        "EleutherAI/pythia-2.7b-deduped",  # EleutherAI renamed this model
        "pythia-2.7b-deduped",
    ],
    "EleutherAI/pythia-6.9b-deduped": [
        "pythia-6.9b-deduped",
        "EleutherAI/pythia-6.7b-deduped",  # EleutherAI renamed this model
        "pythia-6.7b-deduped",
    ],
    "EleutherAI/pythia-12b-deduped": [
        "pythia-12b-deduped",
        "EleutherAI/pythia-13b-deduped",  # EleutherAI renamed this model
        "pythia-13b-deduped",
    ],
    "EleutherAI/pythia-70m-v0": [
        "pythia-70m-v0",
        "pythia-v0",
        "EleutherAI/pythia-19m-v0",
        "pythia-19m-v0",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-160m-v0": [
        "pythia-160m-v0",
        "EleutherAI/pythia-125m-v0",
        "pythia-125m-v0",  # EleutherAI renamed this model"
    ],
    "EleutherAI/pythia-410m-v0": [
        "pythia-410m-v0",
        "EleutherAI/pythia-350m-v0",
        "pythia-350m-v0",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-1b-v0": [
        "pythia-1b-v0",
        "EleutherAI/pythia-800m-v0",
        "pythia-800m-v0",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-1.4b-v0": [
        "pythia-1.4b-v0",
        "EleutherAI/pythia-1.3b-v0",
        "pythia-1.3b-v0",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-2.8b-v0": [
        "pythia-2.8b-v0",
        "EleutherAI/pythia-2.7b-v0",
        "pythia-2.7b-v0",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-6.9b-v0": [
        "pythia-6.9b-v0",
        "EleutherAI/pythia-6.7b-v0",
        "pythia-6.7b-v0",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-12b-v0": [
        "pythia-12b-v0",
        "EleutherAI/pythia-13b-v0",
        "pythia-13b-v0",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-70m-deduped-v0": [
        "pythia-70m-deduped-v0",
        "EleutherAI/pythia-19m-deduped-v0",  # EleutherAI renamed this model
        "pythia-19m-deduped-v0",
    ],
    "EleutherAI/pythia-160m-deduped-v0": [
        "pythia-160m-deduped-v0",
        "EleutherAI/pythia-125m-deduped-v0",  # EleutherAI renamed this model
        "pythia-125m-deduped-v0",
    ],
    "EleutherAI/pythia-410m-deduped-v0": [
        "pythia-410m-deduped-v0",
        "EleutherAI/pythia-350m-deduped-v0",  # EleutherAI renamed this model
        "pythia-350m-deduped-v0",
    ],
    "EleutherAI/pythia-1b-deduped-v0": [
        "pythia-1b-deduped-v0",
        "EleutherAI/pythia-800m-deduped-v0",  # EleutherAI renamed this model
        "pythia-800m-deduped-v0",
    ],
    "EleutherAI/pythia-1.4b-deduped-v0": [
        "pythia-1.4b-deduped-v0",
        "EleutherAI/pythia-1.3b-deduped-v0",  # EleutherAI renamed this model
        "pythia-1.3b-deduped-v0",
    ],
    "EleutherAI/pythia-2.8b-deduped-v0": [
        "pythia-2.8b-deduped-v0",
        "EleutherAI/pythia-2.7b-deduped-v0",  # EleutherAI renamed this model
        "pythia-2.7b-deduped-v0",
    ],
    "EleutherAI/pythia-6.9b-deduped-v0": [
        "pythia-6.9b-deduped-v0",
        "EleutherAI/pythia-6.7b-deduped-v0",  # EleutherAI renamed this model
        "pythia-6.7b-deduped-v0",
    ],
    "EleutherAI/pythia-12b-deduped-v0": [
        "pythia-12b-deduped-v0",
        "EleutherAI/pythia-13b-deduped-v0",  # EleutherAI renamed this model
        "pythia-13b-deduped-v0",
    ],
    "EleutherAI/pythia-160m-seed1": [
        "pythia-160m-seed1",
        "EleutherAI/pythia-125m-seed1",
        "pythia-125m-seed1",  # EleutherAI renamed this model"
    ],
    "EleutherAI/pythia-160m-seed2": [
        "pythia-160m-seed2",
        "EleutherAI/pythia-125m-seed2",
        "pythia-125m-seed2",  # EleutherAI renamed this model"
    ],
    "EleutherAI/pythia-160m-seed3": [
        "pythia-160m-seed3",
        "EleutherAI/pythia-125m-seed3",
        "pythia-125m-seed3",  # EleutherAI renamed this model"
    ],
    "gpt2": ["gpt2-small"],
    "distilgpt2": ["distillgpt2", "distill-gpt2", "distil-gpt2", "gpt2-xs"],
    "facebook/opt-125m": ["opt-125m", "opt-small", "opt"],
    "facebook/opt-1.3b": ["opt-1.3b", "opt-medium"],
    "facebook/opt-2.7b": ["opt-2.7b", "opt-large"],
    "facebook/opt-6.7b": ["opt-6.7b", "opt-xl"],
    "facebook/opt-13b": ["opt-13b", "opt-xxl"],
    "facebook/opt-30b": ["opt-30b", "opt-xxxl"],
    "facebook/opt-66b": ["opt-66b", "opt-xxxxl"],
    "EleutherAI/gpt-neo-125M": ["gpt-neo-125M", "gpt-neo-small", "neo-small", "neo"],
    "EleutherAI/gpt-neo-1.3B": ["gpt-neo-1.3B", "gpt-neo-medium", "neo-medium"],
    "EleutherAI/gpt-neo-2.7B": ["gpt-neo-2.7B", "gpt-neo-large", "neo-large"],
    "EleutherAI/gpt-j-6B": ["gpt-j-6B", "gpt-j", "gptj"],
    "EleutherAI/gpt-neox-20b": ["gpt-neox-20b", "gpt-neox", "neox"],
    "stanford-crfm/alias-gpt2-small-x21": [
        "stanford-gpt2-small-a",
        "alias-gpt2-small-x21",
        "gpt2-mistral-small-a",
        "gpt2-stanford-small-a",
    ],
    "stanford-crfm/battlestar-gpt2-small-x49": [
        "stanford-gpt2-small-b",
        "battlestar-gpt2-small-x49",
        "gpt2-mistral-small-b",
        "gpt2-mistral-small-b",
    ],
    "stanford-crfm/caprica-gpt2-small-x81": [
        "stanford-gpt2-small-c",
        "caprica-gpt2-small-x81",
        "gpt2-mistral-small-c",
        "gpt2-stanford-small-c",
    ],
    "stanford-crfm/darkmatter-gpt2-small-x343": [
        "stanford-gpt2-small-d",
        "darkmatter-gpt2-small-x343",
        "gpt2-mistral-small-d",
        "gpt2-mistral-small-d",
    ],
    "stanford-crfm/expanse-gpt2-small-x777": [
        "stanford-gpt2-small-e",
        "expanse-gpt2-small-x777",
        "gpt2-mistral-small-e",
        "gpt2-mistral-small-e",
    ],
    "stanford-crfm/arwen-gpt2-medium-x21": [
        "stanford-gpt2-medium-a",
        "arwen-gpt2-medium-x21",
        "gpt2-medium-small-a",
        "gpt2-stanford-medium-a",
    ],
    "stanford-crfm/beren-gpt2-medium-x49": [
        "stanford-gpt2-medium-b",
        "beren-gpt2-medium-x49",
        "gpt2-medium-small-b",
        "gpt2-stanford-medium-b",
    ],
    "stanford-crfm/celebrimbor-gpt2-medium-x81": [
        "stanford-gpt2-medium-c",
        "celebrimbor-gpt2-medium-x81",
        "gpt2-medium-small-c",
        "gpt2-medium-small-c",
    ],
    "stanford-crfm/durin-gpt2-medium-x343": [
        "stanford-gpt2-medium-d",
        "durin-gpt2-medium-x343",
        "gpt2-medium-small-d",
        "gpt2-stanford-medium-d",
    ],
    "stanford-crfm/eowyn-gpt2-medium-x777": [
        "stanford-gpt2-medium-e",
        "eowyn-gpt2-medium-x777",
        "gpt2-medium-small-e",
        "gpt2-stanford-medium-e",
    ],
    "ArthurConmy/redwood_attn_2l": ["redwood_attn_2l"],
    "llama-7b-hf": ["llama-7b"],
    "llama-13b-hf": ["llama-13b"],
    "llama-30b-hf": ["llama-30b"],
    "llama-65b-hf": ["llama-65b"],
    "meta-llama/Llama-2-7b-hf": ["Llama-2-7b", "meta-llama/Llama-2-7b-hf"],
    "meta-llama/Llama-2-7b-chat-hf": [
        "Llama-2-7b-chat",
        "meta-llama/Llama-2-7b-chat-hf",
    ],
    "meta-llama/Llama-2-13b-hf": ["Llama-2-13b", "meta-llama/Llama-2-13b-hf"],
    "meta-llama/Llama-2-13b-chat-hf": [
        "Llama-2-13b-chat",
        "meta-llama/Llama-2-13b-chat-hf",
    ],
    "meta-llama/Llama-2-70b-chat-hf": ["Llama-2-70b-chat", "meta-llama-2-70b-chat-hf"],
    "CodeLlama-7b-hf": ["CodeLlamallama-2-7b", "codellama/CodeLlama-7b-hf"],
    "CodeLlama-7b-Python-hf": [
        "CodeLlama-7b-python",
        "codellama/CodeLlama-7b-Python-hf",
    ],
    "CodeLlama-7b-Instruct-hf": [
        "CodeLlama-7b-instruct",
        "codellama/CodeLlama-7b-Instruct-hf",
    ],
    "Baidicoot/Othello-GPT-Transformer-Lens": ["othello-gpt"],
    "roneneldan/TinyStories-1M": ["tiny-stories-1M"],
    "roneneldan/TinyStories-3M": ["tiny-stories-3M"],
    "roneneldan/TinyStories-8M": ["tiny-stories-8M"],
    "roneneldan/TinyStories-28M": ["tiny-stories-28M"],
    "roneneldan/TinyStories-33M": ["tiny-stories-33M"],
    "roneneldan/TinyStories-Instruct-1M": ["tiny-stories-instruct-1M"],
    "roneneldan/TinyStories-Instruct-3M": ["tiny-stories-instruct-3M"],
    "roneneldan/TinyStories-Instruct-8M": ["tiny-stories-instruct-8M"],
    "roneneldan/TinyStories-Instruct-28M": ["tiny-stories-instruct-28M"],
    "roneneldan/TinyStories-Instruct-33M": ["tiny-stories-instruct-33M"],
    "roneneldan/TinyStories-1Layer-21M": ["tiny-stories-1L-21M"],
    "roneneldan/TinyStories-2Layers-33M": ["tiny-stories-2L-33M"],
    "roneneldan/TinyStories-Instuct-1Layer-21M": ["tiny-stories-instruct-1L-21M"],
    "roneneldan/TinyStories-Instruct-2Layers-33M": ["tiny-stories-instruct-2L-33M"],
    "stabilityai/stablelm-base-alpha-3b": [
        "stablelm-base-alpha-3b",
        "stablelm-base-3b",
    ],
    "stabilityai/stablelm-base-alpha-7b": [
        "stablelm-base-alpha-7b",
        "stablelm-base-7b",
    ],
    "stabilityai/stablelm-tuned-alpha-3b": [
        "stablelm-tuned-alpha-3b",
        "stablelm-tuned-3b",
    ],
    "stabilityai/stablelm-tuned-alpha-7b": [
        "stablelm-tuned-alpha-7b",
        "stablelm-tuned-7b",
    ],
    "mistralai/Mistral-7B-v0.1": ["mistral-7b"],
    "mistralai/Mistral-7B-Instruct-v0.1": ["mistral-7b-instruct"],
    "mistralai/Mixtral-8x7B-v0.1": ["mixtral", "mixtral-8x7b"],
    "mistralai/Mixtral-8x7B-Instruct-v0.1": [
        "mixtral-instruct",
        "mixtral-8x7b-instruct",
    ],
    "bigscience/bloom-560m": ["bloom-560m"],
    "bigscience/bloom-1b1": ["bloom-1b1"],
    "bigscience/bloom-1b7": ["bloom-1b7"],
    "bigscience/bloom-3b": ["bloom-3b"],
    "bigscience/bloom-7b1": ["bloom-7b1"],
    "bigcode/santacoder": ["santacoder"],
    "Qwen/Qwen-1_8B": ["qwen-1.8b"],
    "Qwen/Qwen-7B": ["qwen-7b"],
    "Qwen/Qwen-14B": ["qwen-14b"],
    "Qwen/Qwen-1_8B-Chat": ["qwen-1.8b-chat"],
    "Qwen/Qwen-7B-Chat": ["qwen-7b-chat"],
    "Qwen/Qwen-14B-Chat": ["qwen-14b-chat"],
    "Qwen/Qwen1.5-0.5B": ["qwen1.5-0.5b"],
    "Qwen/Qwen1.5-0.5B-Chat": ["qwen1.5-0.5b-chat"],
    "Qwen/Qwen1.5-1.8B": ["qwen1.5-1.8b"],
    "Qwen/Qwen1.5-1.8B-Chat": ["qwen1.5-1.8b-chat"],
    "Qwen/Qwen1.5-4B": ["qwen1.5-4b"],
    "Qwen/Qwen1.5-4B-Chat": ["qwen1.5-4b-chat"],
    "Qwen/Qwen1.5-7B": ["qwen1.5-7b"],
    "Qwen/Qwen1.5-7B-Chat": ["qwen1.5-7b-chat"],
    "Qwen/Qwen1.5-14B": ["qwen1.5-14b"],
    "Qwen/Qwen1.5-14B-Chat": ["qwen1.5-14b-chat"],
    "microsoft/phi-1": ["phi-1"],
    "microsoft/phi-1_5": ["phi-1_5"],
    "microsoft/phi-2": ["phi-2"],
    "microsoft/Phi-3-mini-4k-instruct": ["phi-3"],
    "google/gemma-2b": ["gemma-2b"],
    "google/gemma-7b": ["gemma-7b"],
    "google/gemma-2b-it": ["gemma-2b-it"],
    "google/gemma-7b-it": ["gemma-7b-it"],
    "01-ai/Yi-6B": ["yi-6b", "Yi-6B"],
    "01-ai/Yi-34B": ["yi-34b", "Yi-34B"],
    "01-ai/Yi-6B-Chat": ["yi-6b-chat", "Yi-6B-Chat"],
    "01-ai/Yi-34B-Chat": ["yi-34b-chat", "Yi-34B-Chat"],
    "ai-forever/mGPT": ["mGPT"],
}
"""Model aliases for models on HuggingFace."""

NON_HF_HOSTED_MODEL_NAMES = [
    "llama-7b-hf",
    "llama-13b-hf",
    "llama-30b-hf",
    "llama-65b-hf",
]
"""Official model names for models not hosted on HuggingFace."""

# Sets a default model alias, by convention the first one in the model alias table, else the official name if it has no aliases
DEFAULT_MODEL_ALIASES = [
    MODEL_ALIASES[name][0] if name in MODEL_ALIASES else name for name in OFFICIAL_MODEL_NAMES
]

NEED_REMOTE_CODE_MODELS = (
    "bigcode/santacoder",
    "Qwen/Qwen-",
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
)


def make_model_alias_map():
    """
    Converts OFFICIAL_MODEL_NAMES (the list of actual model names on
    HuggingFace) and MODEL_ALIASES (a dictionary mapping official model names to
    aliases) into a dictionary mapping all aliases to the official model name.
    """
    model_alias_map = {}
    for official_model_name in OFFICIAL_MODEL_NAMES:
        aliases = MODEL_ALIASES.get(official_model_name, [])
        for alias in aliases:
            model_alias_map[alias.lower()] = official_model_name
        model_alias_map[official_model_name.lower()] = official_model_name
    return model_alias_map


def get_official_model_name(model_name: str):
    """
    Returns the official model name for a given model name (or alias).
    """
    model_alias_map = make_model_alias_map()
    official_model_name = model_alias_map.get(model_name.lower(), None)
    if official_model_name is None:
        raise ValueError(
            f"{model_name} not found. Valid official model names (excl aliases): {OFFICIAL_MODEL_NAMES}"
        )
    return official_model_name


def convert_hf_model_config(model_name: str, **kwargs):
    """
    Returns the model config for a HuggingFace model, converted to a dictionary
    in the HookedTransformerConfig format.

    Takes the official_model_name as an input.
    """
    # In case the user passed in an alias
    official_model_name = get_official_model_name(model_name)
    # Load HuggingFace model config
    if "llama" in official_model_name.lower():
        architecture = "LlamaForCausalLM"
    elif "gemma" in official_model_name.lower():
        architecture = "GemmaForCausalLM"
    else:
        huggingface_token = os.environ.get("HF_TOKEN", None)
        hf_config = AutoConfig.from_pretrained(
            official_model_name,
            token=huggingface_token,
            **kwargs,
        )
        architecture = hf_config.architectures[0]

    if official_model_name.startswith(
        ("llama-7b", "meta-llama/Llama-2-7b")
    ):  # same architecture for LLaMA and Llama-2
        cfg_dict = {
            "d_model": 4096,
            "d_head": 4096 // 32,
            "n_heads": 32,
            "d_mlp": 11008,
            "n_layers": 32,
            "n_ctx": 2048 if official_model_name.startswith("llama-7b") else 4096,
            "eps": 1e-6 if official_model_name.startswith("llama-7b") else 1e-5,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 4096 // 32,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif official_model_name.startswith("CodeLlama-7b"):  # same architecture CodeLlama and Llama-2
        cfg_dict = {
            "d_model": 4096,
            "d_head": 4096 // 32,
            "n_heads": 32,
            "d_mlp": 11008,
            "n_layers": 32,
            "n_ctx": 4096,
            "eps": 1e-5,
            "d_vocab": 32016,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": 4096 // 32,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 1000000,
        }
        if "python" in official_model_name.lower():
            # The vocab size of python version of CodeLlama-7b is 32000
            cfg_dict["d_vocab"] = 32000
    elif official_model_name.startswith(
        ("llama-13b", "meta-llama/Llama-2-13b")
    ):  # same architecture for LLaMA and Llama-2
        cfg_dict = {
            "d_model": 5120,
            "d_head": 5120 // 40,
            "n_heads": 40,
            "d_mlp": 13824,
            "n_layers": 40,
            "n_ctx": 2048 if official_model_name.startswith("llama-13b") else 4096,
            "eps": 1e-6 if official_model_name.startswith("llama-13b") else 1e-5,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 5120 // 40,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "llama-30b" in official_model_name:
        cfg_dict = {
            "d_model": 6656,
            "d_head": 6656 // 52,
            "n_heads": 52,
            "d_mlp": 17920,
            "n_layers": 60,
            "n_ctx": 2048,
            "eps": 1e-6,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 6656 // 52,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "llama-65b" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 8192 // 64,
            "n_heads": 64,
            "d_mlp": 22016,
            "n_layers": 80,
            "n_ctx": 2048,
            "eps": 1e-6,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": 8192 // 64,
            "rotary_adjacent_pairs": False,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "Llama-2-70b" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 128,
            "n_heads": 64,
            "d_mlp": 28672,
            "n_layers": 80,
            "n_ctx": 4096,
            "eps": 1e-5,
            "d_vocab": 32000,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "Meta-Llama-3-8B" in official_model_name:
        cfg_dict = {
            "d_model": 4096,
            "d_head": 128,
            "n_heads": 32,
            "d_mlp": 14336,
            "n_layers": 32,
            "n_ctx": 8192,
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "Meta-Llama-3-70B" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 128,
            "n_heads": 64,
            "d_mlp": 28672,
            "n_layers": 80,
            "n_ctx": 8192,
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif architecture == "GPTNeoForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_heads,
            "n_heads": hf_config.num_heads,
            "d_mlp": hf_config.hidden_size * 4,
            "n_layers": hf_config.num_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "attn_types": hf_config.attention_layers,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": False,
            "use_local_attn": True,
            "window_size": hf_config.window_size,
            "scale_attn_by_inverse_layer_idx": False,
            "normalization_type": "LN",
        }
    elif architecture == "GPT2LMHeadModel":
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": hf_config.n_embd * 4,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_ctx,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
            "normalization_type": "LN",
        }
    elif architecture == "OPTForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.ffn_dim,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": 1e-5,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "normalization_type": "LN",
        }
    elif architecture == "GPTJForCausalLM":
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": 4 * hf_config.n_embd,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_positions,
            "eps": 1e-5,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "parallel_attn_mlp": True,
            "positional_embedding_type": "rotary",
            "rotary_dim": hf_config.rotary_dim,
            "rotary_adjacent_pairs": True,
            "normalization_type": "LN",
        }
    elif architecture == "GPTNeoXForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "parallel_attn_mlp": True,
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "normalization_type": "LN",
        }
        rotary_pct = hf_config.rotary_pct
        cfg_dict["rotary_dim"] = round(rotary_pct * cfg_dict["d_head"])
    elif architecture == "BertForMaskedLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": "gelu",
            "attention_dir": "bidirectional",
        }
    elif architecture == "MistralForCausalLM":
        cfg_dict = {
            "d_model": 4096,
            "d_head": 4096 // 32,
            "n_heads": 32,
            "d_mlp": 14336,
            "n_layers": 32,
            "n_ctx": 2048,  # Capped due to memory issues
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "window_size": 4096,
            "attn_types": ["local"] * 32,
            "eps": 1e-05,
            "n_key_value_heads": 8,
            "gated_mlp": True,
            "use_local_attn": True,
            "rotary_dim": 4096 // 32,
        }
    elif architecture == "MixtralForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": 2048,  # hf_config.max_position_embeddings, # Capped due to memory issues
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "window_size": hf_config.sliding_window,  # This is None, as no sliding window was used
            "attn_types": ["global"] * 32,
            "eps": hf_config.rms_norm_eps,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "gated_mlp": True,
            "use_local_attn": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "num_experts": hf_config.num_local_experts,
            "experts_per_token": hf_config.num_experts_per_tok,
        }
    elif architecture == "BloomForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": hf_config.hidden_size * 4,
            "n_layers": hf_config.n_layer,
            "n_ctx": 2048,  # Capped due to HF Tokenizer Constraints
            "d_vocab": hf_config.vocab_size,
            "act_fn": "gelu_fast",
            "eps": hf_config.layer_norm_epsilon,
            "normalization_type": "LN",
            "post_embedding_ln": True,
            "positional_embedding_type": "alibi",
        }
    elif architecture == "GPT2LMHeadCustomModel":
        # santacoder
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": hf_config.n_embd * 4,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_positions,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "trust_remote_code": "santacoder"
            in official_model_name,  # Only santacoder needs trust_remote_code
            "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
            "normalization_type": "LN",
        }
    elif architecture == "LlamaForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "n_key_value_heads": (
                hf_config.num_key_value_heads
                if hf_config.num_key_value_heads != hf_config.num_attention_heads
                else None
            ),
            # This is done because the current implementation of GQA will use Grouped-Query Attention if
            # n_key_value_heads is not None, but hf_config.num_key_value_heads is sometimes specified as
            # the same as hf_config.num_attention_heads, in which case GQA should not be used.
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif architecture == "QWenLMHeadModel":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size // 2,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": 2048,  # Capped bc the actual ctx length is 30k and the attn mask would be too big
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "act_fn": "silu",
            "use_attn_scale": hf_config.scale_attn_weights,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": hf_config.kv_channels,
            "rotary_adjacent_pairs": False,
            "tokenizer_prepends_bos": True,
            "trust_remote_code": True,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif architecture == "Qwen2ForCausalLM":
        # Note that Qwen1.5 models have architecture type Qwen2ForCausalLM.
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": 2048,  # Capped bc the actual ctx length is 30k and the attn mask would be too big
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_base": hf_config.rope_theta,
            "rotary_adjacent_pairs": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "tokenizer_prepends_bos": True,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif architecture == "PhiForCausalLM":
        # Architecture for microsoft/phi models
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "LN",
            "positional_embedding_type": "rotary",
            "trust_remote_code": True,
            "rotary_base": hf_config.rope_theta,
            "use_attn_scale": True,
            "parallel_attn_mlp": True,
        }
        partial_rotary_factor = hf_config.partial_rotary_factor
        cfg_dict["rotary_dim"] = round(partial_rotary_factor * cfg_dict["d_head"])
    elif architecture == "Phi3ForCausalLM":
        # Architecture for microsoft/phi3 models
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "trust_remote_code": True,
            "rotary_base": hf_config.rope_theta,
            "use_attn_scale": True,
            "gated_mlp": True,
            "parallel_attn_mlp": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
        }

    elif official_model_name.startswith("google/gemma-2b"):
        # Architecture for Gemma 2b and Gemma 2b Instruct models
        cfg_dict = {
            "d_model": 2048,
            "d_head": 256,
            "n_heads": 8,
            "d_mlp": 16384,
            "n_layers": 18,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_new",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "rotary_dim": 256,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 1,
            "gated_mlp": True,
            "final_rms": True,
        }
    elif official_model_name.startswith("google/gemma-7b"):
        # Architecture for Gemma 7b and Gemma 7b Instruct models
        cfg_dict = {
            "d_model": 3072,
            "d_head": 256,
            "n_heads": 16,
            "d_mlp": 24576,
            "n_layers": 28,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_new",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "rotary_dim": 256,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 16,
            "gated_mlp": True,
            "final_rms": True,
        }
    else:
        raise NotImplementedError(f"{architecture} is not currently supported.")
    # All of these models use LayerNorm
    cfg_dict["original_architecture"] = architecture
    # The name such that AutoTokenizer.from_pretrained works
    cfg_dict["tokenizer_name"] = official_model_name
    if kwargs.get("trust_remote_code", False):
        cfg_dict["trust_remote_code"] = True
    return cfg_dict


def convert_neel_model_config(official_model_name: str, **kwargs):
    """
    Loads the config for a model trained by me (NeelNanda), converted to a dictionary
    in the HookedTransformerConfig format.

    AutoConfig is not supported, because these models are in the HookedTransformer format, so we directly download and load the json.
    """
    official_model_name = get_official_model_name(official_model_name)
    cfg_json: dict = utils.download_file_from_hf(official_model_name, "config.json", **kwargs)
    cfg_arch = cfg_json.get(
        "architecture", "neel" if "_old" not in official_model_name else "neel-solu-old"
    )
    cfg_dict = {
        "d_model": cfg_json["d_model"],
        "n_layers": cfg_json["n_layers"],
        "d_mlp": cfg_json["d_mlp"],
        "d_head": cfg_json["d_head"],
        "n_heads": cfg_json["n_heads"],
        "n_ctx": cfg_json["n_ctx"],
        "d_vocab": cfg_json["d_vocab"],
        "tokenizer_name": cfg_json.get("tokenizer_name", None),
        "act_fn": cfg_json["act_fn"],
        "attn_only": cfg_json["attn_only"],
        "final_rms": cfg_json.get("final_rms", False),
        "original_architecture": cfg_arch,
    }
    if "normalization" in cfg_json:
        cfg_dict["normalization_type"] = cfg_json["normalization"]
    else:
        cfg_dict["normalization_type"] = cfg_json["normalization_type"]
    if "shortformer_pos" in cfg_json:
        cfg_dict["positional_embedding_type"] = (
            "shortformer" if cfg_json["shortformer_pos"] else "standard"
        )
    else:
        cfg_dict["positional_embedding_type"] = "standard"
    return cfg_dict


def get_pretrained_model_config(
    model_name: str,
    hf_cfg: Optional[dict] = None,
    checkpoint_index: Optional[int] = None,
    checkpoint_value: Optional[int] = None,
    fold_ln: bool = False,
    use_flash_attn: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    n_devices: int = 1,
    default_prepend_bos: bool = True,
    dtype: torch.dtype = torch.float32,
    **kwargs,
):
    """Returns the pretrained model config as an HookedTransformerConfig object.

    There are two types of pretrained models: HuggingFace models (where
    AutoModel and AutoConfig work), and models trained by me (NeelNanda) which
    aren't as integrated with HuggingFace infrastructure.

    Args:
        model_name: The name of the model. This can be either the official
            HuggingFace model name, or the name of a model trained by me
            (NeelNanda).
        hf_cfg (dict, optional): Config of a loaded pretrained HF model,
            converted to a dictionary.
        checkpoint_index (int, optional): If loading from a
            checkpoint, the index of the checkpoint to load. Defaults to None.
        checkpoint_value (int, optional): If loading from a checkpoint, the
        value of
            the checkpoint to load, ie the step or token number (each model has
            checkpoints labelled with exactly one of these). Defaults to None.
        fold_ln (bool, optional): Whether to fold the layer norm into the
            subsequent linear layers (see HookedTransformer.fold_layer_norm for
            details). Defaults to False.
        use_flash_attn (bool): whether to use FlashAttention-2. Please refer to
            https://github.com/Dao-AILab/flash-attention. Defaults to False.
        device (str, optional): The device to load the model onto. By
            default will load to CUDA if available, else CPU.
        n_devices (int, optional): The number of devices to split the model across. Defaults to 1.
        default_prepend_bos (bool, optional): Default behavior of whether to prepend the BOS token when the
            methods of HookedTransformer process input text to tokenize (only when input is a string).
            Defaults to True - even for models not explicitly trained with this, heads often use the
            first position as a resting position and accordingly lose information from the first token,
            so this empirically seems to give better results. To change the default behavior to False, pass in
            default_prepend_bos=False. Note that you can also locally override the default behavior by passing
            in prepend_bos=True/False when you call a method that processes the input string.
        dtype (torch.dtype, optional): The dtype to load the TransformerLens model in.
        kwargs: Other optional arguments passed to HuggingFace's from_pretrained.
            Also given to other HuggingFace functions when compatible.

    """
    official_model_name = get_official_model_name(model_name)
    if (
        official_model_name.startswith("NeelNanda")
        or official_model_name.startswith("ArthurConmy")
        or official_model_name.startswith("Baidicoot")
    ):
        cfg_dict = convert_neel_model_config(official_model_name, **kwargs)
    else:
        if official_model_name.startswith(NEED_REMOTE_CODE_MODELS) and not kwargs.get(
            "trust_remote_code", False
        ):
            logging.warning(
                f"Loading model {official_model_name} requires setting trust_remote_code=True"
            )
            kwargs["trust_remote_code"] = True
        cfg_dict = convert_hf_model_config(official_model_name, **kwargs)
    # Processing common to both model types
    # Remove any prefix, saying the organization who made a model.
    cfg_dict["model_name"] = official_model_name.split("/")[-1]
    # Don't need to initialize weights, we're loading from pretrained
    cfg_dict["init_weights"] = False

    if (
        "positional_embedding_type" in cfg_dict
        and cfg_dict["positional_embedding_type"] == "shortformer"
        and fold_ln
    ):
        logging.warning(
            "You tried to specify fold_ln=True for a shortformer model, but this can't be done! Setting fold_ln=False instead."
        )
        fold_ln = False

    if device is not None:
        cfg_dict["device"] = device

    cfg_dict["dtype"] = dtype

    if fold_ln:
        if cfg_dict["normalization_type"] in ["LN", "LNPre"]:
            cfg_dict["normalization_type"] = "LNPre"
        elif cfg_dict["normalization_type"] in ["RMS", "RMSPre"]:
            cfg_dict["normalization_type"] = "RMSPre"
        else:
            logging.warning("Cannot fold in layer norm, normalization_type is not LN.")
    if use_flash_attn:
        cfg_dict["use_flash_attn"] = True

    if checkpoint_index is not None or checkpoint_value is not None:
        checkpoint_labels, checkpoint_label_type = get_checkpoint_labels(
            official_model_name,
            **kwargs,
        )
        cfg_dict["from_checkpoint"] = True
        cfg_dict["checkpoint_label_type"] = checkpoint_label_type
        if checkpoint_index is not None:
            cfg_dict["checkpoint_index"] = checkpoint_index
            cfg_dict["checkpoint_value"] = checkpoint_labels[checkpoint_index]
        elif checkpoint_value is not None:
            assert (
                checkpoint_value in checkpoint_labels
            ), f"Checkpoint value {checkpoint_value} is not in list of available checkpoints"
            cfg_dict["checkpoint_value"] = checkpoint_value
            cfg_dict["checkpoint_index"] = checkpoint_labels.index(checkpoint_value)
    else:
        cfg_dict["from_checkpoint"] = False

    cfg_dict["device"] = device
    cfg_dict["n_devices"] = n_devices
    cfg_dict["default_prepend_bos"] = default_prepend_bos
    if hf_cfg is not None:
        cfg_dict["load_in_4bit"] = hf_cfg.get("quantization_config", {}).get("load_in_4bit", False)

    cfg = HookedTransformerConfig.from_dict(cfg_dict)
    return cfg


def get_num_params_of_pretrained(model_name):
    """
    Returns the number of parameters of a pretrained model, used to filter to only run code for sufficiently small models.
    """
    cfg = get_pretrained_model_config(model_name)
    return cfg.n_params


# %% Load checkpointed model state dicts
# The steps for which there are checkpoints in the stanford crfm models
STANFORD_CRFM_CHECKPOINTS = (
    list(range(0, 100, 10))
    + list(range(100, 2000, 50))
    + list(range(2000, 20000, 100))
    + list(range(20000, 400000 + 1, 1000))
)

# Linearly spaced checkpoints for Pythia models, taken every 1000 steps.
# Batch size 2,097,152 tokens, so checkpoints every 2.1B tokens
PYTHIA_CHECKPOINTS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(
    range(1000, 143000 + 1, 1000)
)
# Pythia V1 has log-spaced early checkpoints (see line above), but V0 doesn't
PYTHIA_V0_CHECKPOINTS = list(range(1000, 143000 + 1, 1000))


def get_checkpoint_labels(model_name: str, **kwargs):
    """Returns the checkpoint labels for a given model, and the label_type
    (step or token). Raises an error for models that are not checkpointed."""
    official_model_name = get_official_model_name(model_name)
    if official_model_name.startswith("stanford-crfm/"):
        return STANFORD_CRFM_CHECKPOINTS, "step"
    elif official_model_name.startswith("EleutherAI/pythia"):
        if "v0" in official_model_name:
            return PYTHIA_V0_CHECKPOINTS, "step"
        else:
            logging.warning(
                "Pythia models on HF were updated on 4/3/23! add '-v0' to model name to access the old models."
            )
            return PYTHIA_CHECKPOINTS, "step"
    elif official_model_name.startswith("NeelNanda/"):
        api = HfApi()
        files_list = api.list_repo_files(
            official_model_name,
            **utils.select_compatible_kwargs(kwargs, api.list_repo_files),
        )
        labels = []
        for file_name in files_list:
            match = re.match(r"checkpoints/.*_(\d*)\.pth", file_name)
            if match:
                labels.append(int(match.group(1)))
        if labels[-1] > 1e9:
            label_type = "token"
        else:
            label_type = "step"
        return labels, label_type
    else:
        raise ValueError(f"Model {official_model_name} is not checkpointed.")


# %% Loading state dicts
def get_pretrained_state_dict(
    official_model_name: str,
    cfg: HookedTransformerConfig,
    hf_model=None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Loads in the model weights for a pretrained model, and processes them to
    have the HookedTransformer parameter names and shapes. Supports checkpointed
    models (and expects the checkpoint info to be stored in the config object)

    hf_model: Optionally, a HuggingFace model object. If provided, we will use
        these weights rather than reloading the model.
    dtype: The dtype to load the HuggingFace model in.
    kwargs: Other optional arguments passed to HuggingFace's from_pretrained.
        Also given to other HuggingFace functions when compatible.
    """
    if "torch_dtype" in kwargs:
        dtype = kwargs["torch_dtype"]
        del kwargs["torch_dtype"]
    official_model_name = get_official_model_name(official_model_name)
    if official_model_name.startswith(NEED_REMOTE_CODE_MODELS) and not kwargs.get(
        "trust_remote_code", False
    ):
        logging.warning(
            f"Loading model {official_model_name} state dict requires setting trust_remote_code=True"
        )
        kwargs["trust_remote_code"] = True
    if (
        official_model_name.startswith("NeelNanda")
        or official_model_name.startswith("ArthurConmy")
        or official_model_name.startswith("Baidicoot")
    ):
        api = HfApi()
        repo_files = api.list_repo_files(
            official_model_name,
            **utils.select_compatible_kwargs(kwargs, api.list_repo_files),
        )
        if cfg.from_checkpoint:
            file_name = list(
                filter(lambda x: x.endswith(f"{cfg.checkpoint_value}.pth"), repo_files)
            )[0]
        else:
            file_name = list(filter(lambda x: x.endswith("final.pth"), repo_files))[0]
        state_dict = utils.download_file_from_hf(official_model_name, file_name, **kwargs)

        # Convert to dtype
        state_dict = {k: v.to(dtype) for k, v in state_dict.items()}

        if cfg.original_architecture == "neel-solu-old":
            state_dict = convert_neel_solu_old_weights(state_dict, cfg)
        elif cfg.original_architecture == "mingpt":
            state_dict = convert_mingpt_weights(state_dict, cfg)
        return state_dict
    else:
        if cfg.from_checkpoint:
            huggingface_token = os.environ.get("HF_TOKEN", None)
            if official_model_name.startswith("stanford-crfm"):
                hf_model = AutoModelForCausalLM.from_pretrained(
                    official_model_name,
                    revision=f"checkpoint-{cfg.checkpoint_value}",
                    torch_dtype=dtype,
                    token=huggingface_token,
                    **kwargs,
                )
            elif official_model_name.startswith("EleutherAI/pythia"):
                hf_model = AutoModelForCausalLM.from_pretrained(
                    official_model_name,
                    revision=f"step{cfg.checkpoint_value}",
                    torch_dtype=dtype,
                    token=huggingface_token,
                    **kwargs,
                )
            else:
                raise ValueError(f"Checkpoints for model {official_model_name} are not supported")
        elif hf_model is None:
            huggingface_token = os.environ.get("HF_TOKEN", None)
            if official_model_name in NON_HF_HOSTED_MODEL_NAMES:
                raise NotImplementedError("Model not hosted on HuggingFace, must pass in hf_model")
            elif "bert" in official_model_name:
                hf_model = BertForPreTraining.from_pretrained(
                    official_model_name,
                    torch_dtype=dtype,
                    token=huggingface_token,
                    **kwargs,
                )
            else:
                hf_model = AutoModelForCausalLM.from_pretrained(
                    official_model_name,
                    torch_dtype=dtype,
                    token=huggingface_token,
                    **kwargs,
                )

            # Load model weights, and fold in layer norm weights

        for param in hf_model.parameters():
            param.requires_grad = False

        if cfg.original_architecture == "GPT2LMHeadModel":
            state_dict = convert_gpt2_weights(hf_model, cfg)
        elif cfg.original_architecture == "GPTNeoForCausalLM":
            state_dict = convert_neo_weights(hf_model, cfg)
        elif cfg.original_architecture == "OPTForCausalLM":
            state_dict = convert_opt_weights(hf_model, cfg)
        elif cfg.original_architecture == "GPTJForCausalLM":
            state_dict = convert_gptj_weights(hf_model, cfg)
        elif cfg.original_architecture == "GPTNeoXForCausalLM":
            state_dict = convert_neox_weights(hf_model, cfg)
        elif cfg.original_architecture == "LlamaForCausalLM":
            state_dict = convert_llama_weights(hf_model, cfg)
        elif cfg.original_architecture == "BertForMaskedLM":
            state_dict = convert_bert_weights(hf_model, cfg)
        elif cfg.original_architecture == "MistralForCausalLM":
            state_dict = convert_mistral_weights(hf_model, cfg)
        elif cfg.original_architecture == "MixtralForCausalLM":
            state_dict = convert_mixtral_weights(hf_model, cfg)
        elif cfg.original_architecture == "BloomForCausalLM":
            state_dict = convert_bloom_weights(hf_model, cfg)
        elif cfg.original_architecture == "GPT2LMHeadCustomModel":
            state_dict = convert_coder_weights(hf_model, cfg)
        elif cfg.original_architecture == "QWenLMHeadModel":
            state_dict = convert_qwen_weights(hf_model, cfg)
        elif cfg.original_architecture == "Qwen2ForCausalLM":
            state_dict = convert_qwen2_weights(hf_model, cfg)
        elif cfg.original_architecture == "PhiForCausalLM":
            state_dict = convert_phi_weights(hf_model, cfg)
        elif cfg.original_architecture == "Phi3ForCausalLM":
            state_dict = convert_phi3_weights(hf_model, cfg)
        elif cfg.original_architecture == "GemmaForCausalLM":
            state_dict = convert_gemma_weights(hf_model, cfg)
        else:
            raise ValueError(
                f"Loading weights from the architecture is not currently supported: {cfg.original_architecture}, generated from model name {cfg.model_name}. Feel free to open an issue on GitHub to request this feature."
            )

        return state_dict


def fill_missing_keys(model, state_dict):
    """Takes in a state dict from a pretrained model, and fills in any missing keys with the default initialization.

    This function is assumed to be run before weights are initialized.

    Args:
        state_dict (dict): State dict from a pretrained model

    Returns:
        dict: State dict with missing keys filled in
    """
    # Get the default state dict
    default_state_dict = model.state_dict()
    # Get the keys that are missing from the pretrained model
    missing_keys = set(default_state_dict.keys()) - set(state_dict.keys())
    # Fill in the missing keys with the default initialization
    for key in missing_keys:
        if "hf_model" in key:
            # Skip keys that are from the HuggingFace model, if loading from HF.
            continue
        if "W_" in key:
            logging.warning(
                "Missing key for a weight matrix in pretrained, filled in with an empty tensor: {}".format(
                    key
                )
            )
        state_dict[key] = default_state_dict[key]
    return state_dict


# Convert state dicts
def convert_gpt2_weights(gpt2, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = gpt2.transformer.wte.weight
    state_dict["pos_embed.W_pos"] = gpt2.transformer.wpe.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = gpt2.transformer.h[l].ln_1.weight
        state_dict[f"blocks.{l}.ln1.b"] = gpt2.transformer.h[l].ln_1.bias

        # In GPT-2, q,k,v are produced by one big linear map, whose output is
        # concat([q, k, v])
        W = gpt2.transformer.h[l].attn.c_attn.weight
        W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=1)
        W_Q = einops.rearrange(W_Q, "m (i h)->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "m (i h)->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "m (i h)->i m h", i=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        qkv_bias = gpt2.transformer.h[l].attn.c_attn.bias
        qkv_bias = einops.rearrange(
            qkv_bias,
            "(qkv index head)->qkv index head",
            qkv=3,
            index=cfg.n_heads,
            head=cfg.d_head,
        )
        state_dict[f"blocks.{l}.attn.b_Q"] = qkv_bias[0]
        state_dict[f"blocks.{l}.attn.b_K"] = qkv_bias[1]
        state_dict[f"blocks.{l}.attn.b_V"] = qkv_bias[2]

        W_O = gpt2.transformer.h[l].attn.c_proj.weight
        W_O = einops.rearrange(W_O, "(i h) m->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = gpt2.transformer.h[l].attn.c_proj.bias

        state_dict[f"blocks.{l}.ln2.w"] = gpt2.transformer.h[l].ln_2.weight
        state_dict[f"blocks.{l}.ln2.b"] = gpt2.transformer.h[l].ln_2.bias

        W_in = gpt2.transformer.h[l].mlp.c_fc.weight
        state_dict[f"blocks.{l}.mlp.W_in"] = W_in
        state_dict[f"blocks.{l}.mlp.b_in"] = gpt2.transformer.h[l].mlp.c_fc.bias

        W_out = gpt2.transformer.h[l].mlp.c_proj.weight
        state_dict[f"blocks.{l}.mlp.W_out"] = W_out
        state_dict[f"blocks.{l}.mlp.b_out"] = gpt2.transformer.h[l].mlp.c_proj.bias
    state_dict["unembed.W_U"] = gpt2.lm_head.weight.T

    state_dict["ln_final.w"] = gpt2.transformer.ln_f.weight
    state_dict["ln_final.b"] = gpt2.transformer.ln_f.bias
    return state_dict


def convert_neo_weights(neo, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = neo.transformer.wte.weight
    state_dict["pos_embed.W_pos"] = neo.transformer.wpe.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = neo.transformer.h[l].ln_1.weight
        state_dict[f"blocks.{l}.ln1.b"] = neo.transformer.h[l].ln_1.bias

        W_Q = neo.transformer.h[l].attn.attention.q_proj.weight
        W_K = neo.transformer.h[l].attn.attention.k_proj.weight
        W_V = neo.transformer.h[l].attn.attention.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(i h) m->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(i h) m->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(i h) m->i m h", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.b_K"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)

        W_O = neo.transformer.h[l].attn.attention.out_proj.weight
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = neo.transformer.h[l].attn.attention.out_proj.bias

        state_dict[f"blocks.{l}.ln2.w"] = neo.transformer.h[l].ln_2.weight
        state_dict[f"blocks.{l}.ln2.b"] = neo.transformer.h[l].ln_2.bias

        state_dict[f"blocks.{l}.mlp.W_in"] = neo.transformer.h[l].mlp.c_fc.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = neo.transformer.h[l].mlp.c_fc.bias

        state_dict[f"blocks.{l}.mlp.W_out"] = neo.transformer.h[l].mlp.c_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = neo.transformer.h[l].mlp.c_proj.bias
    state_dict["ln_final.w"] = neo.transformer.ln_f.weight
    state_dict["ln_final.b"] = neo.transformer.ln_f.bias

    state_dict["unembed.W_U"] = neo.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)
    return state_dict


def convert_gptj_weights(gptj, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = gptj.transformer.wte.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = gptj.transformer.h[l].ln_1.weight
        state_dict[f"blocks.{l}.ln1.b"] = gptj.transformer.h[l].ln_1.bias

        W_Q = gptj.transformer.h[l].attn.q_proj.weight
        W_K = gptj.transformer.h[l].attn.k_proj.weight
        W_V = gptj.transformer.h[l].attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(i h) m->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(i h) m->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(i h) m->i m h", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.b_K"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)

        W_O = gptj.transformer.h[l].attn.out_proj.weight
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        # Layer Norm 1 and 2 are tied.
        state_dict[f"blocks.{l}.ln2.w"] = state_dict[f"blocks.{l}.ln1.w"]
        state_dict[f"blocks.{l}.ln2.b"] = state_dict[f"blocks.{l}.ln1.b"]

        state_dict[f"blocks.{l}.mlp.W_in"] = gptj.transformer.h[l].mlp.fc_in.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = gptj.transformer.h[l].mlp.fc_in.bias

        state_dict[f"blocks.{l}.mlp.W_out"] = gptj.transformer.h[l].mlp.fc_out.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = gptj.transformer.h[l].mlp.fc_out.bias
    state_dict["ln_final.w"] = gptj.transformer.ln_f.weight
    state_dict["ln_final.b"] = gptj.transformer.ln_f.bias

    state_dict["unembed.W_U"] = gptj.lm_head.weight.T
    # Contains a bias, for some reason?
    state_dict["unembed.b_U"] = gptj.lm_head.bias
    return state_dict


def convert_neox_weights(neox, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = neox.gpt_neox.embed_in.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = neox.gpt_neox.layers[l].input_layernorm.weight
        state_dict[f"blocks.{l}.ln1.b"] = neox.gpt_neox.layers[l].input_layernorm.bias

        # For some inexplicable reason, NeoX both uses the concatenated QKV
        # matmul of GPT-2 (afaict this has a neglible performance impact) AND
        # has the flattened axis in the DIFFERENT order of (head_index qkv
        # d_head) - this took me an hour to debug...
        W = neox.gpt_neox.layers[l].attention.query_key_value.weight
        W = einops.rearrange(W, "(i qkv h) m->qkv i m h", i=cfg.n_heads, qkv=3)

        # Fold in layer norm weights
        state_dict[f"blocks.{l}.attn.W_Q"] = W[0]
        state_dict[f"blocks.{l}.attn.W_K"] = W[1]
        state_dict[f"blocks.{l}.attn.W_V"] = W[2]

        qkv_bias = neox.gpt_neox.layers[l].attention.query_key_value.bias
        qkv_bias = einops.rearrange(
            qkv_bias,
            "(index qkv head)->qkv index head",
            qkv=3,
            index=cfg.n_heads,
            head=cfg.d_head,
        )
        # Fold in layer norm biases
        state_dict[f"blocks.{l}.attn.b_Q"] = qkv_bias[0]
        state_dict[f"blocks.{l}.attn.b_K"] = qkv_bias[1]
        state_dict[f"blocks.{l}.attn.b_V"] = qkv_bias[2]

        W_O = neox.gpt_neox.layers[l].attention.dense.weight
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = neox.gpt_neox.layers[l].attention.dense.bias

        state_dict[f"blocks.{l}.ln2.w"] = neox.gpt_neox.layers[l].post_attention_layernorm.weight
        state_dict[f"blocks.{l}.ln2.b"] = neox.gpt_neox.layers[l].post_attention_layernorm.bias

        state_dict[f"blocks.{l}.mlp.W_in"] = neox.gpt_neox.layers[l].mlp.dense_h_to_4h.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = neox.gpt_neox.layers[l].mlp.dense_h_to_4h.bias

        state_dict[f"blocks.{l}.mlp.W_out"] = neox.gpt_neox.layers[l].mlp.dense_4h_to_h.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = neox.gpt_neox.layers[l].mlp.dense_4h_to_h.bias
    state_dict["ln_final.w"] = neox.gpt_neox.final_layer_norm.weight
    state_dict["ln_final.b"] = neox.gpt_neox.final_layer_norm.bias

    state_dict["unembed.W_U"] = neox.embed_out.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)
    return state_dict


def convert_llama_weights(llama, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = llama.model.embed_tokens.weight

    # Some models with the Llama architecture use Grouped Query Attention, and so for these we need to modify
    # the state dict keys for the K/V attention weight/biases, prepending "_" to the key names.
    using_gqa = cfg.n_key_value_heads is not None
    gqa_uscore = "_" if using_gqa else ""
    # need a cast since MyPy isn't smart enough to realize that using_gqa implies n_key_value_heads is not None
    n_kv_heads = cast(int, cfg.n_key_value_heads if using_gqa else cfg.n_heads)

    # llama has no biases anywhere and deals with everything else roughly like
    # GPTNeoX with different names

    assert cfg.d_mlp is not None  # keep mypy happy

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = llama.model.layers[l].input_layernorm.weight

        W_Q = llama.model.layers[l].self_attn.q_proj.weight
        W_K = llama.model.layers[l].self_attn.k_proj.weight
        W_V = llama.model.layers[l].self_attn.v_proj.weight

        # in case of quantization,
        # parameters should stay as bitsandbytes.nn.modules.Params4bit
        if not cfg.load_in_4bit:
            W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
            W_K = einops.rearrange(W_K, "(n h) m->n m h", n=n_kv_heads)
            W_V = einops.rearrange(W_V, "(n h) m->n m h", n=n_kv_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_K"] = W_K
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(
            cfg.n_heads, cfg.d_head, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"blocks.{l}.attn.{gqa_uscore}b_K"] = torch.zeros(
            n_kv_heads,
            cfg.d_head,
            dtype=cfg.dtype,
            device=cfg.device,
        )
        state_dict[f"blocks.{l}.attn.{gqa_uscore}b_V"] = torch.zeros(
            n_kv_heads,
            cfg.d_head,
            dtype=cfg.dtype,
            device=cfg.device,
        )

        W_O = llama.model.layers[l].self_attn.o_proj.weight

        if not cfg.load_in_4bit:
            W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_O"] = W_O.to(device=cfg.device)

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=cfg.device
        )

        state_dict[f"blocks.{l}.ln2.w"] = llama.model.layers[l].post_attention_layernorm.weight

        # in case of quantization,
        # parameters should stay as bitsandbytes.nn.modules.Params4bit
        if not cfg.load_in_4bit:
            state_dict[f"blocks.{l}.mlp.W_in"] = llama.model.layers[l].mlp.up_proj.weight.T
            state_dict[f"blocks.{l}.mlp.W_gate"] = llama.model.layers[l].mlp.gate_proj.weight.T
            state_dict[f"blocks.{l}.mlp.W_out"] = llama.model.layers[l].mlp.down_proj.weight.T
        else:
            state_dict[f"blocks.{l}.mlp.W_in"] = llama.model.layers[l].mlp.up_proj.weight
            state_dict[f"blocks.{l}.mlp.W_gate"] = llama.model.layers[l].mlp.gate_proj.weight
            state_dict[f"blocks.{l}.mlp.W_out"] = llama.model.layers[l].mlp.down_proj.weight

        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(
            cfg.d_mlp, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=cfg.device
        )

    state_dict["ln_final.w"] = llama.model.norm.weight

    state_dict["unembed.W_U"] = llama.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype, device=cfg.device)

    return state_dict


def convert_qwen_weights(qwen, cfg: HookedTransformerConfig):
    state_dict = {}
    model = qwen.transformer
    state_dict["embed.W_E"] = model.wte.weight

    assert cfg.d_mlp is not None  # keep mypy happy

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = model.h[l].ln_1.weight

        W_Q, W_K, W_V = model.h[l].attn.c_attn.weight.split(split_size=cfg.d_model, dim=0)
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        b_Q, b_K, b_V = model.h[l].attn.c_attn.bias.split(split_size=cfg.d_model, dim=0)
        b_Q = einops.rearrange(
            b_Q,
            "(n_head d_head) -> n_head d_head",
            n_head=cfg.n_heads,
        )
        b_K = einops.rearrange(
            b_K,
            "(n_head d_head) -> n_head d_head",
            n_head=cfg.n_heads,
        )
        b_V = einops.rearrange(
            b_V,
            "(n_head d_head) -> n_head d_head",
            n_head=cfg.n_heads,
        )
        state_dict[f"blocks.{l}.attn.b_Q"] = b_Q
        state_dict[f"blocks.{l}.attn.b_K"] = b_K
        state_dict[f"blocks.{l}.attn.b_V"] = b_V

        W_O = model.h[l].attn.c_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.ln2.w"] = model.h[l].ln_2.weight

        state_dict[f"blocks.{l}.mlp.W_in"] = model.h[l].mlp.w1.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = model.h[l].mlp.w2.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.mlp.W_out"] = model.h[l].mlp.c_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

    state_dict["ln_final.w"] = model.ln_f.weight

    state_dict["unembed.W_U"] = qwen.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict


def convert_qwen2_weights(qwen, cfg: HookedTransformerConfig):
    # Note that this method is also applied for Qwen1.5 models, since they
    # have architecture type Qwen2ForCausalLM.

    state_dict = {}

    state_dict["embed.W_E"] = qwen.model.embed_tokens.weight

    assert cfg.d_mlp is not None  # keep mypy happy

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = qwen.model.layers[l].input_layernorm.weight

        W_Q = qwen.model.layers[l].self_attn.q_proj.weight
        W_K = qwen.model.layers[l].self_attn.k_proj.weight
        W_V = qwen.model.layers[l].self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        b_Q = qwen.model.layers[l].self_attn.q_proj.bias
        b_Q = einops.rearrange(
            b_Q,
            "(n_head d_head) -> n_head d_head",
            n_head=cfg.n_heads,
        )

        b_K = qwen.model.layers[l].self_attn.k_proj.bias
        b_K = einops.rearrange(
            b_K,
            "(n_head d_head) -> n_head d_head",
            n_head=cfg.n_heads,
        )

        b_V = qwen.model.layers[l].self_attn.v_proj.bias
        b_V = einops.rearrange(
            b_V,
            "(n_head d_head) -> n_head d_head",
            n_head=cfg.n_heads,
        )

        state_dict[f"blocks.{l}.attn.b_Q"] = b_Q
        state_dict[f"blocks.{l}.attn.b_K"] = b_K
        state_dict[f"blocks.{l}.attn.b_V"] = b_V

        W_O = qwen.model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.ln2.w"] = qwen.model.layers[l].post_attention_layernorm.weight

        state_dict[f"blocks.{l}.mlp.W_in"] = qwen.model.layers[l].mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = qwen.model.layers[l].mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.mlp.W_out"] = qwen.model.layers[l].mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

    state_dict["ln_final.w"] = qwen.model.norm.weight

    state_dict["unembed.W_U"] = qwen.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict


def convert_mistral_weights(mistral, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = mistral.model.embed_tokens.weight

    assert cfg.n_key_value_heads is not None  # keep mypy happy
    assert cfg.d_mlp is not None  # keep mypy happy

    # Mistral has no biases anywhere
    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = mistral.model.layers[l].input_layernorm.weight

        W_Q = mistral.model.layers[l].self_attn.q_proj.weight
        W_K = mistral.model.layers[l].self_attn.k_proj.weight
        W_V = mistral.model.layers[l].self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_key_value_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_key_value_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn._W_K"] = W_K
        state_dict[f"blocks.{l}.attn._W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn._b_K"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype
        )
        state_dict[f"blocks.{l}.attn._b_V"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype
        )

        W_O = mistral.model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.ln2.w"] = mistral.model.layers[l].post_attention_layernorm.weight

        state_dict[f"blocks.{l}.mlp.W_in"] = mistral.model.layers[l].mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = mistral.model.layers[l].mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.mlp.W_out"] = mistral.model.layers[l].mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

    state_dict["ln_final.w"] = mistral.model.norm.weight

    state_dict["unembed.W_U"] = mistral.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict


def convert_mixtral_weights(mixtral, cfg: HookedTransformerConfig):
    # The same as Mistral, but with the MLP replaced with MoE
    # As with Mistral, Mixtral has no biases

    state_dict = {}

    assert cfg.n_key_value_heads is not None  # keep mypy happy
    assert cfg.d_mlp is not None
    assert cfg.num_experts is not None

    state_dict["embed.W_E"] = mixtral.model.embed_tokens.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = mixtral.model.layers[l].input_layernorm.weight

        W_Q = mixtral.model.layers[l].self_attn.q_proj.weight
        W_K = mixtral.model.layers[l].self_attn.k_proj.weight
        W_V = mixtral.model.layers[l].self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_key_value_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_key_value_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn._W_K"] = W_K
        state_dict[f"blocks.{l}.attn._W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn._b_K"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype
        )
        state_dict[f"blocks.{l}.attn._b_V"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype
        )

        W_O = mixtral.model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.ln2.w"] = mixtral.model.layers[l].post_attention_layernorm.weight

        state_dict[f"blocks.{l}.mlp.W_gate"] = mixtral.model.layers[
            l
        ].block_sparse_moe.gate.weight.T

        # The mapping here from wn to W_{in/out/gate} is a bit confusing:
        # w1 -> W_gate
        # w2 -> W_out
        # w3 -> W_in
        # See https://github.com/mistralai/mistral-src/blob/main/mistral/model.py#L128 for reference
        for e in range(cfg.num_experts):
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_in"] = (
                mixtral.model.layers[l].block_sparse_moe.experts[e].w3.weight.T
            )
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_gate"] = (
                mixtral.model.layers[l].block_sparse_moe.experts[e].w1.weight.T
            )
            state_dict[f"blocks.{l}.mlp.experts.{e}.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_out"] = (
                mixtral.model.layers[l].block_sparse_moe.experts[e].w2.weight.T
            )
            state_dict[f"blocks.{l}.mlp.experts.{e}.b_out"] = torch.zeros(
                cfg.d_model, dtype=cfg.dtype
            )

    state_dict["ln_final.w"] = mixtral.model.norm.weight.data

    state_dict["unembed.W_U"] = mixtral.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict


def convert_opt_weights(opt, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = opt.model.decoder.embed_tokens.weight
    state_dict["pos_embed.W_pos"] = opt.model.decoder.embed_positions.weight[2:, :]

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = opt.model.decoder.layers[l].self_attn_layer_norm.weight
        state_dict[f"blocks.{l}.ln1.b"] = opt.model.decoder.layers[l].self_attn_layer_norm.bias

        W_Q = opt.model.decoder.layers[l].self_attn.q_proj.weight
        W_K = opt.model.decoder.layers[l].self_attn.k_proj.weight
        W_V = opt.model.decoder.layers[l].self_attn.v_proj.weight
        W_Q = einops.rearrange(
            W_Q,
            "(index d_head) d_model->index d_model d_head",
            index=cfg.n_heads,
        )
        W_K = einops.rearrange(
            W_K,
            "(index d_head) d_model->index d_model d_head",
            index=cfg.n_heads,
        )
        W_V = einops.rearrange(
            W_V,
            "(index d_head) d_model->index d_model d_head",
            index=cfg.n_heads,
        )

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        q_bias = einops.rearrange(
            opt.model.decoder.layers[l].self_attn.q_proj.bias,
            "(head_index d_head)->head_index d_head",
            head_index=cfg.n_heads,
            d_head=cfg.d_head,
        )
        k_bias = einops.rearrange(
            opt.model.decoder.layers[l].self_attn.k_proj.bias,
            "(head_index d_head)->head_index d_head",
            head_index=cfg.n_heads,
            d_head=cfg.d_head,
        )
        v_bias = einops.rearrange(
            opt.model.decoder.layers[l].self_attn.v_proj.bias,
            "(head_index d_head)->head_index d_head",
            head_index=cfg.n_heads,
            d_head=cfg.d_head,
        )

        state_dict[f"blocks.{l}.attn.b_Q"] = q_bias
        state_dict[f"blocks.{l}.attn.b_K"] = k_bias
        state_dict[f"blocks.{l}.attn.b_V"] = v_bias

        W_O = opt.model.decoder.layers[l].self_attn.out_proj.weight
        W_O = einops.rearrange(
            W_O,
            "d_model (index d_head)->index d_head d_model",
            index=cfg.n_heads,
        )
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = opt.model.decoder.layers[l].self_attn.out_proj.bias

        state_dict[f"blocks.{l}.ln2.w"] = opt.model.decoder.layers[l].final_layer_norm.weight
        state_dict[f"blocks.{l}.ln2.b"] = opt.model.decoder.layers[l].final_layer_norm.bias

        state_dict[f"blocks.{l}.mlp.W_in"] = opt.model.decoder.layers[l].fc1.weight.T
        state_dict[f"blocks.{l}.mlp.W_out"] = opt.model.decoder.layers[l].fc2.weight.T

        state_dict[f"blocks.{l}.mlp.b_in"] = opt.model.decoder.layers[l].fc1.bias
        state_dict[f"blocks.{l}.mlp.b_out"] = opt.model.decoder.layers[l].fc2.bias
    state_dict["ln_final.w"] = opt.model.decoder.final_layer_norm.weight
    state_dict["ln_final.b"] = opt.model.decoder.final_layer_norm.bias
    state_dict["unembed.W_U"] = opt.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)
    return state_dict


def convert_neel_solu_old_weights(state_dict: dict, cfg: HookedTransformerConfig):
    """
    Converts the weights of my old SoLU models to the HookedTransformer format.
    Takes as input a state dict, *not* a model object.

    There are a bunch of dumb bugs in the original code, sorry!

    Models 1L, 2L, 4L and 6L have left facing weights (ie, weights have shape
    [dim_out, dim_in]) while HookedTransformer does right facing (ie [dim_in,
    dim_out]).

    8L has *just* a left facing W_pos, the rest right facing.

    And some models were trained with
    """
    # Early models have left facing W_pos
    reverse_pos = cfg.n_layers <= 8

    # Models prior to 8L have left facing everything (8L has JUST left facing W_pos - sorry! Stupid bug)
    reverse_weights = cfg.n_layers <= 6

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("norm", "ln")
        if k.startswith("ln."):
            k = k.replace("ln.", "ln_final.")
        new_state_dict[k] = v

    if reverse_pos:
        new_state_dict["pos_embed.W_pos"] = new_state_dict["pos_embed.W_pos"].T
    if reverse_weights:
        for k, v in new_state_dict.items():
            if "W_" in k and "W_pos" not in k:
                new_state_dict[k] = v.transpose(-2, -1)
    return new_state_dict


def convert_mingpt_weights(old_state_dict, cfg: HookedTransformerConfig):
    # mingpt (https://github.com/karpathy/minGPT) is mostly similar to GPT-2,
    # but doesn't concat the QKV matrices.
    state_dict = {}

    state_dict["embed.W_E"] = old_state_dict["tok_emb.weight"]
    state_dict["pos_embed.W_pos"] = old_state_dict["pos_emb"].squeeze()

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = old_state_dict[f"blocks.{l}.ln1.weight"]
        state_dict[f"blocks.{l}.ln1.b"] = old_state_dict[f"blocks.{l}.ln1.bias"]

        W_Q = old_state_dict[f"blocks.{l}.attn.query.weight"]
        W_K = old_state_dict[f"blocks.{l}.attn.key.weight"]
        W_V = old_state_dict[f"blocks.{l}.attn.value.weight"]
        W_Q = einops.rearrange(W_Q, "(i h) m->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(i h) m->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(i h) m->i m h", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        q_bias = einops.rearrange(
            old_state_dict[f"blocks.{l}.attn.query.bias"], "(i h)->i h", i=cfg.n_heads
        )
        k_bias = einops.rearrange(
            old_state_dict[f"blocks.{l}.attn.key.bias"], "(i h)->i h", i=cfg.n_heads
        )
        v_bias = einops.rearrange(
            old_state_dict[f"blocks.{l}.attn.value.bias"], "(i h)->i h", i=cfg.n_heads
        )

        state_dict[f"blocks.{l}.attn.b_Q"] = q_bias
        state_dict[f"blocks.{l}.attn.b_K"] = k_bias
        state_dict[f"blocks.{l}.attn.b_V"] = v_bias

        W_O = old_state_dict[f"blocks.{l}.attn.proj.weight"]
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = old_state_dict[f"blocks.{l}.attn.proj.bias"]

        state_dict[f"blocks.{l}.ln2.w"] = old_state_dict[f"blocks.{l}.ln2.weight"]
        state_dict[f"blocks.{l}.ln2.b"] = old_state_dict[f"blocks.{l}.ln2.bias"]

        W_in = old_state_dict[f"blocks.{l}.mlp.0.weight"]
        state_dict[f"blocks.{l}.mlp.W_in"] = W_in.T
        state_dict[f"blocks.{l}.mlp.b_in"] = old_state_dict[f"blocks.{l}.mlp.0.bias"]

        W_out = old_state_dict[f"blocks.{l}.mlp.2.weight"]
        state_dict[f"blocks.{l}.mlp.W_out"] = W_out.T
        state_dict[f"blocks.{l}.mlp.b_out"] = old_state_dict[f"blocks.{l}.mlp.2.bias"]

    state_dict["unembed.W_U"] = old_state_dict["head.weight"].T

    state_dict["ln_final.w"] = old_state_dict["ln_f.weight"]
    state_dict["ln_final.b"] = old_state_dict["ln_f.bias"]

    return state_dict


def convert_nanogpt_weights(old_state_dict, cfg: HookedTransformerConfig):
    """For https://github.com/karpathy/nanoGPT
    There are two complications with converting nanogpt models:
    The first is that some state dicts have an unwanted prefix on keys that needs to be removed.
    The second is that the models can be saved with or without bias. By default, there
    is no bias. This function can handle both cases."""
    # Nanogpt models saved after torch.compile() have this unwanted prefix
    # This is a simple way to remove it
    unwanted_prefix = "_orig_mod."
    for k, v in list(old_state_dict.items()):
        if k.startswith(unwanted_prefix):
            old_state_dict[k[len(unwanted_prefix) :]] = old_state_dict.pop(k)

    new_state_dict = {}
    new_state_dict["pos_embed.W_pos"] = old_state_dict["transformer.wpe.weight"]
    new_state_dict["embed.W_E"] = old_state_dict["transformer.wte.weight"]

    new_state_dict["ln_final.w"] = old_state_dict["transformer.ln_f.weight"]
    new_state_dict["ln_final.b"] = torch.zeros_like(old_state_dict["transformer.ln_f.weight"])
    new_state_dict["unembed.W_U"] = old_state_dict["lm_head.weight"].T

    bias = False
    if "transformer.ln_f.bias" in old_state_dict:
        bias = True
        new_state_dict["ln_final.b"] = old_state_dict["transformer.ln_f.bias"]

    for layer in range(cfg.n_layers):
        layer_key = f"transformer.h.{layer}"

        new_state_dict[f"blocks.{layer}.ln1.w"] = old_state_dict[f"{layer_key}.ln_1.weight"]
        # A bias of zeros is required for folding layer norm
        new_state_dict[f"blocks.{layer}.ln1.b"] = torch.zeros_like(
            old_state_dict[f"{layer_key}.ln_1.weight"]
        )
        new_state_dict[f"blocks.{layer}.ln2.w"] = old_state_dict[f"{layer_key}.ln_2.weight"]
        new_state_dict[f"blocks.{layer}.ln2.b"] = torch.zeros_like(
            old_state_dict[f"{layer_key}.ln_2.weight"]
        )

        W = old_state_dict[f"{layer_key}.attn.c_attn.weight"]
        W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=0)
        W_Q = einops.rearrange(W_Q, "(i h) m->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(i h) m->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(i h) m->i m h", i=cfg.n_heads)
        new_state_dict[f"blocks.{layer}.attn.W_Q"] = W_Q
        new_state_dict[f"blocks.{layer}.attn.W_K"] = W_K
        new_state_dict[f"blocks.{layer}.attn.W_V"] = W_V

        W_O = old_state_dict[f"{layer_key}.attn.c_proj.weight"]
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        new_state_dict[f"blocks.{layer}.attn.W_O"] = W_O

        new_state_dict[f"blocks.{layer}.mlp.W_in"] = old_state_dict[
            f"{layer_key}.mlp.c_fc.weight"
        ].T
        new_state_dict[f"blocks.{layer}.mlp.W_out"] = old_state_dict[
            f"{layer_key}.mlp.c_proj.weight"
        ].T

        if bias:
            new_state_dict[f"blocks.{layer}.ln1.b"] = old_state_dict[f"{layer_key}.ln_1.bias"]
            new_state_dict[f"blocks.{layer}.ln2.b"] = old_state_dict[f"{layer_key}.ln_2.bias"]
            new_state_dict[f"blocks.{layer}.mlp.b_in"] = old_state_dict[
                f"{layer_key}.mlp.c_fc.bias"
            ]
            new_state_dict[f"blocks.{layer}.mlp.b_out"] = old_state_dict[
                f"{layer_key}.mlp.c_proj.bias"
            ]

            B = old_state_dict[f"{layer_key}.attn.c_attn.bias"]
            B_Q, B_K, B_V = torch.tensor_split(B, 3, dim=0)
            B_Q = einops.rearrange(B_Q, "(i h)->i h", i=cfg.n_heads)
            B_K = einops.rearrange(B_K, "(i h)->i h", i=cfg.n_heads)
            B_V = einops.rearrange(B_V, "(i h)->i h", i=cfg.n_heads)
            new_state_dict[f"blocks.{layer}.attn.b_Q"] = B_Q
            new_state_dict[f"blocks.{layer}.attn.b_K"] = B_K
            new_state_dict[f"blocks.{layer}.attn.b_V"] = B_V
            new_state_dict[f"blocks.{layer}.attn.b_O"] = old_state_dict[
                f"{layer_key}.attn.c_proj.bias"
            ]

    return new_state_dict


def convert_bert_weights(bert, cfg: HookedTransformerConfig):
    embeddings = bert.bert.embeddings
    state_dict = {
        "embed.embed.W_E": embeddings.word_embeddings.weight,
        "embed.pos_embed.W_pos": embeddings.position_embeddings.weight,
        "embed.token_type_embed.W_token_type": embeddings.token_type_embeddings.weight,
        "embed.ln.w": embeddings.LayerNorm.weight,
        "embed.ln.b": embeddings.LayerNorm.bias,
    }

    for l in range(cfg.n_layers):
        block = bert.bert.encoder.layer[l]
        state_dict[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
            block.attention.self.query.weight, "(i h) m -> i m h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
            block.attention.self.query.bias, "(i h) -> i h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_K"] = einops.rearrange(
            block.attention.self.key.weight, "(i h) m -> i m h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.b_K"] = einops.rearrange(
            block.attention.self.key.bias, "(i h) -> i h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_V"] = einops.rearrange(
            block.attention.self.value.weight, "(i h) m -> i m h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.b_V"] = einops.rearrange(
            block.attention.self.value.bias, "(i h) -> i h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_O"] = einops.rearrange(
            block.attention.output.dense.weight,
            "m (i h) -> i h m",
            i=cfg.n_heads,
        )
        state_dict[f"blocks.{l}.attn.b_O"] = block.attention.output.dense.bias
        state_dict[f"blocks.{l}.ln1.w"] = block.attention.output.LayerNorm.weight
        state_dict[f"blocks.{l}.ln1.b"] = block.attention.output.LayerNorm.bias
        state_dict[f"blocks.{l}.mlp.W_in"] = einops.rearrange(
            block.intermediate.dense.weight, "mlp model -> model mlp"
        )
        state_dict[f"blocks.{l}.mlp.b_in"] = block.intermediate.dense.bias
        state_dict[f"blocks.{l}.mlp.W_out"] = einops.rearrange(
            block.output.dense.weight, "model mlp -> mlp model"
        )
        state_dict[f"blocks.{l}.mlp.b_out"] = block.output.dense.bias
        state_dict[f"blocks.{l}.ln2.w"] = block.output.LayerNorm.weight
        state_dict[f"blocks.{l}.ln2.b"] = block.output.LayerNorm.bias

    mlm_head = bert.cls.predictions
    state_dict["mlm_head.W"] = mlm_head.transform.dense.weight
    state_dict["mlm_head.b"] = mlm_head.transform.dense.bias
    state_dict["mlm_head.ln.w"] = mlm_head.transform.LayerNorm.weight
    state_dict["mlm_head.ln.b"] = mlm_head.transform.LayerNorm.bias
    # Note: BERT uses tied embeddings
    state_dict["unembed.W_U"] = embeddings.word_embeddings.weight.T
    # "unembed.W_U": mlm_head.decoder.weight.T,
    state_dict["unembed.b_U"] = mlm_head.bias

    return state_dict


def convert_bloom_weights(bloom, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = bloom.transformer.word_embeddings.weight

    # Bloom uses post embedding layer norm
    state_dict["embed.ln.w"] = bloom.transformer.word_embeddings_layernorm.weight
    state_dict["embed.ln.b"] = bloom.transformer.word_embeddings_layernorm.bias

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = bloom.transformer.h[l].input_layernorm.weight
        state_dict[f"blocks.{l}.ln1.b"] = bloom.transformer.h[l].input_layernorm.bias

        W = bloom.transformer.h[l].self_attention.query_key_value.weight

        W_split = W.T.reshape(cfg.d_model, cfg.n_heads, 3, cfg.d_head)

        W_Q, W_K, W_V = W_split[..., 0, :], W_split[..., 1, :], W_split[..., 2, :]
        W_Q = einops.rearrange(W_Q, "m n h ->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "m n h ->n m h", n=cfg.n_heads)
        W_V = einops.rearrange(W_V, "m n h ->n m h", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        qkv_bias = bloom.transformer.h[l].self_attention.query_key_value.bias
        qkv_bias = qkv_bias.reshape(cfg.n_heads, 3, cfg.d_head)

        state_dict[f"blocks.{l}.attn.b_Q"] = qkv_bias[:, 0, :]
        state_dict[f"blocks.{l}.attn.b_K"] = qkv_bias[:, 1, :]
        state_dict[f"blocks.{l}.attn.b_V"] = qkv_bias[:, 2, :]

        W_O = bloom.transformer.h[l].self_attention.dense.weight.T  # [1024, 1024]
        W_O = einops.rearrange(W_O, "(n h) m->n h m", n=cfg.n_heads)  # [n_heads, d_head, d_model]
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = bloom.transformer.h[l].self_attention.dense.bias

        state_dict[f"blocks.{l}.ln2.w"] = bloom.transformer.h[l].post_attention_layernorm.weight
        state_dict[f"blocks.{l}.ln2.b"] = bloom.transformer.h[l].post_attention_layernorm.bias

        W_in = bloom.transformer.h[l].mlp.dense_h_to_4h.weight.T
        state_dict[f"blocks.{l}.mlp.W_in"] = W_in
        state_dict[f"blocks.{l}.mlp.b_in"] = bloom.transformer.h[l].mlp.dense_h_to_4h.bias

        W_out = bloom.transformer.h[l].mlp.dense_4h_to_h.weight.T
        state_dict[f"blocks.{l}.mlp.W_out"] = W_out
        state_dict[f"blocks.{l}.mlp.b_out"] = bloom.transformer.h[l].mlp.dense_4h_to_h.bias
    state_dict["unembed.W_U"] = bloom.lm_head.weight.T

    state_dict["ln_final.w"] = bloom.transformer.ln_f.weight
    state_dict["ln_final.b"] = bloom.transformer.ln_f.bias
    return state_dict


def convert_coder_weights(model, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = model.transformer.wte.weight
    state_dict["pos_embed.W_pos"] = model.transformer.wpe.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = model.transformer.h[l].ln_1.weight
        state_dict[f"blocks.{l}.ln1.b"] = model.transformer.h[l].ln_1.bias

        # In GPT-2, q,k,v are produced by one big linear map, whose output is
        # concat([q, k, v])
        W_KV = model.transformer.h[l].attn.kv_attn.weight  # [d_model, 2 * d_head]
        W_K, W_V = torch.tensor_split(W_KV, 2, dim=1)
        W_Q = model.transformer.h[l].attn.q_attn.weight  # [d_model, d_model]
        W_Q = einops.rearrange(W_Q, "m (i h)->i m h", i=cfg.n_heads)
        W_K = einops.repeat(W_K, "m h -> i m h", i=cfg.n_heads)
        W_V = einops.repeat(W_V, "m h -> i m h", i=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        b_Q = einops.rearrange(
            model.transformer.h[l].attn.q_attn.bias,
            "(index head)-> index head",
            index=cfg.n_heads,
            head=cfg.d_head,
        )
        b_KV = model.transformer.h[l].attn.kv_attn.bias  # [2 * d_head]
        b_K, b_V = torch.tensor_split(b_KV, 2, dim=0)
        b_K = einops.repeat(b_K, "head -> index head", index=cfg.n_heads)
        b_V = einops.repeat(b_V, "head -> index head", index=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.b_Q"] = b_Q
        state_dict[f"blocks.{l}.attn.b_K"] = b_K
        state_dict[f"blocks.{l}.attn.b_V"] = b_V

        W_O = model.transformer.h[l].attn.c_proj.weight
        W_O = einops.rearrange(W_O, "(i h) m->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = model.transformer.h[l].attn.c_proj.bias

        state_dict[f"blocks.{l}.ln2.w"] = model.transformer.h[l].ln_2.weight
        state_dict[f"blocks.{l}.ln2.b"] = model.transformer.h[l].ln_2.bias

        W_in = model.transformer.h[l].mlp.c_fc.weight
        state_dict[f"blocks.{l}.mlp.W_in"] = W_in
        state_dict[f"blocks.{l}.mlp.b_in"] = model.transformer.h[l].mlp.c_fc.bias

        W_out = model.transformer.h[l].mlp.c_proj.weight
        state_dict[f"blocks.{l}.mlp.W_out"] = W_out
        state_dict[f"blocks.{l}.mlp.b_out"] = model.transformer.h[l].mlp.c_proj.bias
    state_dict["unembed.W_U"] = model.lm_head.weight.T

    state_dict["ln_final.w"] = model.transformer.ln_f.weight
    state_dict["ln_final.b"] = model.transformer.ln_f.bias
    return state_dict


def convert_phi_weights(phi, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = phi.model.embed_tokens.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = phi.model.layers[l].input_layernorm.weight
        state_dict[f"blocks.{l}.ln1.b"] = phi.model.layers[l].input_layernorm.bias

        W_Q = phi.model.layers[l].self_attn.q_proj.weight
        W_K = phi.model.layers[l].self_attn.k_proj.weight
        W_V = phi.model.layers[l].self_attn.v_proj.weight
        W_Q = einops.rearrange(
            W_Q, "(n_head d_head) d_model -> n_head d_model d_head", n_head=cfg.n_heads
        )
        W_K = einops.rearrange(
            W_K, "(n_head d_head) d_model  -> n_head d_model d_head", n_head=cfg.n_heads
        )
        W_V = einops.rearrange(
            W_V, "(n_head d_head) d_model  -> n_head d_model d_head", n_head=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        b_Q = phi.model.layers[l].self_attn.q_proj.bias
        b_K = phi.model.layers[l].self_attn.k_proj.bias
        b_V = phi.model.layers[l].self_attn.v_proj.bias
        b_Q = einops.rearrange(b_Q, "(n_head d_head) -> n_head d_head", n_head=cfg.n_heads)
        b_K = einops.rearrange(b_K, "(n_head d_head) -> n_head d_head", n_head=cfg.n_heads)
        b_V = einops.rearrange(b_V, "(n_head d_head) -> n_head d_head", n_head=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.b_Q"] = b_Q
        state_dict[f"blocks.{l}.attn.b_K"] = b_K
        state_dict[f"blocks.{l}.attn.b_V"] = b_V

        W_O = phi.model.layers[l].self_attn.dense.weight
        W_O = einops.rearrange(
            W_O, "d_model (n_head d_head) -> n_head d_head d_model", n_head=cfg.n_heads
        )

        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = phi.model.layers[l].self_attn.dense.bias

        # Layer Norm 1 and 2 are tied.
        state_dict[f"blocks.{l}.ln2.w"] = state_dict[f"blocks.{l}.ln1.w"]
        state_dict[f"blocks.{l}.ln2.b"] = state_dict[f"blocks.{l}.ln1.b"]

        state_dict[f"blocks.{l}.mlp.W_in"] = phi.model.layers[l].mlp.fc1.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = phi.model.layers[l].mlp.fc1.bias
        state_dict[f"blocks.{l}.mlp.W_out"] = phi.model.layers[l].mlp.fc2.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = phi.model.layers[l].mlp.fc2.bias

    state_dict["ln_final.w"] = phi.model.final_layernorm.weight
    state_dict["ln_final.b"] = phi.model.final_layernorm.bias

    state_dict["unembed.W_U"] = phi.lm_head.weight.T
    state_dict["unembed.b_U"] = phi.lm_head.bias

    return state_dict


def convert_phi3_weights(phi, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = phi.model.embed_tokens.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = phi.model.layers[l].input_layernorm.weight
        state_dict[f"blocks.{l}.ln1.b"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

        W = phi.model.layers[l].self_attn.qkv_proj.weight
        W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=0)
        W_Q = einops.rearrange(
            W_Q, "(n_head d_head) d_model -> n_head d_model d_head", n_head=cfg.n_heads
        )
        W_K = einops.rearrange(
            W_K, "(n_head d_head) d_model  -> n_head d_model d_head", n_head=cfg.n_heads
        )
        W_V = einops.rearrange(
            W_V, "(n_head d_head) d_model  -> n_head d_model d_head", n_head=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.b_K"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.W_V"] = W_V
        state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)

        W_O = phi.model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(
            W_O, "d_model (n_head d_head) -> n_head d_head d_model", n_head=cfg.n_heads
        )

        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.ln2.w"] = phi.model.layers[l].post_attention_layernorm.weight
        state_dict[f"blocks.{l}.ln2.b"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

        W = phi.model.layers[l].mlp.gate_up_proj.weight.T
        W_gate, W_in = torch.tensor_split(W, 2, dim=1)
        state_dict[f"blocks.{l}.mlp.W_in"] = W_in
        state_dict[f"blocks.{l}.mlp.W_gate"] = W_gate
        state_dict[f"blocks.{l}.mlp.W_out"] = phi.model.layers[l].mlp.down_proj.weight.T

    state_dict["ln_final.w"] = phi.model.norm.weight

    state_dict["unembed.W_U"] = phi.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict


def convert_gemma_weights(gemma, cfg: HookedTransformerConfig):
    state_dict = {}

    assert cfg.n_key_value_heads is not None  # keep mypy happy
    assert cfg.d_mlp is not None  # keep mypy happy

    # Gemma Models scale embeddings by multiplying by sqrt(d_model), use hidden state type to match
    # HF implementation
    state_dict["embed.W_E"] = gemma.model.embed_tokens.weight * torch.tensor(
        cfg.d_model**0.5, dtype=cfg.dtype
    )

    # Gemma has no biases anywhere
    for l in range(cfg.n_layers):
        # GemmaRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
        state_dict[f"blocks.{l}.ln1.w"] = gemma.model.layers[
            l
        ].input_layernorm.weight.float() + torch.ones_like(
            gemma.model.layers[l].input_layernorm.weight, dtype=torch.float32
        )

        W_Q = gemma.model.layers[l].self_attn.q_proj.weight
        W_K = gemma.model.layers[l].self_attn.k_proj.weight
        W_V = gemma.model.layers[l].self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_key_value_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_key_value_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn._W_K"] = W_K
        state_dict[f"blocks.{l}.attn._W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn._b_K"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype
        )
        state_dict[f"blocks.{l}.attn._b_V"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype
        )

        W_O = gemma.model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        # GemmaRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
        state_dict[f"blocks.{l}.ln2.w"] = gemma.model.layers[
            l
        ].post_attention_layernorm.weight.float() + torch.ones_like(
            gemma.model.norm.weight, dtype=torch.float32
        )

        state_dict[f"blocks.{l}.mlp.W_in"] = gemma.model.layers[l].mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = gemma.model.layers[l].mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.mlp.W_out"] = gemma.model.layers[l].mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

    # GemmaRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
    state_dict["ln_final.w"] = gemma.model.norm.weight.float() + torch.ones_like(
        gemma.model.norm.weight, dtype=torch.float32
    )

    state_dict["unembed.W_U"] = gemma.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict


@dataclasses.dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


# Returns the configuration parameters of the model as a basic Config dataclass
def get_basic_config(model_name: str, **kwargs) -> Config:
    return Config(
        **{
            k: v
            for k, v in get_pretrained_model_config(model_name, **kwargs).to_dict().items()
            if k
            in [
                "d_model",
                "debug",
                "layer_norm_eps",
                "d_vocab",
                "init_range",
                "n_ctx",
                "d_head",
                "d_mlp",
                "n_heads",
                "n_layers",
            ]
        }
    )
