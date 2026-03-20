import argparse
import gc
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from datasets import Dataset

from lm_saes import (
    ActivationFactoryActivationsSource,
    ActivationFactoryConfig,
    ActivationFactoryTarget,
    DatasetConfig,
    GenerateActivationsSettings,
    InitializerConfig,
    LanguageModelConfig,
    LorsaConfig,
    ReplacementModel,
    SAEConfig,
    TrainerConfig,
    TrainLorsaSettings,
    TrainSAESettings,
    generate_activations,
    train_lorsa,
    train_sae,
)
from lm_saes.circuit.attribution import attribute
from lm_saes.circuit.utils.transcoder_set import TranscoderSet, TranscoderSetConfig
from lm_saes.evaluator import compute_graph_scores
from lm_saes.models.lorsa import LowRankSparseAttention
from lm_saes.models.sae import SparseAutoEncoder
from lm_saes.resource_loaders import load_model


CORPUS = [
    "The Eiffel Tower is in Paris and it was completed in 1889.",
    "Python is a programming language focused on readability and batteries included.",
    "Attention lets transformers move information across token positions.",
    "Sparse autoencoders can reveal interpretable features in language models.",
    "Mechanistic interpretability studies circuits inside neural networks.",
    "The Pacific Ocean is larger than the Atlantic Ocean.",
    "Mount Everest is the highest mountain above sea level.",
    "The Moon orbits the Earth roughly once every twenty seven days.",
    "Gradient descent updates parameters in the direction that reduces loss.",
    "Low rank structure can hide useful sparse features in neural activations.",
    "The capital of Japan is Tokyo.",
    "An attention head can attend from one token to another token.",
]


@dataclass
class ModelSpec:
    model_name: str
    device: str
    n_layers: int
    d_model: int
    n_heads: int
    d_head: int
    n_ctx: int
    rotary_dim: int
    rotary_base: int
    rotary_adjacent_pairs: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=Path("tmp/crm_pythia70m"))
    parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--parallel-devices", nargs="+", default=None)
    parser.add_argument("--context-size", type=int, default=64)
    parser.add_argument("--activation-total-tokens", type=int, default=4096)
    parser.add_argument("--transcoder-training-tokens", type=int, default=1024)
    parser.add_argument("--lorsa-training-tokens", type=int, default=512)
    parser.add_argument("--transcoder-expansion", type=float, default=2.0)
    parser.add_argument("--lorsa-expansion", type=float, default=2.0)
    parser.add_argument("--transcoder-top-k", type=int, default=16)
    parser.add_argument("--lorsa-top-k", type=int, default=32)
    parser.add_argument("--activation-writer-chunk-size", type=int, default=16)
    parser.add_argument("--eval-prompt", type=str, default="The capital of France is")
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--max-feature-nodes", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_set_cuda_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_dataset(dataset_dir: Path, force: bool) -> Path:
    if force and dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    if dataset_dir.exists():
        return dataset_dir

    texts = CORPUS * 64
    dataset = Dataset.from_dict({"text": texts})
    dataset.save_to_disk(str(dataset_dir))
    return dataset_dir


def resolve_saved_model_dir(result_dir: Path) -> Path:
    if (result_dir / "config.json").exists() and (result_dir / "sae_weights.safetensors").exists():
        return result_dir

    checkpoint_root = result_dir / "checkpoints"
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"No saved SAE found under {result_dir}")

    checkpoint_dirs = sorted(
        (path for path in checkpoint_root.glob("step_*") if path.is_dir()),
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found under {checkpoint_root}")
    return checkpoint_dirs[-1]


def derive_model_spec(model_cfg: LanguageModelConfig) -> ModelSpec:
    language_model = load_model(model_cfg)
    model = language_model.model
    assert model is not None
    spec = ModelSpec(
        model_name=model_cfg.model_name,
        device=model_cfg.device,
        n_layers=model.cfg.n_layers,
        d_model=model.cfg.d_model,
        n_heads=model.cfg.n_heads,
        d_head=model.cfg.d_head,
        n_ctx=model.cfg.n_ctx,
        rotary_dim=model.cfg.rotary_dim,
        rotary_base=model.cfg.rotary_base,
        rotary_adjacent_pairs=model.cfg.rotary_adjacent_pairs,
    )
    del language_model
    cleanup_cuda()
    return spec


def all_hook_points(spec: ModelSpec) -> list[str]:
    hook_points: list[str] = []
    for layer in range(spec.n_layers):
        hook_points.extend(
            [
                f"blocks.{layer}.ln1.hook_normalized",
                f"blocks.{layer}.hook_attn_out",
                f"blocks.{layer}.ln2.hook_normalized",
                f"blocks.{layer}.hook_mlp_out",
            ]
        )
    return hook_points


def ensure_activation_cache(
    args: argparse.Namespace,
    model_cfg: LanguageModelConfig,
    spec: ModelSpec,
    dataset_dir: Path,
    activation_dir: Path,
) -> None:
    if args.force and activation_dir.exists():
        shutil.rmtree(activation_dir)
    if activation_dir.exists():
        return

    settings = GenerateActivationsSettings(
        model=model_cfg,
        model_name="pythia-70m-crm",
        dataset=DatasetConfig(dataset_name_or_path=str(dataset_dir), is_dataset_on_disk=True),
        dataset_name="crm-local-corpus",
        hook_points=all_hook_points(spec),
        output_dir=str(activation_dir),
        target=ActivationFactoryTarget.ACTIVATIONS_2D,
        model_batch_size=4,
        batch_size=4,
        total_tokens=args.activation_total_tokens,
        context_size=args.context_size,
        n_samples_per_chunk=args.activation_writer_chunk_size,
        device_type="cuda" if args.device.startswith("cuda") else "cpu",
    )
    generate_activations(settings)
    cleanup_cuda()


def train_transcoder_layer(
    layer: int,
    args: argparse.Namespace,
    spec: ModelSpec,
    model_cfg: LanguageModelConfig,
    activation_dir: Path,
    result_dir: Path,
) -> Path:
    if args.force and result_dir.exists():
        shutil.rmtree(result_dir)
    if result_dir.exists():
        try:
            return resolve_saved_model_dir(result_dir)
        except FileNotFoundError:
            shutil.rmtree(result_dir)

    settings = TrainSAESettings(
        sae=SAEConfig(
            hook_point_in=f"blocks.{layer}.ln2.hook_normalized",
            hook_point_out=f"blocks.{layer}.hook_mlp_out",
            d_model=spec.d_model,
            expansion_factor=args.transcoder_expansion,
            act_fn="topk",
            top_k=args.transcoder_top_k,
            dtype=torch.float32,
            device=args.device,
        ),
        sae_name=f"pythia-70m-crm-transcoder-l{layer}",
        sae_series="pythia-70m-crm",
        initializer=InitializerConfig(
            bias_init_method="geometric_median",
            init_encoder_bias_with_mean_hidden_pre=True,
            init_encoder_with_decoder_transpose=False,
            grid_search_init_norm=True,
            initialize_tc_with_mlp=True,
            model_layer=layer,
        ),
        trainer=TrainerConfig(
            amp_dtype=torch.float32,
            lr=1e-4,
            total_training_tokens=args.transcoder_training_tokens,
            initial_k=max(args.transcoder_top_k * 2, args.transcoder_top_k),
            k_warmup_steps=0.1,
            l1_coefficient_warmup_steps=1,
            log_frequency=2,
            eval_frequency=1_000_000,
            n_checkpoints=0,
            check_point_save_mode="linear",
            exp_result_path=str(result_dir),
        ),
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryActivationsSource(
                    path=str(activation_dir),
                    name="crm-activations",
                    device=args.device,
                    dtype=torch.float32,
                )
            ],
            target=ActivationFactoryTarget.ACTIVATIONS_1D,
            hook_points=[
                f"blocks.{layer}.ln2.hook_normalized",
                f"blocks.{layer}.hook_mlp_out",
            ],
            batch_size=256,
            buffer_size=None,
        ),
        model=model_cfg,
        model_name="pythia-70m-crm",
    )
    train_sae(settings)
    cleanup_cuda()
    return resolve_saved_model_dir(result_dir)


def train_lorsa_layer(
    layer: int,
    args: argparse.Namespace,
    spec: ModelSpec,
    model_cfg: LanguageModelConfig,
    activation_dir: Path,
    result_dir: Path,
) -> Path:
    if args.force and result_dir.exists():
        shutil.rmtree(result_dir)
    if result_dir.exists():
        try:
            return resolve_saved_model_dir(result_dir)
        except FileNotFoundError:
            shutil.rmtree(result_dir)

    settings = TrainLorsaSettings(
        sae=LorsaConfig(
            hook_point_in=f"blocks.{layer}.ln1.hook_normalized",
            hook_point_out=f"blocks.{layer}.hook_attn_out",
            d_model=spec.d_model,
            expansion_factor=args.lorsa_expansion,
            n_qk_heads=spec.n_heads,
            d_qk_head=spec.d_head,
            positional_embedding_type="rotary",
            rotary_dim=spec.rotary_dim,
            rotary_base=spec.rotary_base,
            rotary_adjacent_pairs=spec.rotary_adjacent_pairs,
            n_ctx=args.context_size,
            act_fn="topk",
            top_k=args.lorsa_top_k,
            dtype=torch.float32,
            device=args.device,
            use_post_qk_ln=False,
        ),
        sae_name=f"pythia-70m-crm-lorsa-l{layer}",
        sae_series="pythia-70m-crm",
        initializer=InitializerConfig(
            init_encoder_with_decoder_transpose=False,
            grid_search_init_norm=True,
            initialize_lorsa_with_mhsa=True,
            initialize_W_D_with_active_subspace=True,
            model_layer=layer,
        ),
        trainer=TrainerConfig(
            amp_dtype=torch.float32,
            lr=2e-4,
            total_training_tokens=args.lorsa_training_tokens,
            initial_k=args.lorsa_top_k,
            k_warmup_steps=0,
            l1_coefficient_warmup_steps=1,
            log_frequency=2,
            eval_frequency=1_000_000,
            n_checkpoints=0,
            check_point_save_mode="linear",
            exp_result_path=str(result_dir),
        ),
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryActivationsSource(
                    path=str(activation_dir),
                    name="crm-activations",
                    device=args.device,
                    dtype=torch.float32,
                )
            ],
            target=ActivationFactoryTarget.ACTIVATIONS_2D,
            hook_points=[
                f"blocks.{layer}.ln1.hook_normalized",
                f"blocks.{layer}.hook_attn_out",
            ],
            batch_size=4,
            buffer_size=None,
        ),
        model=model_cfg,
        model_name="pythia-70m-crm",
    )
    train_lorsa(settings)
    cleanup_cuda()
    return resolve_saved_model_dir(result_dir)


def load_transcoder_set(transcoder_dirs: list[Path], device: str) -> TranscoderSet:
    transcoders = {
        layer: SparseAutoEncoder.from_pretrained(str(path), device=device, dtype=torch.float32)
        for layer, path in enumerate(transcoder_dirs)
    }
    first = transcoders[0]
    config = TranscoderSetConfig(
        n_layers=len(transcoders),
        d_sae=first.cfg.d_sae,
        feature_input_hook="ln2.hook_normalized",
        feature_output_hook="hook_mlp_out",
    )
    return TranscoderSet(config, transcoders)


def load_lorsas(lorsa_dirs: list[Path], device: str) -> list[LowRankSparseAttention]:
    return [LowRankSparseAttention.from_pretrained(str(path), device=device, dtype=torch.float32) for path in lorsa_dirs]


def build_replacement_model(
    model_cfg: LanguageModelConfig,
    transcoder_dirs: list[Path],
    lorsa_dirs: list[Path],
) -> ReplacementModel:
    transcoders = load_transcoder_set(transcoder_dirs, model_cfg.device)
    lorsas = load_lorsas(lorsa_dirs, model_cfg.device)
    model = ReplacementModel.from_pretrained(model_cfg, transcoders, lorsas, use_lorsa=True)
    model.eval()
    return model


def run_eval(
    replacement_model: ReplacementModel,
    prompt: str,
    eval_batch_size: int,
    max_feature_nodes: int,
    parallel_devices: list[str] | None,
) -> dict[str, object]:
    graph = attribute(
        prompt=prompt,
        model=replacement_model,
        max_n_logits=2,
        desired_logit_prob=0.95,
        batch_size=eval_batch_size,
        max_feature_nodes=max_feature_nodes,
        offload=None,
        use_lorsa=True,
        parallel_devices=parallel_devices,
    )
    replacement_score, completeness_score = compute_graph_scores(graph, use_lorsa=True)
    logit_tokens = graph.logit_tokens.tolist() if isinstance(graph.logit_tokens, torch.Tensor) else graph.logit_tokens
    return {
        "prompt": prompt,
        "replacement_score": float(replacement_score),
        "completeness_score": float(completeness_score),
        "n_selected_features": len(graph.selected_features),
        "n_logits": len(graph.logit_tokens),
        "logit_tokens": [int(token) for token in logit_tokens],
    }


def main() -> None:
    args = parse_args()
    set_seed()
    primary_device = args.parallel_devices[0] if args.parallel_devices else args.device
    maybe_set_cuda_device(primary_device)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    workdir = args.workdir
    dataset_dir = workdir / "dataset"
    activation_dir = workdir / "activations"
    transcoder_root = workdir / "transcoders"
    lorsa_root = workdir / "lorsas"
    summary_path = workdir / "eval_summary.json"
    spec_path = workdir / "model_spec.json"

    workdir.mkdir(parents=True, exist_ok=True)
    transcoder_root.mkdir(parents=True, exist_ok=True)
    lorsa_root.mkdir(parents=True, exist_ok=True)

    model_cfg = LanguageModelConfig(
        model_name=args.model_name,
        device=primary_device,
        dtype=torch.float16,
    )
    spec = derive_model_spec(model_cfg)
    spec_path.write_text(json.dumps(asdict(spec), indent=2))

    dataset_dir = ensure_dataset(dataset_dir, args.force)
    ensure_activation_cache(args, model_cfg, spec, dataset_dir, activation_dir)

    transcoder_dirs: list[Path] = []
    lorsa_dirs: list[Path] = []

    for layer in range(spec.n_layers):
        print(f"[CRM] training transcoder layer {layer}/{spec.n_layers - 1}")
        transcoder_dirs.append(
            train_transcoder_layer(
                layer=layer,
                args=args,
                spec=spec,
                model_cfg=model_cfg,
                activation_dir=activation_dir,
                result_dir=transcoder_root / f"layer_{layer}",
            )
        )

    for layer in range(spec.n_layers):
        print(f"[CRM] training lorsa layer {layer}/{spec.n_layers - 1}")
        lorsa_dirs.append(
            train_lorsa_layer(
                layer=layer,
                args=args,
                spec=spec,
                model_cfg=model_cfg,
                activation_dir=activation_dir,
                result_dir=lorsa_root / f"layer_{layer}",
            )
        )

    if args.skip_eval:
        print("[CRM] training finished; evaluation skipped")
        return

    print("[CRM] building replacement model")
    replacement_model = build_replacement_model(model_cfg, transcoder_dirs, lorsa_dirs)
    print("[CRM] running attribution")
    summary = run_eval(
        replacement_model=replacement_model,
        prompt=args.eval_prompt,
        eval_batch_size=args.eval_batch_size,
        max_feature_nodes=args.max_feature_nodes,
        parallel_devices=args.parallel_devices,
    )
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
