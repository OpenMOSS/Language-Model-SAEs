from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from lm_saes.activation.processors.activation import ActivationGenerator
from lm_saes.activation.processors.huggingface import HuggingFaceDatasetLoader
from lm_saes.analysis.feature_analyzer import FeatureAnalyzer
from lm_saes.analysis.post_analysis import get_post_analysis_processor
from lm_saes.config import FeatureAnalyzerConfig
from lm_saes.cnnsae import CNNSparseAutoEncoder
from lm_saes.utils.discrete import KeyedDiscreteMapper


@dataclass
class AnalyzerState:
    act_times: torch.Tensor
    max_feature_acts: torch.Tensor
    sum_feature_acts: torch.Tensor
    sample_result: dict[str, dict[str, torch.Tensor] | None]
    mapper: KeyedDiscreteMapper


def _build_discrete_meta(meta: dict[str, list[Any]], mapper: KeyedDiscreteMapper, device: torch.device) -> dict[str, torch.Tensor]:
    """Mimic FeatureAnalyzer.analyze_chunk meta encoding."""
    import json

    discrete_meta: dict[str, torch.Tensor] = {}
    for k, v in meta.items():
        if all(isinstance(item, str) for item in v):
            discrete_meta[k] = torch.tensor(mapper.encode(k, v), device=device, dtype=torch.int32)
            continue
        if all(isinstance(item, (int, float, bool)) for item in v):
            discrete_meta[k] = torch.tensor(v, device=device)
            continue

        def to_str(x: Any) -> str:
            if isinstance(x, str):
                return x
            try:
                return json.dumps(x, sort_keys=True, ensure_ascii=False, default=str)
            except Exception:
                return str(x)

        v_str = [to_str(item) for item in v]
        discrete_meta[k] = torch.tensor(mapper.encode(k, v_str), device=device, dtype=torch.int32)
    return discrete_meta


def _make_state(analyzer_cfg: FeatureAnalyzerConfig, sae: CNNSparseAutoEncoder) -> AnalyzerState:
    d_sae = int(sae.cfg.d_sae)
    return AnalyzerState(
        act_times=torch.zeros((d_sae,), dtype=torch.long, device=sae.cfg.device),
        max_feature_acts=torch.zeros((d_sae,), dtype=sae.cfg.dtype, device=sae.cfg.device),
        sum_feature_acts=torch.zeros((d_sae,), dtype=sae.cfg.dtype, device=sae.cfg.device),
        sample_result={k: None for k in analyzer_cfg.subsamples.keys()},
        mapper=KeyedDiscreteMapper(),
    )


def _inject_sample_reliance(
    *,
    results: list[dict[str, Any]],
    sample_result_top: dict[str, torch.Tensor] | None,
) -> None:
    """Compute and inject per-sample reliance fields into the `top_activations` sampling."""
    if sample_result_top is None:
        return
    img_base = sample_result_top.get("img_base")
    img_shape = sample_result_top.get("img_shape")
    img_texture = sample_result_top.get("img_texture")
    img_color = sample_result_top.get("img_color")
    if img_base is None or img_shape is None or img_texture is None or img_color is None:
        return

    # All tensors shape: [k, d_sae]
    img_base = img_base.detach().cpu()
    img_shape = img_shape.detach().cpu()
    img_texture = img_texture.detach().cpu()
    img_color = img_color.detach().cpu()

    k, d_sae = img_base.shape

    for feat_idx in range(d_sae):
        rs_list: list[float] = []
        rt_list: list[float] = []
        rc_list: list[float] = []
        ps_list: list[float] = []
        pt_list: list[float] = []
        pc_list: list[float] = []
        labels: list[str] = []

        for j in range(k):
            a = float(img_base[j, feat_idx].item())
            bs = float(img_shape[j, feat_idx].item())
            bt = float(img_texture[j, feat_idx].item())
            bc = float(img_color[j, feat_idx].item())
            if a > 0:
                rs = abs(a - bs) / a
                rt = abs(a - bt) / a
                rc = abs(a - bc) / a
            else:
                rs = rt = rc = 0.0
            denom = rs + rt + rc
            if denom > 0:
                ps = rs / denom
                pt = rt / denom
                pc = rc / denom
            else:
                ps = pt = pc = 0.0
            rs_list.append(rs)
            rt_list.append(rt)
            rc_list.append(rc)
            ps_list.append(ps)
            pt_list.append(pt)
            pc_list.append(pc)
            if rs >= rt and rs >= rc:
                labels.append("shape")
            elif rt >= rs and rt >= rc:
                labels.append("texture")
            else:
                labels.append("color")

        # attach to sampling named "top_activations"
        samplings = results[feat_idx].get("samplings", [])
        for s in samplings:
            if s.get("name") == "top_activations":
                s["sample_reliance_relative_changes"] = {
                    "shape": rs_list,
                    "texture": rt_list,
                    "color": rc_list,
                }
                s["sample_reliance_probabilities"] = {
                    "shape": ps_list,
                    "texture": pt_list,
                    "color": pc_list,
                }
                s["sample_reliance_label"] = labels
                break


@torch.no_grad()
def run_joint_reliance(
    *,
    dataset,
    dataset_name: str,
    dataset_meta: Optional[dict[str, Any]],
    model,
    model_name: str,
    sae: CNNSparseAutoEncoder,
    hook_point: str,
    analyzer_cfg: FeatureAnalyzerConfig,
    suppression_params: dict[str, Any],
    top_n_samples_by_cond: Optional[dict[str, int]] = None,
    model_batch_size: int,
    total_analyzing_tokens: int,
    context_size: Optional[int] = None,
    num_workers: int = 8,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
    """
    Single-pass computation of:
      - four analysis tabs (baseline/shape/texture/color)
      - feature-level image-level reliance p_{shape,texture,color}
      - sample-level reliance injected into top_activations samplings
    """
    # Allow per-condition override for top_activations sample count
    cond_cfgs: dict[str, FeatureAnalyzerConfig] = {}
    for cond in ["baseline", "shape", "texture", "color"]:
        cfg = analyzer_cfg.model_copy(deep=True)
        if top_n_samples_by_cond is not None and "top_activations" in cfg.subsamples:
            n = top_n_samples_by_cond.get(cond)
            if n is not None:
                cfg.subsamples["top_activations"]["n_samples"] = int(n)
        cond_cfgs[cond] = cfg

    analyzers = {cond: FeatureAnalyzer(cfg) for cond, cfg in cond_cfgs.items()}

    loader = HuggingFaceDatasetLoader(batch_size=1, with_info=True, show_progress=True, num_workers=num_workers)
    raw_stream = loader.process(dataset, dataset_name=dataset_name, metadata=dataset_meta)
    batcher = ActivationGenerator(hook_points=[hook_point], batch_size=model_batch_size, n_context=context_size)

    states = {cond: _make_state(cfg, sae) for cond, cfg in cond_cfgs.items()}

    a_sum = torch.zeros((int(sae.cfg.d_sae),), dtype=torch.float64, device=sae.cfg.device)
    c_sum = {
        "shape": torch.zeros((int(sae.cfg.d_sae),), dtype=torch.float64, device=sae.cfg.device),
        "texture": torch.zeros((int(sae.cfg.d_sae),), dtype=torch.float64, device=sae.cfg.device),
        "color": torch.zeros((int(sae.cfg.d_sae),), dtype=torch.float64, device=sae.cfg.device),
    }

    def make_meta_list(meta_list, kind: str):
        sup_cfg = suppression_params.get(kind, {"kind": "none"})
        return [{**(m or {}), "reliance_suppression": sup_cfg} for m in meta_list]

    n_tokens = 0

    for batch in batcher.batched(raw_stream):
        meta_list = batch.get("meta", [{} for _ in range(len(batch.get("image", [])))])
        cond_data: dict[str, dict[str, Any]] = {}
        imgs: dict[str, torch.Tensor] = {}

        # forward for each condition
        for cond in ["baseline", "shape", "texture", "color"]:
            variant = dict(batch)
            variant["meta"] = make_meta_list(meta_list, cond)
            processed = model.preprocess_raw_data(variant)
            activations = model.to_activations(processed, [hook_point], n_context=context_size)
            existing_meta = processed.get("meta", [{} for _ in range(len(processed.get("text", [])))])
            meta = [{"model_name": model_name} | existing_meta[i] for i in range(len(existing_meta))]
            cond_batch = {**activations, "meta": meta}

            x, enc_kwargs, _ = sae.prepare_input(cond_batch)
            fa: torch.Tensor = sae.encode(x, **enc_kwargs)
            imgs[cond] = fa.clamp(min=0.0).sum(dim=1).to(torch.float64)
            cond_data[cond] = {
                "feature_acts": fa,
                "meta": meta,
                "tokens": cond_batch.get("tokens"),
            }

        # reliance accumulation
        a_sum += imgs["baseline"].sum(dim=0)
        for k in ["shape", "texture", "color"]:
            c_sum[k] += (imgs["baseline"] - imgs[k]).abs().sum(dim=0)

        # analyzer updates per condition
        for cond in ["baseline", "shape", "texture", "color"]:
            st = states[cond]
            analyzer = analyzers[cond]
            fa = cond_data[cond]["feature_acts"]
            meta = cond_data[cond]["meta"]

            st.act_times += fa.gt(0.0).sum(dim=(0, 1))
            st.max_feature_acts = torch.max(st.max_feature_acts, fa.max(dim=0).values.max(dim=0).values)
            st.sum_feature_acts += fa.clamp(min=0.0).sum(dim=(0, 1))

            meta_dict = {k: [m.get(k) for m in meta] for k in meta[0].keys()}
            discrete_meta = _build_discrete_meta(meta_dict, st.mapper, device=sae.cfg.device)

            st.sample_result = analyzer._process_batch(  # type: ignore[attr-defined]
                fa,
                discrete_meta,
                st.sample_result,
                st.max_feature_acts,
                device_mesh=None,
                sae_is_cnnsae=True,
                extra_batch_data={
                    "img_base": imgs["baseline"],
                    "img_shape": imgs["shape"],
                    "img_texture": imgs["texture"],
                    "img_color": imgs["color"],
                },
            )

        tokens = cond_data["baseline"]["tokens"]
        if tokens is not None:
            n_tokens += int(tokens.numel())
        if n_tokens >= total_analyzing_tokens:
            break

    # post process analyses
    post = get_post_analysis_processor(sae.cfg.sae_type)
    analyses: dict[str, list[dict[str, Any]]] = {}
    for cond, st in states.items():
        sample_result = {k: v for k, v in st.sample_result.items() if v is not None}
        sample_result = {k: {kk: vv for kk, vv in v.items()} for k, v in sample_result.items()}
        res = post.process(
            sae=sae,
            act_times=st.act_times,
            n_analyzed_tokens=n_tokens,
            max_feature_acts=st.max_feature_acts,
            sum_feature_acts=st.sum_feature_acts,
            sample_result=sample_result,
            mapper=st.mapper,
            device_mesh=None,
        )
        # inject sample-level reliance into top_activations
        top_sr = sample_result.get("top_activations")
        _inject_sample_reliance(results=res, sample_result_top=top_sr)
        analyses[cond] = res

    a_np = a_sum.detach().cpu().numpy()
    p = {
        k: (c_sum[k] / torch.clamp(a_sum, min=1e-12)).detach().cpu().numpy()
        for k in ["shape", "texture", "color"]
    }

    return analyses, p

