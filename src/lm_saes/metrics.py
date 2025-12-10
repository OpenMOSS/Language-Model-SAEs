import functools
from abc import ABC
from typing import Generic, TypeVar, cast

import einops
import torch
from torch import Tensor
from torch.types import Number
from transformer_lens import HookedTransformer

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.lorsa import LowRankSparseAttention
from lm_saes.sae import SparseAutoEncoder
from lm_saes.utils.distributed.ops import item
from lm_saes.utils.logging import get_distributed_logger

logger = get_distributed_logger("metrics")


def h(names: tuple[str, ...]) -> str:
    return " ".join(names)


def reduce(tensor: Tensor, specs: tuple[str, ...], reduction_map: dict[str, str]) -> tuple[Tensor, tuple[str, ...]]:
    """Reduce the tensor by the mapping of dimension names to reduction functions."""

    assert tensor.ndim == len(specs), f"Tensor has {tensor.ndim} dimensions, but specs have {len(specs)} dimensions"

    def _reduce(tensor: Tensor, specs: tuple[str, ...], dim: str, reduction: str) -> tuple[Tensor, tuple[str, ...]]:
        target_specs = tuple(filter(lambda x: x != dim, specs))
        if specs == target_specs:
            return tensor, target_specs
        return einops.reduce(tensor, f"{h(specs)} -> {h(target_specs)}", reduction), target_specs

    return functools.reduce(lambda acc, item: _reduce(*acc, *item), reduction_map.items(), (tensor, specs))


def apply_token_mask(
    tensor: Tensor, specs: tuple[str, ...], mask: Tensor | None = None, reduction: str = "sum"
) -> tuple[Tensor, tuple[str, ...]]:
    """Apply the token mask to the tensor. Mask should be a 0/1 tensor with the same shape as the token part of the tensor, i.e. the shape of batch and context dimensions."""

    assert tensor.ndim == len(specs), f"Tensor has {tensor.ndim} dimensions, but specs have {len(specs)} dimensions"

    if mask is None:
        return reduce(tensor, specs, {"batch": reduction, "context": reduction})

    token_specs = tuple(filter(lambda x: x in ["batch", "context"], specs))
    token_shape = tuple([size for size, spec in zip(tensor.shape, specs) if spec in token_specs])
    assert mask.shape == token_shape, (
        f"Mask has shape {mask.shape}, but input tensor has token part of shape {token_shape}"
    )

    target_specs = tuple(filter(lambda x: x not in token_specs, specs))
    result = einops.einsum(tensor, mask, f"{h(specs)} {h(token_specs)} -> {h(target_specs)}")
    if reduction == "mean":
        result = result / mask.sum()
    return result, target_specs


T = TypeVar("T", bound=Number | Tensor)


class Record(Generic[T]):
    def __init__(self):
        self.value: T | None = None
        self.count = 0

    def update(self, value: T, count: Tensor | int = 1):
        self.value = cast(T, self.value + value) if self.value is not None else value
        self.count += count

    def compute(self) -> T:
        assert self.value is not None, "Record must be updated before computing"
        result = cast(T, self.value / self.count)
        self.value = None
        self.count = 0
        return result


class Metric(ABC):
    """Base class for all metrics."""

    def update(self, ctx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Update the metric with the given context.

        Args:
            ctx: Dictionary of context tensors.

        Returns:
            Update to the context.
        """
        return {}

    def compute(self) -> dict[str, Number]:
        """Compute the metric.

        Returns:
            Dictionary of metric names to values."""
        raise NotImplementedError("Subclasses must implement this method")


class FrequencyMetric(Metric):
    def __init__(self, sae: AbstractSparseAutoEncoder):
        self.sae = sae
        self.act_freq_scores: Record[Tensor] = Record()
        self.specs: tuple[str, ...] | None = None

    def update(self, ctx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feature_acts = ctx["feature_acts"]
        mask = ctx.get("mask")
        n_tokens = ctx["n_tokens"]
        act_freq_scores, specs = apply_token_mask(
            (feature_acts > 0).float(), self.sae.specs.feature_acts(feature_acts), mask, "sum"
        )
        self.act_freq_scores.update(act_freq_scores, n_tokens)
        self.specs = specs
        return {"act_freq_scores": act_freq_scores}

    def compute(self) -> dict[str, Number]:
        assert self.specs is not None, "Metrics must be updated before computing"
        feature_sparsity = self.act_freq_scores.compute()
        metrics = {
            "sparsity/above_1e-1": item((feature_sparsity > 1e-1).sum()),
            "sparsity/above_1e-2": item((feature_sparsity > 1e-2).sum()),
            "sparsity/below_1e-5": item((feature_sparsity < 1e-5).sum()),
            "sparsity/below_1e-6": item((feature_sparsity < 1e-6).sum()),
            "sparsity/below_1e-7": item((feature_sparsity < 1e-7).sum()),
        }
        if "layers" in self.specs:
            for l in range(feature_sparsity.size(self.specs.index("layers"))):
                metrics[f"sparsity/above_1e-1_layer{l}"] = item(
                    (feature_sparsity > 1e-1).select(self.specs.index("layers"), l).sum()
                )
                metrics[f"sparsity/above_1e-2_layer{l}"] = item(
                    (feature_sparsity > 1e-2).select(self.specs.index("layers"), l).sum()
                )
                metrics[f"sparsity/below_1e-5_layer{l}"] = item(
                    (feature_sparsity < 1e-5).select(self.specs.index("layers"), l).sum()
                )
                metrics[f"sparsity/below_1e-6_layer{l}"] = item(
                    (feature_sparsity < 1e-6).select(self.specs.index("layers"), l).sum()
                )
                metrics[f"sparsity/below_1e-7_layer{l}"] = item(
                    (feature_sparsity < 1e-7).select(self.specs.index("layers"), l).sum()
                )
        return metrics


class LossMetric(Metric):
    def __init__(self, sae: AbstractSparseAutoEncoder):
        self.sae = sae
        self.loss: Record[Tensor] = Record()
        self.l_rec: Record[Tensor] = Record()
        self.l_s: Record[Tensor] = Record()
        self.l_p: Record[Tensor] = Record()

    def update(self, ctx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        loss, l_rec, l_s, l_p = ctx["loss"], ctx["l_rec"], ctx.get("l_s"), ctx.get("l_p")

        self.loss.update(loss)
        self.l_rec.update(l_rec.mean())
        if l_s is not None:
            self.l_s.update(l_s.mean())
        if l_p is not None:
            self.l_p.update(l_p.mean())

        return {}

    def compute(self) -> dict[str, Number]:
        metrics = {
            "losses/overall_loss": item(self.loss.compute()),
            "losses/mse_loss": item(self.l_rec.compute()),
        }
        if self.l_s.value is not None:
            metrics["losses/sparsity_loss"] = item(self.l_s.compute())
        if self.l_p.value is not None:
            metrics["losses/lp_loss"] = item(self.l_p.compute())
        return metrics


class MeanFeatureActMetric(Metric):
    def __init__(self, sae: AbstractSparseAutoEncoder):
        self.sae = sae
        self.mean_feature_act: Record[Tensor] = Record()

    def update(self, ctx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feature_acts = ctx["feature_acts"]

        act_feature_counts = feature_acts.gt(0).float().sum()
        mean_feature_act = feature_acts.sum() / act_feature_counts

        self.mean_feature_act.update(mean_feature_act)
        return {}

    def compute(self) -> dict[str, Number]:
        return {"metrics/mean_feature_act": item(self.mean_feature_act.compute())}


class ExplainedVarianceMetric(Metric):
    def __init__(self, sae: AbstractSparseAutoEncoder):
        self.sae = sae
        self.explained_variance: Record[Tensor] = Record()
        self.explained_variance_legacy: Record[Tensor] = Record()

    def update(self, ctx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        label, reconstructed, mask = ctx["label"], ctx["reconstructed"], ctx.get("mask")

        # The following computations assume the model dimension is the last dimension,
        # without further assumptions about the tensor specs.
        label_mean = apply_token_mask(
            label,
            self.sae.specs.label(label),
            mask,
            reduction="mean",
        )[0]  # shape: (d_model)
        per_token_l2_loss = (reconstructed - label).pow(2).sum(dim=-1)
        total_variance = (label - label_mean).pow(2).sum(dim=-1)
        explained_variance_legacy = apply_token_mask(
            1 - per_token_l2_loss / total_variance, self.sae.specs.label(label)[:-1], mask, reduction="mean"
        )[0]
        l2_loss_mean = apply_token_mask(
            per_token_l2_loss,
            self.sae.specs.label(label)[:-1],
            mask,
            reduction="mean",
        )[0]
        total_variance_mean = apply_token_mask(
            total_variance,
            self.sae.specs.label(label)[:-1],
            mask,
            reduction="mean",
        )[0]
        if torch.any(torch.isinf(total_variance_mean)):
            logger.warning("Some of total_variance_mean is inf. Check dtype or scaling.")
        explained_variance = 1 - l2_loss_mean / total_variance_mean
        self.explained_variance.update(explained_variance)
        self.explained_variance_legacy.update(explained_variance_legacy)
        return {
            "explained_variance": explained_variance,
            "explained_variance_legacy": explained_variance_legacy,
        }

    def compute(self) -> dict[str, Number]:
        return {
            "metrics/explained_variance": item(self.explained_variance.compute().mean()),
            "metrics/explained_variance_legacy": item(self.explained_variance_legacy.compute().mean()),
        }


class L0Metric(Metric):
    def __init__(self, sae: AbstractSparseAutoEncoder):
        self.sae = sae
        self.l0: Record[Tensor] = Record()
        self.specs: tuple[str, ...] | None = None

    def update(self, ctx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feature_acts = ctx["feature_acts"]
        mask = ctx.get("mask")
        n_tokens = ctx["n_tokens"]
        l0, l0_specs = apply_token_mask(
            (feature_acts > 0).float(),
            self.sae.specs.feature_acts(feature_acts),
            mask,
            "sum",
        )
        l0, l0_specs = reduce(l0, l0_specs, {"sae": "sum"})

        self.l0.update(l0, n_tokens)
        self.specs = l0_specs

        return {
            "l0": l0,
        }

    def compute(self) -> dict[str, Number]:
        assert self.specs is not None, "Metrics must be updated before computing"
        l0 = self.l0.compute()
        l0 = reduce(l0, self.specs, {"layers": "sum"})[0]
        return {"metrics/l0": item(l0.mean())}


class L2NormErrorMetric(Metric):
    def __init__(self, sae: AbstractSparseAutoEncoder):
        self.sae = sae
        self.l2_norm_error: Record[Tensor] = Record()
        self.l2_norm_error_ratio: Record[Tensor] = Record()

    def update(self, ctx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        label, reconstructed, mask = ctx["label"], ctx["reconstructed"], ctx.get("mask")
        l2_norm_error = apply_token_mask(
            (reconstructed - label).pow(2).sum(dim=-1).sqrt(), self.sae.specs.label(label)[:-1], mask, "mean"
        )[0]
        l2_norm_error_ratio = (
            l2_norm_error / apply_token_mask(label.norm(p=2, dim=-1), self.sae.specs.label(label)[:-1], mask, "mean")[0]
        )

        self.l2_norm_error.update(l2_norm_error)
        self.l2_norm_error_ratio.update(l2_norm_error_ratio)

        return {}

    def compute(self) -> dict[str, Number]:
        return {
            "metrics/l2_norm_error": item(self.l2_norm_error.compute()),
            "metrics/l2_norm_error_ratio": item(self.l2_norm_error_ratio.compute()),
        }


class ModelSpecificMetric(Metric):
    def __init__(self, sae: AbstractSparseAutoEncoder):
        self.sae = sae
        self.count = 0
        self.metrics: dict[str, Record[float]] = {}

    def update(self, ctx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        metrics = self.sae.compute_training_metrics(**ctx)
        for k, v in metrics.items():
            if k not in self.metrics:
                self.metrics[k] = Record()
            self.metrics[k].update(v)
        return {}

    def compute(self) -> dict[str, Number]:
        return {k: v.compute() for k, v in self.metrics.items()}


class DownstreamMetric(Metric):
    def __init__(self, sae: SparseAutoEncoder | LowRankSparseAttention, model: HookedTransformer):
        self.sae = sae
        self.model = model
        self.downstream_loss_original: Record[Tensor] = Record()
        self.downstream_loss_reconstructed: Record[Tensor] = Record()
        self.downstream_loss_ablated: Record[Tensor] = Record()
        self.downstream_loss_ratio: Record[Tensor] = Record()

    def update(self, ctx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokens = ctx["tokens"]
        mask = ctx["mask"]

        assert tokens.ndim == 2, "Tokens must be a 2D tensor"
        specs = ("batch", "context")

        loss, cache = cast(
            tuple[torch.Tensor, dict[str, torch.Tensor]],
            self.model.run_with_cache(
                tokens,
                return_type="loss",
                loss_per_token=True,
                names_filter=[self.sae.cfg.hook_point_in, self.sae.cfg.hook_point_out],
                return_cache_object=False,
            ),
        )

        batch = {"tokens": tokens, "mask": mask, **cache}

        batch, scale_factors = self.sae.normalize_activations(batch, return_scale_factor=True)

        x, encoder_kwargs, decoder_kwargs = self.sae.prepare_input(batch)
        reconstructed = self.sae.forward(x, encoder_kwargs=encoder_kwargs, decoder_kwargs=decoder_kwargs)

        reconstructed_batch = batch | {self.sae.cfg.hook_point_out: reconstructed}
        reconstructed_batch = self.sae.denormalize_activations(reconstructed_batch, scale_factors=scale_factors)

        def replace_hook_reconstructed(activations: torch.Tensor, hook_point: str) -> torch.Tensor:
            return torch.where(mask, reconstructed_batch[self.sae.cfg.hook_point_out], activations)

        reconstructed_loss: torch.Tensor = self.model.run_with_hooks(
            tokens,
            return_type="loss",
            fwd_hooks=[(self.sae.cfg.hook_point_out, replace_hook_reconstructed)],
            loss_per_token=True,
        )

        def replace_hook_ablated(activations: torch.Tensor, hook_point: str) -> torch.Tensor:
            return torch.where(mask, torch.zeros_like(activations), activations)

        ablated_loss: torch.Tensor = self.model.run_with_hooks(
            tokens,
            return_type="loss",
            fwd_hooks=[(self.sae.cfg.hook_point_out, replace_hook_ablated)],
            loss_per_token=True,
        )

        loss = apply_token_mask(loss, specs, mask, "mean")[0]
        reconstructed_loss = apply_token_mask(reconstructed_loss, specs, mask, "mean")[0]
        ablated_loss = apply_token_mask(ablated_loss, specs, mask, "mean")[0]

        self.downstream_loss_original.update(loss)
        self.downstream_loss_reconstructed.update(reconstructed_loss)
        self.downstream_loss_ablated.update(ablated_loss)
        self.downstream_loss_ratio.update((ablated_loss - loss) / (ablated_loss - reconstructed_loss))

        return {}

    def compute(self) -> dict[str, Number]:
        return {
            "metrics/downstream_loss_original": item(self.downstream_loss_original.compute()),
            "metrics/downstream_loss_reconstructed": item(self.downstream_loss_reconstructed.compute()),
            "metrics/downstream_loss_ablated": item(self.downstream_loss_ablated.compute()),
            "metrics/downstream_loss_ratio": item(self.downstream_loss_ratio.compute()),
        }
