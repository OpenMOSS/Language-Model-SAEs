import functools
from typing import Iterable

import torch
import torch.distributed.tensor
from lm_saes.config import BaseConfig
from lm_saes.metrics import (
    DownstreamMetric,
    ExplainedVarianceMetric,
    FrequencyMetric,
    L0Metric,
    L2NormErrorMetric,
    LossMetric,
    MeanFeatureActMetric,
    Metric,
    ModelSpecificMetric,
)
from lm_saes.models.lorsa import LowRankSparseAttention
from lm_saes.models.sae import SparseAutoEncoder
from lm_saes.models.sparse_dictionary import SparseDictionary
from lm_saes.utils.distributed.ops import item
from lm_saes.utils.logging import get_distributed_logger, log_metrics
from tqdm import tqdm
from transformer_lens import HookedTransformer
from wandb.sdk.wandb_run import Run

logger = get_distributed_logger("evaluator")


class EvalConfig(BaseConfig):
    total_eval_tokens: int = 1000000
    """Total number of tokens to evaluate on"""


class Evaluator:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg

    def evaluate(
        self,
        sae: SparseDictionary,
        data_stream: Iterable[dict[str, torch.Tensor]],
        wandb_logger: Run | None = None,
        model: HookedTransformer | None = None,
    ) -> dict[str, float]:
        metrics: list[Metric] = [
            FrequencyMetric(sae),
            LossMetric(sae),
            MeanFeatureActMetric(sae),
            ExplainedVarianceMetric(sae),
            L0Metric(sae),
            L2NormErrorMetric(sae),
            ModelSpecificMetric(sae),
        ]

        if model is not None and isinstance(sae, SparseAutoEncoder | LowRankSparseAttention):
            logger.info("Downstream metrics available.")
            metrics.append(DownstreamMetric(sae, model))

        total_tokens = 0

        proc_bar = tqdm(total=self.cfg.total_eval_tokens)

        for batch in data_stream:
            batch = sae.normalize_activations(batch)

            ctx = sae.compute_loss(batch, return_aux_data=True)

            for metric in metrics:
                ctx = {**ctx, **metric.update(ctx)}

            total_tokens += batch["tokens"].numel() if batch.get("mask") is None else int(item(batch["mask"].sum()))
            proc_bar.update(batch["tokens"].numel() if batch.get("mask") is None else int(item(batch["mask"].sum())))
            if total_tokens >= self.cfg.total_eval_tokens:
                break

        proc_bar.close()

        results = functools.reduce(lambda x, y: x | y, [metric.compute() for metric in metrics])

        log_metrics(logger.logger, results, title="Evaluation Metrics")

        if wandb_logger is not None:
            wandb_logger.log(results)

        return results
