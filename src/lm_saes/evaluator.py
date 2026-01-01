import functools
import json
from typing import Iterable, Literal

import torch
import torch.distributed.tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer
from wandb.sdk.wandb_run import Run

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.circuit.attribution import attribute
from lm_saes.circuit.graph import Graph, compute_influence, normalize_matrix
from lm_saes.circuit.replacement_model import ReplacementModel
from lm_saes.config import BaseConfig
from lm_saes.lorsa import LowRankSparseAttention
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
from lm_saes.sae import SparseAutoEncoder
from lm_saes.utils.distributed.ops import item
from lm_saes.utils.logging import get_distributed_logger, log_metrics
from lm_saes.utils.timer import timer

logger = get_distributed_logger("evaluator")


class EvalConfig(BaseConfig):
    total_eval_tokens: int = 1000000


class GraphEvalConfig(BaseConfig):
    max_n_logits: int = 2
    # How many logits to attribute from, max. We attribute to min(max_n_logits, n_logits_to_reach_desired_log_prob); see below for the latter

    desired_logit_prob: float = 0.95
    # Attribution will attribute from the minimum number of logits needed to reach this probability mass (or max_n_logits, whichever is lower)

    max_feature_nodes: int = 1024
    # Only attribute from this number of feature nodes, max. Lower is faster, but you will lose more of the graph. None means no limit.

    batch_size: int = 2
    # Batch size when attributing

    offload: Literal[None, "disk", "cpu"] = None
    # Offload various parts of the model during attribution to save memory. Can be 'disk', 'cpu', or None (keep on GPU)

    start_from: int = 0


class Evaluator:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg

    def evaluate(
        self,
        sae: AbstractSparseAutoEncoder,
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


def compute_graph_scores(graph: Graph, use_lorsa: bool = True) -> tuple[float, float]:
    """Copy from circuit-tracer
    Compute metrics for evaluating how well the graph captures the model's computation.
    This function calculates two complementary scores that measure how much of the model's
    computation flows through interpretable feature nodes versus reconstruction error nodes:
    1. Replacement Score: Measures the fraction of end-to-end influence from input tokens
       to output logits that flows through feature nodes rather than error nodes. This is
       a strict metric that rewards complete explanations where tokens influence logits
       entirely through features.
    2. Completeness Score: Measures the fraction of incoming edges to all nodes (weighted
       by each node's influence on the output) that originate from feature or token nodes
       rather than error nodes. This metric gives partial credit for nodes that are mostly
       explained by features, even if some error influence remains.
    Args:
        graph: The computation graph containing nodes for features, errors, tokens, and logits,
               along with their connections and influence weights.
    Returns:
        tuple[float, float]: A tuple containing:
            - replacement_score: Fraction of token-to-logit influence through features (0-1)
            - completeness_score: Weighted fraction of non-error inputs across all nodes (0-1)
    Note:
        Higher scores indicate better model interpretability, with 1.0 representing perfect
        reconstruction where all computation flows through interpretable features. Lower
        scores indicate more reliance on error nodes, suggesting incomplete feature coverage.
    """

    # Extract dimensions
    n_logits = len(graph.logit_tokens)
    n_features = len(graph.selected_features)
    layers = graph.cfg.n_layers
    error_end_idx = n_features + 2 * graph.n_pos * layers if use_lorsa else n_features + graph.n_pos * layers
    token_end_idx = error_end_idx + len(graph.input_tokens)

    logit_weights = torch.zeros(graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device)
    logit_weights[-n_logits:] = graph.logit_probabilities

    normalized_matrix = normalize_matrix(graph.adjacency_matrix)
    node_influence = compute_influence(normalized_matrix, logit_weights)
    token_influence = node_influence[error_end_idx:token_end_idx].sum()
    error_influence = node_influence[n_features:error_end_idx].sum()

    replacement_score = token_influence / (token_influence + error_influence)

    # non_error_fractions = normalized_matrix[:, :].sum(dim=-1) - normalized_matrix[:, n_features:error_end_idx].sum(dim=-1) # not from error (Ibelieve this is correct)
    non_error_fractions = 1 - normalized_matrix[:, n_features:error_end_idx].sum(dim=-1)  # not from error
    output_influence = node_influence + logit_weights
    completeness_score = (non_error_fractions * output_influence).sum() / output_influence.sum()

    return replacement_score.item(), completeness_score.item()


class GraphEval:
    def __init__(self, cfg: GraphEvalConfig):
        self.cfg = cfg
        self.replacement_scores = []
        self.completeness_scores = []
        self.prompt = []

    def eval(
        self,
        replacement_model: ReplacementModel,
        dataset_path: str,
        use_lorsa: bool = True,
        add_bos: bool = True,
    ):
        timer.reset()

        with timer.time("Init. dataset"):
            dataset = json.load(open(dataset_path, "r"))

        for i in range(self.cfg.start_from, len(dataset)):
            data = dataset[i]

            # Add <BOS> if there doesn't have
            if add_bos and data["prompt"][0] != "<":
                prompt = "<|endoftext|> " + data["prompt"]
            else:
                prompt = data["prompt"]

            replacement_model._configure_gradient_flow()
            replacement_model._deduplicate_attention_buffers()
            replacement_model.setup()
            graph = attribute(
                prompt=prompt,
                model=replacement_model,
                max_n_logits=self.cfg.max_n_logits,
                desired_logit_prob=self.cfg.desired_logit_prob,
                batch_size=self.cfg.batch_size,
                max_feature_nodes=self.cfg.max_feature_nodes,
                offload=self.cfg.offload,
                use_lorsa=use_lorsa,
            )

            replacement_score, completeness_score = compute_graph_scores(graph, use_lorsa=use_lorsa)

            self.replacement_scores.append(replacement_score)
            self.completeness_scores.append(completeness_score)
