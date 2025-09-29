import json
from typing import Iterable

import torch
import torch.distributed.tensor
from einops import reduce
from torch._tensor import Tensor
from torch.distributed.tensor import DTensor
from tqdm import tqdm
from transformer_lens import HookedTransformer

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.circuit.attribution import attribute
from lm_saes.circuit.graph import Graph, compute_influence, normalize_matrix
from lm_saes.circuit.replacement_model import ReplacementModel
from lm_saes.config import EvalConfig, GraphEvalConfig
from lm_saes.sae import SparseAutoEncoder
from lm_saes.utils.logging import get_distributed_logger, log_metrics
from lm_saes.utils.timer import timer
from wandb.sdk.wandb_run import Run

logger = get_distributed_logger("evaluator")


def item(x: torch.Tensor) -> float:
    return x.item() if not isinstance(x, DTensor) else x.full_tensor().item()


class Evaluator:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.cur_step = 0
        self.cur_tokens = 0
        self.metrics = {}

    @torch.no_grad()
    def _evaluate_activations(
        self,
        sae: AbstractSparseAutoEncoder,
        log_info: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        useful_token_mask: torch.Tensor,
    ) -> None:
        """Evaluate SAE activations and compute various metrics.

        Args:
            sae: Sparse autoencoder model
            log_info: Dictionary containing logging information and feature activations
            batch: Dictionary of activation tensors at different hook points
            useful_token_mask: Boolean mask indicating valid tokens
        """

        def log_metric(metric: str, value: float) -> None:
            """Add or append a metric value to self.metrics."""
            self.metrics[metric] = (
                torch.tensor([value], device=self.cfg.device)
                if metric not in self.metrics
                else torch.cat([self.metrics[metric], torch.tensor([value], device=self.cfg.device)])
            )

        # 1. Basic token and loss metrics
        log_metric("n_tokens", item(useful_token_mask.sum()))
        for loss_key in [
            "loss_mean",
            "loss_reconstruction_mean",
        ]:  # when using token input loss_mean & loss_reconstruction_mean are expected to be present
            if loss_key in log_info:
                log_metric(loss_key, item(log_info.pop(loss_key)))

        # 2. Get activations and compute reconstructions
        x, encode_kwargs = sae.prepare_input(batch)
        label = sae.prepare_label(batch)
        feature_acts = sae.encode(x, **encode_kwargs)
        reconstructed = sae.decode(feature_acts)

        # 3. Compute sparsity metrics
        l0 = (feature_acts > 0).float().sum(-1)
        if sae.device_mesh is not None:
            l0 = l0.full_tensor()
        if sae.cfg.sae_type == "clt":
            label = label.permute(1, 0, 2)
            reconstructed = reconstructed.permute(1, 0, 2)
            l0_dict = {f"l0_layer{l}": l0[:, l].mean().item() for l in range(l0.size(1))}
            for key, value in l0_dict.items():
                log_metric(key, value)

            l0 = l0.sum(-1)  # for clt, l0 is the sum of l0s of all layers

        log_metric("l0", item(l0.mean()))

        # 4. Compute reconstruction quality metrics
        # L2 reconstruction error
        per_token_l2_loss = (reconstructed - label).pow(2).sum(dim=-1)
        l2_norm_error = per_token_l2_loss.sqrt().mean()
        log_metric("l2_norm_error", item(l2_norm_error))

        # Normalized metrics
        l2_norm_error_ratio = l2_norm_error / label.norm(p=2, dim=-1).mean()
        log_metric("l2_norm_error_ratio", item(l2_norm_error_ratio))

        activation_variance = (label - label.mean(0, keepdim=True)).pow(2)
        l_rec = (reconstructed - label).pow(2) / activation_variance.sum(dim=-1, keepdim=True).clamp(
            min=1e-8
        ).sqrt().mean()
        log_metric("l_rec", item(l_rec.mean()))

        # Explained variance
        total_variance: Tensor = (label - label.mean(0)).pow(2).sum(dim=-1)
        explained_variance = 1 - per_token_l2_loss / total_variance
        log_metric("explained_variance", item(explained_variance.mean()))

        if sae.cfg.sae_type == "clt":
            per_layer_ev = explained_variance.mean(0)
            per_layer_ev_dict = {
                f"explained_variance_layer{l}": per_layer_ev[l].item() for l in range(per_layer_ev.size(0))
            }
            for key, value in per_layer_ev_dict.items():
                log_metric(key, value)

        # 5. Update feature activation tracking
        if sae.cfg.sae_type == "clt":
            reduce_str = "... layers d_sae -> layers d_sae"
        else:
            reduce_str = "... d_sae -> d_sae"

        act_freq_scores = reduce(
            (feature_acts.abs().gt(0)).float(),
            reduce_str,
            "sum",
        )
        if isinstance(act_freq_scores, DTensor):
            act_freq_scores = act_freq_scores.full_tensor()

        log_info["act_freq_scores"] += act_freq_scores
        log_info["n_frac_active_tokens"] += item(useful_token_mask.sum())

        # 6. Periodic feature sparsity logging
        if (self.cur_step + 1) % self.cfg.feature_sampling_window == 0:
            feature_sparsity = log_info["act_freq_scores"] / log_info["n_frac_active_tokens"]
            if sae.cfg.sae_type == "clt":
                above_1e_1 = (feature_sparsity > 1e-1).sum(-1)
                above_1e_2 = (feature_sparsity > 1e-2).sum(-1)
                below_1e_5 = (feature_sparsity < 1e-5).sum(-1)
                below_1e_6 = (feature_sparsity < 1e-6).sum(-1)
                sparsity_results = {}

                for l in range(sae.cfg.n_layers):
                    sparsity_results[f"above_1e-1_layer{l}"] = above_1e_1[l].item()
                    sparsity_results[f"above_1e-2_layer{l}"] = above_1e_2[l].item()

                for l in range(sae.cfg.n_layers):
                    sparsity_results[f"below_1e-5_layer{l}"] = below_1e_5[l].item()
                    sparsity_results[f"below_1e-6_layer{l}"] = below_1e_6[l].item()

                sparsity_results["above_1e-1"] = above_1e_1.sum().item()
                sparsity_results["above_1e-2"] = above_1e_2.sum().item()
                sparsity_results["below_1e-5"] = below_1e_5.sum().item()
                sparsity_results["below_1e-6"] = below_1e_6.sum().item()

            else:
                sparsity_results = {
                    "above_1e-1": (feature_sparsity > 1e-1).sum(-1).item(),
                    "above_1e-2": (feature_sparsity > 1e-2).sum(-1).item(),
                    "below_1e-5": (feature_sparsity < 1e-5).sum(-1).item(),
                    "below_1e-6": (feature_sparsity < 1e-6).sum(-1).item(),
                }

            for key, value in sparsity_results.items():
                log_metric(key, value)

            # Reset tracking counters
            log_info["act_freq_scores"].zero_()
            log_info["n_frac_active_tokens"].zero_()

    @torch.no_grad()
    def process_metrics(self, wandb_logger: Run | None = None) -> None:
        def calc_mean(metric: str) -> float:
            return item(
                torch.sum(self.metrics[metric] * self.metrics["n_tokens"]) / torch.sum(self.metrics["n_tokens"])
            )

        sparsity_metrics = [k for k in self.metrics.keys() if "above_1e" in k or "below_1e" in k]
        for metric in sparsity_metrics:
            self.metrics[metric] = self.metrics[metric].float().mean().item()

        self.metrics["l_rec"] = calc_mean("l_rec")
        l0_metrics = [k for k in self.metrics.keys() if "l" in k]
        for metric in l0_metrics:
            self.metrics[metric] = calc_mean(metric)

        self.metrics["l2_norm_error"] = calc_mean("l2_norm_error")
        self.metrics["l2_norm_error_ratio"] = calc_mean("l2_norm_error_ratio")
        ev_metrics = [k for k in self.metrics.keys() if "explained_variance" in k]
        for metric in ev_metrics:
            self.metrics[metric] = calc_mean(metric)

        for loss_key in [
            "loss_mean",
            "loss_reconstruction_mean",
        ]:  # when using token input, loss_mean & loss_reconstruction_mean are expected to be present
            if loss_key in self.metrics:
                self.metrics[loss_key] = calc_mean(loss_key)

        if wandb_logger is not None:
            wandb_logger.log(self.metrics)

        self.metrics.pop("n_tokens")
        log_metrics(logger.logger, self.metrics, title="Evaluation Metrics")

    @torch.no_grad()
    def _evaluate_tokens(
        self, sae: SparseAutoEncoder, batch: dict[str, torch.Tensor], model: HookedTransformer
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        input_ids = batch["tokens"]  # shape: (seq_len)
        assert model.tokenizer is not None, "Tokenizer is required for token input"
        filter_tokens = [model.tokenizer.eos_token_id, model.tokenizer.bos_token_id, model.tokenizer.pad_token_id]
        useful_token_mask = torch.isin(input_ids, torch.tensor(filter_tokens), invert=True)  # shape: (seq_len)

        loss, cache = model.run_with_cache(
            input_ids,
            return_type="loss",
            loss_per_token=True,
            names_filter=[sae.cfg.hook_point_in, sae.cfg.hook_point_out],
            return_cache_object=False,
        )
        # TODO: check normalization
        reconstructed_activations = sae.forward(cache[sae.cfg.hook_point_in], tokens=input_ids)

        def replace_hook(activations: torch.Tensor, hook_point: str) -> torch.Tensor:
            return torch.where(useful_token_mask, reconstructed_activations, activations)

        reconstructed_loss: torch.Tensor = model.run_with_hooks(
            input_ids,
            return_type="loss",
            fwd_hooks=[(sae.cfg.hook_point_out, replace_hook)],
            loss_per_token=True,
        )

        def get_useful_token_loss(loss: torch.Tensor):
            return loss[useful_token_mask].mean()

        loss_dict = {
            "loss_mean": get_useful_token_loss(loss),  # type: ignore
            "loss_reconstruction_mean": get_useful_token_loss(reconstructed_loss),
        }
        return cache, loss_dict, useful_token_mask

    def evaluate(
        self,
        sae: AbstractSparseAutoEncoder,
        data_stream: Iterable[dict[str, torch.Tensor]],
        wandb_logger: Run | None = None,
        model: HookedTransformer | None = None,
    ) -> None:
        act_freq_scores_shape = (sae.cfg.n_layers, sae.cfg.d_sae) if sae.cfg.sae_type == "clt" else (sae.cfg.d_sae,)  # type: ignore
        log_info = {
            "act_freq_scores": torch.zeros(act_freq_scores_shape, device=sae.cfg.device, dtype=sae.cfg.dtype),
            "n_frac_active_tokens": torch.tensor([0], device=sae.cfg.device, dtype=torch.int),
        }
        proc_bar = tqdm(total=self.cfg.total_eval_tokens)
        for batch in data_stream:
            if not self.cfg.fold_activation_scale:
                batch = sae.normalize_activations(batch)
            if not self.cfg.use_cached_activations:
                assert model is not None, "Model is required for token input"
                assert isinstance(sae, SparseAutoEncoder), "Must be a SparseAutoEncoder for token input"
                activation_dict, loss_reconstruction_dict, useful_token_mask = self._evaluate_tokens(sae, batch, model)
                log_info.update(loss_reconstruction_dict)
            else:
                activation_dict = batch
                useful_token_mask: torch.Tensor = torch.ones(
                    batch["tokens"].shape[0], device=sae.cfg.device, dtype=torch.bool
                )
            self._evaluate_activations(sae, log_info, activation_dict, useful_token_mask)
            proc_bar.update(item(useful_token_mask.sum()))
            self.cur_tokens += item(useful_token_mask.sum())
            self.cur_step += 1
            if self.cur_tokens > self.cfg.total_eval_tokens:
                break
        self.process_metrics(wandb_logger)


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


class GrahEval:
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
        show: bool = False,
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

            if show:
                print("prompt:", prompt)
                print(f"complete: {completeness_score}")
                print(f"replace: {replacement_score}")
