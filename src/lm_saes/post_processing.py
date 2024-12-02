import torch

from .activation.activation_store import ActivationStore
from .config import LanguageModelSAERunnerConfig, SAEConfig
from .sae import SparseAutoEncoder


@torch.no_grad()
def post_process_topk_to_jumprelu_for_inference(
    sae: SparseAutoEncoder,
    activation_store: ActivationStore,
    cfg: LanguageModelSAERunnerConfig,
):
    batch = activation_store.next(batch_size=32768)
    assert batch is not None, "Activation store is empty"

    activation_in, _ = (
        batch[sae.cfg.hook_point_in],
        batch[sae.cfg.hook_point_out],
    )

    _, hidden_pre = sae.encode(activation_in, return_hidden_pre=True)
    hidden_pre = torch.clamp(hidden_pre, min=0.0)
    hidden_pre = hidden_pre.flatten()

    threshold = hidden_pre.topk(k=32768 * sae.cfg.top_k).values[-1]

    original_hyperparams = SAEConfig.from_pretrained(cfg.exp_result_path)
    original_hyperparams.act_fn = "jumprelu"
    original_hyperparams.jump_relu_threshold = threshold.item()
    original_hyperparams.save_hyperparameters(cfg.exp_result_path)

    return threshold
