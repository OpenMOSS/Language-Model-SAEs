import wandb
from tqdm import tqdm
from transformer_lens import HookedTransformer
import torch
from lm_saes.sae import SparseAutoEncoder
from lm_saes.activation.activation_store import ActivationStore

# from lm_saes.activation_store_theirs import ActivationStoreTheirs
from lm_saes.config import LanguageModelSAERunnerConfig, SAEConfig
from lm_saes.utils.misc import is_master

@torch.no_grad()
def post_process_topk_to_jumprelu_for_inference(
    sae: SparseAutoEncoder,
    activation_store: ActivationStore,
    cfg: LanguageModelSAERunnerConfig
):
    batch = activation_store.next(batch_size=32768)
    activation_in, activation_out = (
        batch[sae.cfg.hook_point_in],
        batch[sae.cfg.hook_point_out],
    )

    _, hidden_pre = sae.encode(activation_in, return_hidden_pre=True, during_init=False)
    hidden_pre = torch.clamp(hidden_pre, min=0.0)
    hidden_pre = hidden_pre.flatten()

    threshold = hidden_pre.topk(k=32768 * sae.cfg.top_k).values[-1]
    
    original_hyperparams = SAEConfig.from_pretrained(cfg.exp_result_path)
    original_hyperparams.act_fn = 'jumprelu'
    original_hyperparams.jump_relu_threshold = threshold.item()
    original_hyperparams.save_hyperparameters(cfg.exp_result_path)

    return threshold