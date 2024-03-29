import torch
from transformer_lens import HookedTransformer

from datasets import Dataset

from core.sae import SparseAutoEncoder
from core.config import FeaturesDecoderConfig

from core.utils.misc import check_file_path_unused

@torch.no_grad()
def features_to_logits(sae: SparseAutoEncoder, model: HookedTransformer, cfg: FeaturesDecoderConfig):

    num_ones = int(torch.sum(sae.feature_act_mask).item())

    feature_acts = torch.zeros(num_ones, 24576).to(cfg.device)

    index = 0
    for i in range(len(sae.feature_act_mask)):
        if sae.feature_act_mask[i] == 1:
            feature_acts[index, i] = 1
            index += 1

    feature_acts = torch.unsqueeze(feature_acts, dim=1)
    
    # print(feature_acts.shape)
    residual = sae.features_decoder(feature_acts)
    # print(residual.shape)
    
    if model.cfg.normalization_type is not None:
        residual = model.ln_final(residual)  # [batch, pos, d_model]
    logits = model.unembed(residual)  # [batch, pos, d_vocab]

    # print(logits.shape)
    active_indices = [i for i, val in enumerate(sae.feature_act_mask) if val == 1]
    # print(len(active_indices))
    result_dict = {str(feature_index): logits[idx][0] for idx, feature_index in enumerate(active_indices)}
    # print(result_dict)
    
    check_file_path_unused(cfg.file_path)
    
    Dataset.from_dict(result_dict).save_to_disk(cfg.file_path, num_shards=1024)
    # Dataset.from_dict(result_dict).save_to_disk(cfg.file_path, num_shards=1024, progress_callback=progress_callback)
    # Key.Type: <class 'int'> Value.Type: <class 'numpy.ndarray'>

    # print(model.cfg.normalization_type)
    # LNPre
    # print(model.cfg.final_rms)
    # False
    
    # print(mid.shape)
    # torch.Size([768])

    # print(state_dict['unembed.W_U'].shape)
    # torch.Size([768, 50257])
    # print(state_dict['unembed.b_U'].shape)
    # torch.Size([50257])
    # print(hf_model.state_dict()['transformer.ln_f.weight'].shape)
    # torch.Size([768])
    # print(hf_model.state_dict()['transformer.ln_f.bias'].shape)
    # torch.Size([768])
    # print(hf_model.state_dict()['lm_head.weight'].shape)
    # torch.Size([50257, 768])
    
def progress_callback(current, total):
    print(f"Progress: {current}/{total}")