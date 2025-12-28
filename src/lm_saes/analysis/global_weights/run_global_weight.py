from lm_saes import LanguageModelConfig, LowRankSparseAttention, SparseAutoEncoder, MongoClient, MongoDBConfig
from lm_saes.analysis.global_weights.forward_only_replacement_model import ForwardOnlyReplacementModel
from lm_saes.circuit.utils.transcoder_set import TranscoderSet, TranscoderSetConfig
import torch
from datasets import load_from_disk
import os
from tqdm import tqdm
from jaxtyping import Float, Bool
from exp.global_weights.batch_saver import create_dataloader
from lm_saes.analysis.global_weights.batched_features import BatchedFeatures
from lm_saes.analysis.global_weights.atlas import Atlas
import json

prompt = "<|endoftext|>I always loved visiting Aunt Sally. Whenever I was feeling sad, Aunt"

sae_series = "pythia-test"
device = "cuda"
model_name = "/inspire/hdd/global_user/hezhengfu-240208120186/models/pythia-160m"
n_layers = 12


model_cfg = LanguageModelConfig(
    model_name="EleutherAI/pythia-160m",
    device=device,
    dtype="torch.float32",
    model_from_pretrained_path=model_name,
)

transcoders = {
    i: SparseAutoEncoder.from_pretrained(
        f'/inspire/hdd/global_user/hezhengfu-240208120186/wtshu_project/llama2-scope/train/pythia-test/plt-layer{i}',
        device=device
    )
    for i in range(n_layers)
}

plt_set = TranscoderSet(
    TranscoderSetConfig(
        n_layers=n_layers,
        d_sae=transcoders[0].cfg.d_sae,
        feature_input_hook="ln2.hook_normalized",
        feature_output_hook="hook_mlp_out"
    ),
    transcoders,
)

lorsas = [
    LowRankSparseAttention.from_pretrained(
        f'/inspire/hdd/global_user/hezhengfu-240208120186/wtshu_project/llama2-scope/train/pythia-test/lorsa-layer{i}',
        device=device
    )
    for i in range(n_layers)
]

model = ForwardOnlyReplacementModel.from_pretrained(
    model_cfg=model_cfg,
    transcoders=plt_set,
    lorsas=lorsas,
)

name = 'closing_bracket'
initial_features = BatchedFeatures(
    layer=torch.tensor([7], device='cuda', dtype=torch.int32),
    index=torch.tensor([4941], device='cuda', dtype=torch.int32),
    is_lorsa=torch.tensor([False], device='cuda', dtype=torch.bool),
)

atlas = Atlas(initial_features)

feature_to_explore = initial_features

for i in range(2):
    model.prepare_virtual_weights(feature_to_explore)

    loader = create_dataloader(
        "/inspire/hdd/project/reasoning/public/activations/temp/pythia-160m-global-weight-batches",
        num_workers=8,
        prefetch_factor=2,
        pin_memory=False,
    )

    for sample in tqdm(loader, desc=f"Iteration {i}"):    
        lorsa_activation_matrix = sample['lorsa_activation_matrix'].cuda()
        lorsa_attention_pattern = sample['lorsa_attention_pattern'].cuda()
        clt_activation_matrix = sample['clt_activation_matrix'].cuda()
        attn_hook_scales = sample['attn_hook_scales'].cuda()
        mlp_hook_scales = sample['mlp_hook_scales'].cuda()
        input_ids = sample['input_ids'].cuda()

        # Save sample
        model.record_activation_matrix(
            input_ids,
            lorsa_activation_matrix,
            clt_activation_matrix,
            lorsa_attention_pattern,
            attn_hook_scales,
            mlp_hook_scales
        )

    connected_features = model.parse_global_weight_results(5)
    feature_to_explore = atlas.update(feature_to_explore, connected_features)[:20]

    save_dir = f"atlases/{name}"
    os.makedirs(save_dir, exist_ok=True)
    json.dump(atlas.export_to_json(), open(f"{save_dir}/atlas_{i}.json", "w"))
