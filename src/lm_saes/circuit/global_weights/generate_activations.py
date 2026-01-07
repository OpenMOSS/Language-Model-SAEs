from lm_saes import LanguageModelConfig, LowRankSparseAttention, SparseAutoEncoder, MongoClient, MongoDBConfig
from lm_saes.analysis.global_weights.forward_only_replacement_model import ForwardOnlyReplacementModel
from lm_saes.circuit.utils.transcoder_set import TranscoderSet, TranscoderSetConfig
import torch
from datasets import load_from_disk
import os
from tqdm import tqdm
from jaxtyping import Float, Bool
from exp.global_weights.batch_saver import SampleSaver


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

model.cfg.n_ctx = 512

output_dir = "/inspire/hdd/project/reasoning/public/activations/temp/pythia-160m-global-weight-batches"
sample_saver = SampleSaver(output_dir=output_dir)

# Load dataset from disk
dataset_path = os.path.expanduser(
    "/inspire/hdd/global_user/hezhengfu-240208120186/data/pretrain-data/SlimPajama-3B"
)
dataset = load_from_disk(dataset_path)
dataset = dataset.select(range(int(0.001 * dataset.num_rows)))
# Iterate through the dataset and process each text
for row in tqdm(dataset):
    text = row['text']
    # Tokenize to get input_ids
    tokenized = model.tokenizer(text, return_tensors="pt").input_ids.to(model.cfg.device)
    input_ids = tokenized.squeeze(0)
    input_ids = input_ids[:model.cfg.n_ctx]
    
    # Forward the text and get activation matrices
    lorsa_activation_matrix, lorsa_attention_pattern, clt_activation_matrix, attn_hook_scales, mlp_hook_scales = model.setup_attribution(
        text,
        sparse=True,
        zero_bos=True,
    )
    # Save sample
    saved_path = sample_saver.save_sample(
        lorsa_activation_matrix=lorsa_activation_matrix,
        lorsa_attention_pattern=lorsa_attention_pattern,
        clt_activation_matrix=clt_activation_matrix,
        attn_hook_scales=attn_hook_scales,
        mlp_hook_scales=mlp_hook_scales,
        input_ids=input_ids,
    )