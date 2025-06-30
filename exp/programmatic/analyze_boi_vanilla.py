import os

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from lm_saes import (
    ActivationFactoryActivationsSource,
    ActivationFactoryConfig,
    ActivationFactoryTarget,
    AnalyzeSAESettings,
    CrossCoderConfig,
    FeatureAnalyzerConfig,
    MongoDBConfig,
    analyze_sae,
)

dist.init_process_group(backend="nccl")
torch.cuda.set_device(f'cuda:{os.environ["LOCAL_RANK"]}')

l = 15
lr = 5e-05
l1_coefficient = 1.5
expfactor = 8
# shared_decoder_sparsity_factor = 0.1

models = {
    "b": "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs/exp/diff_models/Llama-3.1-8B",
    "i": "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs/exp/diff_models/Llama-3.1-8B-Instruct",
    "o": "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs/exp/diff_models/DeepSeek-R1-Distill-Llama-8B",
}


rank = dist.get_rank()
modelname_dict = {0: "base_llama", 1: "Instruct_llama", 2: "o_llama"}
subject_model = modelname_dict.get(rank)

bio_id_dict = {0: "b", 1: "i", 2: "o"}
bio_id = bio_id_dict.get(rank)

tokenizer = AutoTokenizer.from_pretrained(models[bio_id])
ignore_token_ids = [
    tokenizer.bos_token_id,
    tokenizer.eos_token_id,
    tokenizer.pad_token_id,
]

cc_path = f"/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs/exp/JR_CC_tests_shared/vanilla/{subject_model}_l{l}_{expfactor}x_lr{lr}_jumprelu_l1coef{l1_coefficient}_p1r1_fulltokens"

path_dict = {
    0: "/inspire/hdd/global_user/hezhengfu-240208120186/jiaxing_activations/reasondata-b-2d-801M-f",
    1: "/inspire/hdd/global_user/hezhengfu-240208120186/jiaxing_activations/reasondata-i-2d-801M-f",
    2: "/inspire/hdd/global_user/hezhengfu-240208120186/jiaxing_activations/reasondata-o-2d-801M-f",
}
p_path_dict = {
    0: "/inspire/hdd/global_user/hezhengfu-240208120186/jiaxing_activations/pretraindata-o-2d-801M-ctx8192",
    1: "/inspire/hdd/global_user/hezhengfu-240208120186/jiaxing_activations/pretraindata-o-2d-801M-ctx8192",
    2: "/inspire/hdd/global_user/hezhengfu-240208120186/jiaxing_activations/pretraindata-o-2d-801M-ctx8192",
}
path = path_dict.get(rank)
p_path = p_path_dict.get(rank)

if __name__ == "__main__":
    settings = AnalyzeSAESettings(
        sae=CrossCoderConfig.from_pretrained(
            cc_path,
            device="cuda",
            dtype=torch.float32,
        ),
        analyzer=FeatureAnalyzerConfig(
            total_analyzing_tokens=800_000_000,  ####
            ignore_token_ids=ignore_token_ids,
            batch_size=16,
            enable_sampling=False,
            subsamples={
                "top_activations": {"proportion": 1.0, "n_samples": 16},
                # "top_90_percent": {"proportion": 0.9, "n_samples": 16},
            },
        ),
        sae_name=f"BIO-{subject_model}-l{l}-shared-p1r1",  ##### should have been "vanilla"
        sae_series="BOI-sweeplr",
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryActivationsSource(
                    path=f"{path}",
                    type="activations",
                    name="OpenR1-Math-220k-L15R-800M",
                    device="cuda",
                    dtype=torch.float32,
                    num_workers=4,
                ),
                ActivationFactoryActivationsSource(
                    path=f"{p_path}",
                    type="activations",
                    name="SlimPajama-3B-LXR-800M",
                    device="cuda",
                    dtype=torch.float32,
                    num_workers=4,
                ),
            ],
            target=ActivationFactoryTarget.ACTIVATIONS_2D,
            hook_points=[f"blocks.{l}.hook_resid_post"],
            ignore_token_ids=[id for id in ignore_token_ids if id != None],
        ),
        mongo=MongoDBConfig(mongo_uri="mongodb://10.244.130.39:27017/", mongodb="mechinterp"),
    )
    analyze_sae(settings)
