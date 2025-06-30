import multiprocessing

import torch

from lm_saes import (
    ActivationFactoryTarget,
    DatasetConfig,
    GenerateActivationsSettings,
    LanguageModelConfig,
    generate_activations,
)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    # def create_parser():
    #     parser = argparse.ArgumentParser(description="nshards and startshard")

    #     parser.add_argument('--nshards', type=int, required=True)
    #     parser.add_argument('--startshard', type=int, required=True)

    #     return parser

    # parser = create_parser()
    # args, unknown = parser.parse_known_args()
    # nshards = args.nshards
    # startshard = args.startshard

    settings = GenerateActivationsSettings(
        model=LanguageModelConfig(
            model_name="meta-llama/Llama-3.1-8B",
            model_from_pretrained_path="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs/exp/diff_models/DeepSeek-R1-Distill-Llama-8B",
            device="cuda",
            dtype="torch.bfloat16",
        ),
        model_name="meta-llama/Llama-3.1-8B",
        dataset=DatasetConfig(
            dataset_name_or_path="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs/exp/diff_datas/OpenR1-Math-220k-o",
            is_dataset_on_disk=True,
        ),
        dataset_name="OpenR1-Math-220k-o",
        hook_points=["blocks.15.hook_resid_post"],
        output_dir="/inspire/hdd/global_user/hezhengfu-240208120186/jiaxing_activations/reasondata-o-2d-801M",
        total_tokens=801_000_000,
        n_shards=16,
        # start_shard=startshard,
        context_size=8192,
        n_samples_per_chunk=None,
        model_batch_size=1,
        num_workers=36,
        target=ActivationFactoryTarget.ACTIVATIONS_2D,
        batch_size=4096,
        buffer_size=0,
    )
    generate_activations(settings)  # runner 都不会污染全局环境，所以可以安全地在一个脚本里运行多个 runner。
    torch.distributed.destroy_process_group()  # 若没有这行无法正确退出 NCCL。
