import argparse
from enum import Enum

import torch


class SupportedRunner(Enum):
    TRAIN = "train"
    EVAL = "eval"
    ANALYZE = "analyze"
    PRUNE = "prune"
    GENERATE_ACTIVATIONS = "gen-activations"

    def __str__(self):
        return self.value


def entrypoint():
    parser = argparse.ArgumentParser(description="Launch runners from given configuration.")
    parser.add_argument(
        "runner",
        type=SupportedRunner,
        help=f'The runner to launch. Supported runners: {", ".join([str(runner) for runner in SupportedRunner])}.',
        choices=list(SupportedRunner),
        metavar="runner",
    )
    parser.add_argument("config", type=str, help="The configuration to use.")
    parser.add_argument("--sae", type=str, help="The path to the pretrained SAE model.")
    args = parser.parse_args()

    config_file: str = args.config
    if config_file.endswith(".json"):
        import json

        with open(config_file, "r") as f:
            config = json.load(f)
    elif config_file.endswith(".yaml") or config_file.endswith(".yml"):
        import yaml

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    elif config_file.endswith(".toml"):
        import tomlkit

        with open(config_file, "r") as f:
            config = tomlkit.load(f).unwrap()
    elif config_file.endswith(".py"):
        import importlib.util

        spec = importlib.util.spec_from_file_location("__lm_sae_config__", config_file)
        assert spec is not None and spec.loader is not None, f"Failed to load configuration file: {config_file}."
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.config
    else:
        raise ValueError(
            f"Unsupported configuration file format: {config_file}. Supported formats: json, yaml, toml, py."
        )

    if args.sae is not None:
        from lm_saes.config import SAEConfig

        sae_config = SAEConfig.from_pretrained(args.sae).to_dict()
        if "sae" in config:
            sae_config.update(config["sae"])
        config["sae"] = sae_config
    print(config)

    tp_size = config.get("tp_size", 1)
    ddp_size = config.get("ddp_size", 1)
    if tp_size > 1 or ddp_size > 1:
        import os

        import torch.distributed as dist

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        dist.init_process_group(backend="nccl")
        print(f"Setting device to {dist.get_rank()}")
        torch.cuda.set_device(dist.get_rank())

    if args.runner == SupportedRunner.TRAIN:
        from lm_saes.config import LanguageModelSAETrainingConfig
        from lm_saes.runner import language_model_sae_runner

        config = LanguageModelSAETrainingConfig.from_flattened(config)
        language_model_sae_runner(config)
    elif args.runner == SupportedRunner.EVAL:
        from lm_saes.config import LanguageModelSAERunnerConfig
        from lm_saes.runner import language_model_sae_eval_runner

        config = LanguageModelSAERunnerConfig.from_flattened(config)
        language_model_sae_eval_runner(config)
    elif args.runner == SupportedRunner.ANALYZE:
        from lm_saes.config import LanguageModelSAEAnalysisConfig
        from lm_saes.runner import sample_feature_activations_runner

        config = LanguageModelSAEAnalysisConfig.from_flattened(config)
        sample_feature_activations_runner(config)
    elif args.runner == SupportedRunner.PRUNE:
        from lm_saes.config import LanguageModelSAEPruningConfig
        from lm_saes.runner import language_model_sae_prune_runner

        config = LanguageModelSAEPruningConfig.from_flattened(config)
        language_model_sae_prune_runner(config)
    elif args.runner == SupportedRunner.GENERATE_ACTIVATIONS:
        from lm_saes.config import ActivationGenerationConfig
        from lm_saes.runner import activation_generation_runner

        config = ActivationGenerationConfig.from_flattened(config)
        activation_generation_runner(config)
    else:
        raise ValueError(f"Unsupported runner: {args.runner}.")

    if tp_size > 1 or ddp_size > 1:
        if dist.is_initialized():  # type: ignore
            dist.destroy_process_group()  # type: ignore
