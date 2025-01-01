import argparse
from enum import Enum

import torch
from pydantic_settings import CliApp, CliSettingsSource


class SupportedRunner(Enum):
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
    args, _ = parser.parse_known_args()

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

    if args.runner == SupportedRunner.GENERATE_ACTIVATIONS:
        from lm_saes.runner import GenerateActivationsSettings, generate_activations

        cli_settings = CliSettingsSource(GenerateActivationsSettings, root_parser=parser)
        config = CliApp.run(GenerateActivationsSettings, cli_settings_source=cli_settings, **config)
        generate_activations(config)
    else:
        raise ValueError(f"Unsupported runner: {args.runner}.")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
