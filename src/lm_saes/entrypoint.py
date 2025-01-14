from __future__ import annotations

import argparse
import enum
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable

import tomlkit
import torch
import yaml
from pydantic_settings import CliApp, CliSettingsSource


class RunnerType(str, enum.Enum):
    """Available runner types."""

    GENERATE = "gen-activations"
    TRAIN = "train"
    ANALYZE = "analyze"
    CREATE = "create"
    REMOVE = "remove"


class CreateType(str, enum.Enum):
    """Available creation types."""

    DATASET = "dataset"
    MODEL = "model"
    SAE = "sae"


class RemoveType(str, enum.Enum):
    """Available removal types."""

    SAE = "sae"
    ANALYSIS = "analysis"


def setup_generate_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Setup parser for activation generation.

    Args:
        subparsers: Subparser action to add the generate parser to.
    """
    gen_parser = subparsers.add_parser("gen-activations", help="Generate activations from a model")
    gen_parser.add_argument("config", type=str, help="The GenerateActivationsSettings configuration file to use")


def setup_train_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Setup parser for model training.

    Args:
        subparsers: Subparser action to add the train parser to.
    """
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("config", type=str, help="The TrainSAESettings configuration file to use")


def setup_analyze_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Setup parser for model analysis.

    Args:
        subparsers: Subparser action to add the analyze parser to.
    """
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a model")
    analyze_parser.add_argument("config", type=str, help="The AnalyzeSAESettings configuration file to use")


def setup_create_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Setup parser for database record creation.

    Args:
        subparsers: Subparser action to add the create parser to.
    """
    create_parser = subparsers.add_parser("create", help="Create a database record")
    create_parser.add_argument(
        "--mongo-uri",
        type=str,
        help="The MongoDB URI to use",
        default=os.environ.get("MONGO_URI", "mongodb://localhost:27017/"),
    )
    create_parser.add_argument(
        "--mongo-db", type=str, help="The MongoDB database to use", default=os.environ.get("MONGO_DB", "mechinterp")
    )
    create_subparsers = create_parser.add_subparsers(
        dest="type", required=True, help="The type of database record to create"
    )

    # Dataset creation parser
    create_dataset_parser = create_subparsers.add_parser("dataset", help="Create a dataset record")
    create_dataset_parser.add_argument("name", type=str, help="The name of the dataset")
    create_dataset_parser.add_argument("config", type=str, help="The DatasetConfig to save")

    # Model creation parser
    create_model_parser = create_subparsers.add_parser("model", help="Create a model record")
    create_model_parser.add_argument("name", type=str, help="The name of the model")
    create_model_parser.add_argument("config", type=str, help="The ModelConfig to save")

    # SAE creation parser
    create_sae_parser = create_subparsers.add_parser("sae", help="Create an SAE record")
    create_sae_parser.add_argument("name", type=str, help="The name of the SAE")
    create_sae_parser.add_argument("series", type=str, help="The series of the SAE")
    create_sae_parser.add_argument("path", type=str, help="The path to the pretrained SAE directory")


def setup_remove_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Setup parser for database record removal.

    Args:
        subparsers: Subparser action to add the remove parser to.
    """
    remove_parser = subparsers.add_parser("remove", help="Remove a database record")
    remove_parser.add_argument(
        "--mongo-uri",
        type=str,
        help="The MongoDB URI to use",
        default=os.environ.get("MONGO_URI", "mongodb://localhost:27017/"),
    )
    remove_parser.add_argument(
        "--mongo-db", type=str, help="The MongoDB database to use", default=os.environ.get("MONGO_DB", "mechinterp")
    )
    remove_subparsers = remove_parser.add_subparsers(
        dest="type", required=True, help="The type of database record to remove"
    )

    # SAE removal parser
    remove_sae_parser = remove_subparsers.add_parser(
        "sae",
        help="Remove an SAE record. The corresponding features and analyses will also be removed.",
    )
    remove_sae_parser.add_argument("name", type=str, help="The name of the SAE to remove")
    remove_sae_parser.add_argument("series", type=str, help="The series of the SAE to remove")

    # Analysis removal parser
    remove_analysis_parser = remove_subparsers.add_parser("analysis", help="Remove an analysis record")
    remove_analysis_parser.add_argument("sae_name", type=str, help="The name of the SAE to remove")
    remove_analysis_parser.add_argument("sae_series", type=str, help="The series of the SAE to remove")
    remove_analysis_parser.add_argument("name", type=str, help="The name of the analysis to remove")


def handle_runner(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Handle different runner types based on command line arguments.

    Args:
        args: Parsed command line arguments
        parser: Main argument parser for CLI settings

    Raises:
        ValueError: If the runner type is not supported
    """
    runner_type = RunnerType(args.runner)

    if runner_type == RunnerType.GENERATE:
        from lm_saes.runner import GenerateActivationsSettings, generate_activations

        _run_with_config(args, parser, GenerateActivationsSettings, generate_activations)

    elif runner_type == RunnerType.TRAIN:
        from lm_saes.runner import TrainSAESettings, train_sae

        _run_with_config(args, parser, TrainSAESettings, train_sae)

    elif runner_type == RunnerType.ANALYZE:
        from lm_saes.runner import AnalyzeSAESettings, analyze_sae

        _run_with_config(args, parser, AnalyzeSAESettings, analyze_sae)

    elif runner_type == RunnerType.CREATE:
        from lm_saes.config import MongoDBConfig
        from lm_saes.database import MongoClient

        client = MongoClient(
            MongoDBConfig(
                mongo_uri=args.mongo_uri,
                mongo_db=args.mongo_db,
            )
        )

        if args.type == CreateType.DATASET:
            from lm_saes.config import DatasetConfig

            config = _load_config(args.config)
            client.add_dataset(name=args.name, cfg=DatasetConfig.model_validate(config))

        elif args.type == CreateType.MODEL:
            from lm_saes.config import LanguageModelConfig

            config = _load_config(args.config)
            client.add_model(name=args.name, cfg=LanguageModelConfig.model_validate(config))

        elif args.type == CreateType.SAE:
            from lm_saes.config import SAEConfig

            client.create_sae(
                name=args.name,
                series=args.series,
                path=args.path,
                cfg=SAEConfig.from_pretrained(args.path),
            )

        else:
            raise ValueError(f"Unsupported create type: {args.type}")

    elif runner_type == RunnerType.REMOVE:
        from lm_saes.config import MongoDBConfig
        from lm_saes.database import MongoClient

        client = MongoClient(
            MongoDBConfig(
                mongo_uri=args.mongo_uri,
                mongo_db=args.mongo_db,
            )
        )

        if args.type == RemoveType.SAE:
            client.remove_sae(sae_name=args.name, sae_series=args.series)

        elif args.type == RemoveType.ANALYSIS:
            client.remove_feature_analysis(name=args.name, sae_name=args.sae_name, sae_series=args.sae_series)

        else:
            raise ValueError(f"Unsupported remove type: {args.type}")

    else:
        raise ValueError(f"Unsupported runner: {runner_type}")


def _run_with_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    settings_cls: type,
    runner_fn: Callable,
) -> None:
    """Run a function with configuration loaded from file and CLI.

    Args:
        args: Parsed command line arguments
        parser: Main argument parser for CLI settings
        settings_cls: Pydantic settings class to use
        runner_fn: Function to run with the configuration
    """
    config = _load_config(args.config)
    cli_settings = CliSettingsSource(settings_cls, root_parser=parser)
    config = CliApp.run(settings_cls, cli_settings_source=cli_settings, **config)
    runner_fn(config)


def _load_config(config_file: str | Path) -> dict[str, Any]:
    """Load configuration from a file.

    Args:
        config_file: Path to the configuration file.

    Returns:
        The loaded configuration as a dictionary.

    Raises:
        ValueError: If the file format is not supported.
        FileNotFoundError: If the configuration file doesn't exist.
    """
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    if config_path.suffix == ".json":
        with open(config_path) as f:
            return json.load(f)

    elif config_path.suffix in {".yaml", ".yml"}:
        with open(config_path) as f:
            return yaml.safe_load(f)

    elif config_path.suffix == ".toml":
        with open(config_path) as f:
            return tomlkit.load(f).unwrap()

    elif config_path.suffix == ".py":
        spec = importlib.util.spec_from_file_location("__lm_sae_config__", config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load configuration file: {config_file}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.config

    raise ValueError(
        f"Unsupported configuration file format: {config_path.suffix}. "
        "Supported formats: .json, .yaml, .yml, .toml, .py"
    )


def entrypoint():
    """Main entrypoint for the CLI application."""
    parser = argparse.ArgumentParser(description="Launch runners from given configuration.")
    subparsers = parser.add_subparsers(dest="runner", required=True, help="The runner to launch")

    # Setup parsers for different commands
    setup_generate_parser(subparsers)
    setup_train_parser(subparsers)
    setup_analyze_parser(subparsers)
    setup_create_parser(subparsers)

    # Parse args and handle the selected runner
    args, _ = parser.parse_known_args()
    handle_runner(args, parser)

    # Cleanup distributed process group if initialized
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
