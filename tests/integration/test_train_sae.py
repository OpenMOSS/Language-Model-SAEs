import os

import pytest
import torch
from pytest_mock import MockerFixture
from torch.distributed.device_mesh import init_device_mesh

# if not torch.cuda.is_available():
#     pytest.skip("CUDA is not available", allow_module_level=True)
from lm_saes.config import InitializerConfig, MixCoderConfig, SAEConfig, TrainerConfig
from lm_saes.initializer import Initializer
from lm_saes.trainer import Trainer


@pytest.fixture
def sae_config() -> SAEConfig:
    return SAEConfig(
        hook_point_in="in",
        hook_point_out="out",
        d_model=2,
        expansion_factor=2,
        device="cpu",
        dtype=torch.bfloat16,  # the precision of bfloat16 is not enough for the tests
        act_fn="topk",
        norm_activation="dataset-wise",
        sparsity_include_decoder_norm=True,
        top_k=2,
    )


@pytest.fixture
def mixcoder_config() -> MixCoderConfig:
    return MixCoderConfig(
        hook_point_in="in",
        hook_point_out="out",
        d_model=2,
        expansion_factor=2,
        device="cpu",
        dtype=torch.bfloat16,  # the precision of bfloat16 is not enough for the tests
        act_fn="topk",
        norm_activation="dataset-wise",
        top_k=2,
        modalities={"image": 2, "text": 2, "shared": 2},
    )


@pytest.fixture
def initializer_config() -> InitializerConfig:
    return InitializerConfig(
        state="training",
        init_search=True,
        l1_coefficient=0.00008,
    )


@pytest.fixture
def trainer_config(tmp_path) -> TrainerConfig:
    # Remove tmp path
    os.rmdir(tmp_path)
    return TrainerConfig(
        initial_k=3,
        total_training_tokens=400,
        log_frequency=10,
        eval_frequency=10,
        n_checkpoints=0,
        exp_result_path=str(tmp_path),
    )


def test_train_sae(
    sae_config: SAEConfig,
    initializer_config: InitializerConfig,
    trainer_config: TrainerConfig,
    mocker: MockerFixture,
    tmp_path,
) -> None:
    wandb_runner = mocker.Mock()
    wandb_runner.log = lambda *args, **kwargs: None
    device_mesh = (
        init_device_mesh(
            device_type="cuda",
            mesh_shape=(int(os.environ.get("WORLD_SIZE", 1)), 1),
            mesh_dim_names=("data", "model"),
        )
        if os.environ.get("WORLD_SIZE") is not None
        else None
    )
    activation_stream = [
        {
            "in": torch.randn(4, 2, dtype=sae_config.dtype, device=sae_config.device),
            "out": torch.randn(4, 2, dtype=sae_config.dtype, device=sae_config.device),
            "tokens": torch.tensor([2, 3, 4, 5], dtype=torch.long, device=sae_config.device),
        }
        for _ in range(200)
    ]
    initializer = Initializer(initializer_config)
    sae = initializer.initialize_sae_from_config(
        sae_config,
        device_mesh=device_mesh,
        activation_stream=activation_stream,
    )
    trainer = Trainer(trainer_config)
    trainer.fit(
        sae=sae,
        activation_stream=activation_stream,
        eval_fn=lambda x: None,
        wandb_logger=wandb_runner,
    )


def test_train_mixcoder(
    mixcoder_config: MixCoderConfig,
    initializer_config: InitializerConfig,
    trainer_config: TrainerConfig,
    mocker: MockerFixture,
    tmp_path,
) -> None:
    wandb_runner = mocker.Mock()
    wandb_runner.log = lambda *args, **kwargs: None
    device_mesh = (
        init_device_mesh(
            device_type="cuda",
            mesh_shape=(int(os.environ.get("WORLD_SIZE", 1)), 1),
            mesh_dim_names=("data", "model"),
        )
        if os.environ.get("WORLD_SIZE") is not None
        else None
    )
    activation_stream = [
        {
            "in": torch.randn(4, 2, dtype=mixcoder_config.dtype, device=mixcoder_config.device),
            "out": torch.randn(4, 2, dtype=mixcoder_config.dtype, device=mixcoder_config.device),
            "tokens": torch.tensor([2, 3, 4, 5], dtype=torch.long, device=mixcoder_config.device),
        }
        for _ in range(200)
    ]
    initializer = Initializer(initializer_config)
    tokenizer = mocker.Mock()
    tokenizer.get_vocab.return_value = {
        "IMGIMG1": 1,
        "IMGIMG2": 2,
        "IMGIMG3": 3,
        "IMGIMG4": 4,
        "TEXT1": 5,
        "TEXT2": 6,
        "TEXT3": 7,
        "TEXT4": 8,
    }
    model_name = "facebook/chameleon-7b"

    mixcoder_settings = {"tokenizer": tokenizer, "model_name": model_name}
    mixcoder = initializer.initialize_sae_from_config(
        mixcoder_config,
        device_mesh=device_mesh,
        activation_stream=activation_stream,
        mixcoder_settings=mixcoder_settings,
    )
    trainer = Trainer(trainer_config)
    trainer.fit(
        sae=mixcoder,
        activation_stream=activation_stream,
        eval_fn=lambda x: None,
        wandb_logger=wandb_runner,
    )
