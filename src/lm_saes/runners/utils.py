"""Utility functions for runners."""

from typing import Literal, Optional, TypeVar, overload

from lm_saes.database import MongoClient
from lm_saes.utils.logging import get_logger

logger = get_logger("runners.utils")

T = TypeVar("T")


@overload
def load_config(
    config: Optional[T],
    name: Optional[str],
    mongo_client: Optional[MongoClient],
    config_type: str,
    required: Literal[True] = True,
) -> T: ...


@overload
def load_config(
    config: Optional[T],
    name: Optional[str],
    mongo_client: Optional[MongoClient],
    config_type: str,
    required: Literal[False] = False,
) -> Optional[T]: ...


def load_config(
    config: Optional[T],
    name: Optional[str],
    mongo_client: Optional[MongoClient],
    config_type: str,
    required: bool = True,
) -> Optional[T]:
    """Load configuration from settings or database.

    Args:
        config: Configuration provided directly in settings
        name: Name of the config to load from database
        mongo_client: Optional MongoDB client for database operations
        config_type: String identifier for error messages ('model' or 'dataset')
        required: Whether the config must be present

    Returns:
        Loaded configuration or None if not required and not found

    Raises:
        AssertionError: If config is required but not found
    """
    if mongo_client is not None and name is not None:
        if config is None:
            config = getattr(mongo_client, f"get_{config_type}_cfg")(name)
            logger.info(f"Loaded {config_type} config from database: {name}")
        else:
            getattr(mongo_client, f"add_{config_type}")(name, config)
            logger.info(f"Added {config_type} config to database: {name}")

    if required:
        assert config is not None, f"{config_type} config not provided and not found in database"
    return config
