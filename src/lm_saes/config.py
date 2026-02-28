from typing import Annotated

import torch
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    WithJsonSchema,
)

from lm_saes.utils.misc import (
    convert_str_to_torch_dtype,
    convert_torch_dtype_to_str,
)


class BaseConfig(BaseModel):
    pass


class BaseModelConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # allow parsing torch.dtype

    device: str = Field(default="cpu", exclude=True)
    """ The device to use for the model. """

    dtype: Annotated[
        torch.dtype,
        BeforeValidator(lambda v: convert_str_to_torch_dtype(v) if isinstance(v, str) else v),
        PlainSerializer(convert_torch_dtype_to_str),
        WithJsonSchema(
            {
                "type": "string",
            },
            mode="serialization",
        ),
    ] = Field(default=torch.float32, exclude=True, validate_default=False)
    """ The dtype of weights of the model. """


class DatasetConfig(BaseConfig):
    dataset_name_or_path: str = "openwebtext"
    """ The name or path to the dataset. Should be a valid dataset name or path for `datasets.load_dataset` or `datasets.load_from_disk`, depending on `is_dataset_on_disk`."""

    cache_dir: str | None = None
    """ The directory to cache the dataset. Will be passed to `datasets.load_dataset`."""

    is_dataset_on_disk: bool = False
    """ Whether the dataset is saved through `datasets.save_to_disk`. If True, the dataset will be loaded through `datasets.load_from_disk`. Otherwise, it will be loaded through `datasets.load_dataset`."""
