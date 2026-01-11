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


class DatasetConfig(BaseConfig):
    dataset_name_or_path: str = "openwebtext"
    cache_dir: str | None = None
    is_dataset_on_disk: bool = False
