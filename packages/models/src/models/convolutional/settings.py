from typing import Literal

from pydantic import BaseModel, Field


class ConvLayerSettings(BaseModel):
    """
    Settings for a convolutional layer within the model.
    """

    input_channels: int = Field(default=1, gt=0)
    output_channels: int = Field(default=4, gt=0)
    kernel_size: tuple[int, int] = (3, 3)
    stride: tuple[int, int] = (1, 1)
    add_padding: bool = True
    add_dropout: bool = True
    normalise: bool = False

    pool_type: Literal["average", "max"] = "max"
    pool_kernel_size: tuple[int, int] = (2, 2)
    pool_stride: tuple[int, int] = (2, 2)

    activation: Literal["linear", "relu", "gelu", "tanh", "sigmoid"] = "relu"


class LinearLayerSettings(BaseModel):
    """
    Settings for a linear fully connected layer.
    """

    input_dim: int = Field(default=64, gt=0)
    output_dim: int = Field(default=64, gt=0)
    add_dropout: bool = True
    normalise: bool = False
    activation: Literal["linear", "relu", "gelu", "tanh", "sigmoid"] = "relu"


class ModelSettings(BaseModel):
    """
    Full settings to instantiate the model.
    """

    dtype: Literal["float16", "bfloat16", "float32", "float64"] = "float32"
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)

    conv_layers: list[ConvLayerSettings]
    linear_layers: list[LinearLayerSettings] = Field(min_length=1)
