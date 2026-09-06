from typing import Literal

import torch
from torch import nn

from .settings import ModelSettings


class ConvolutionalModel(nn.Module):
    """
    Main model to classify the Cistercian numeral images to extract the digits.
    This uses a simple Convolutional Network architecture.
    """

    def __init__(self, settings: ModelSettings):
        super().__init__()
        self.dtype = get_dtype(settings.dtype)
        self.conv_block = self._make_conv_block(settings, self.dtype)
        self.lin_block = self._make_linear_block(settings, self.dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass on the data and return the logits for the
        digits in the images. Output is shaped to work well with torch's
        CrossEntropy loss for multidimensional outputs.

        Input dimension: (batch size, channels, img_size, img_size)
        Output dimension: (batch size, 10, 4)
        """
        x = self.conv_block(inputs)
        x = self.lin_block(x)
        x = torch.reshape(x, (x.shape[0], 10, 4))
        return x

    def get_digits(
        self, inputs: torch.Tensor, probabilities: bool = False
    ) -> torch.Tensor:
        """
        Perform prediction on the given inputs. Classify the digits and return
        their predicted labels or probabilities.

        Input dimension: (batch size, channels, img_size, img_size)
        Output dimension: (batch size, 4) if probabilities is false, otherwise
            (batch size, 10, 4)
        """
        logits = self.forward(inputs)
        probs = torch.softmax(logits, dim=1)
        if probabilities:
            return probs

        labels = probs.argmax(dim=1).to(torch.uint32)
        return labels

    @staticmethod
    def _make_conv_block(
        settings: ModelSettings, dtype: torch.dtype
    ) -> nn.Module:
        """
        Instantiate the initial convolutional layers for the model.
        """
        layers = []
        for lyr_cfg in settings.conv_layers:
            padding = 0
            if lyr_cfg.add_padding:
                padding = (
                    lyr_cfg.kernel_size[0] // 2,
                    lyr_cfg.kernel_size[1] // 2,
                )
            layers.append(
                nn.Conv2d(
                    in_channels=lyr_cfg.input_channels,
                    out_channels=lyr_cfg.output_channels,
                    kernel_size=lyr_cfg.kernel_size,
                    padding=padding,
                    dtype=dtype,
                )
            )

            match lyr_cfg.activation:
                case "linear":
                    pass
                case "relu":
                    layers.append(nn.ReLU())
                case "gelu":
                    layers.append(nn.GELU())
                case "tanh":
                    layers.append(nn.Tanh())
                case "sigmoid":
                    layers.append(nn.Sigmoid())

            layers.append(
                nn.MaxPool2d(
                    kernel_size=lyr_cfg.pool_kernel_size,
                    stride=lyr_cfg.pool_kernel_size,
                )
            )

            if lyr_cfg.add_dropout:
                layers.append(nn.Dropout(p=settings.dropout))

        layers.append(nn.Flatten())
        return nn.Sequential(*layers)

    @staticmethod
    def _make_linear_block(
        settings: ModelSettings, dtype: torch.dtype
    ) -> nn.Module:
        """
        Make the final block of linear Fully Connected layers to serve as the
        model's classification head after the convolutional layers.
        """
        if settings.linear_layers[-1].output_dim != 40:
            # We have 4 digits to classify among 10 options
            raise ValueError("Output dimension of the model must be 40!")

        layers = []
        for i, lyr_cfg in enumerate(settings.linear_layers):
            layers.append(
                nn.Linear(
                    in_features=lyr_cfg.input_dim,
                    out_features=lyr_cfg.output_dim,
                    dtype=dtype,
                )
            )

            match lyr_cfg.activation:
                case "linear":
                    pass
                case "relu":
                    layers.append(nn.ReLU())
                case "gelu":
                    layers.append(nn.GELU())
                case "tanh":
                    layers.append(nn.Tanh())
                case "sigmoid":
                    layers.append(nn.Sigmoid())

            if lyr_cfg.normalise:
                layers.append(nn.LayerNorm(normalized_shape=lyr_cfg.output_dim))

            if lyr_cfg.add_dropout and i < (len(settings.linear_layers) - 1):
                # Do not add dropout to head
                layers.append(nn.Dropout(p=settings.dropout))

        return nn.Sequential(*layers)


def get_dtype(
    name: Literal["float16", "bfloat16", "float32", "float64"],
) -> torch.dtype:
    """
    Get PyTorch dtype object from its name.
    """
    match name:
        case "float16":
            return torch.float16
        case "bfloat16":
            return torch.bfloat16
        case "float32":
            return torch.float32
        case "float64":
            return torch.float64
        case _:
            raise ValueError(f"Unknown data type given: {name}")
