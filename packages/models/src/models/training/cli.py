import logging
import sys
from pathlib import Path

from typer import Typer

from .. import convolutional as conv
from ..training import run_training
from .settings import TrainingSettings

app = Typer(name="Model Training")


def logging_setup(level: int = logging.INFO):
    """
    Basic logging setup
    """
    logging.basicConfig(
        level=level,
        format="[%(asctime)s][%(levelname)10s][%(name)s] %(message)s",
        stream=sys.stderr,
    )


@app.command("train-conv")
def train_convolutional(model_cfg: Path, train_cfg: Path | None = None):
    """
    Train a Convolutional-type model for the digit classification task.
    """
    if train_cfg:
        with train_cfg.open("r") as f:
            train_settings = TrainingSettings.model_validate_json(f.read())
    else:
        # Use defaults if not given
        train_settings = TrainingSettings()

    with model_cfg.open("r") as f:
        model_settings = conv.ModelSettings.model_validate_json(f.read())

    model = conv.ConvolutionalModel(model_settings)
    run_training(model, train_settings)
