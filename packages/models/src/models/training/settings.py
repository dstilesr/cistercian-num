from pathlib import Path

from data_utils.torch_dataset import PerturbationConfig
from pydantic import BaseModel, Field

from ..constants import DEFAULT_DATA_PATH


class TracingSettings(BaseModel):
    """
    Configuration for experiment tracking.
    """

    experiment_name: str = "cistercian-nums"
    mlflow_path: Path = (
        Path(__file__).parents[5] / "data" / "mlflow" / "mlflow.db"
    )


class TrainingSettings(BaseModel):
    """
    Settings for model training
    """

    tracking_cfg: TracingSettings = Field(default_factory=TracingSettings)
    learning_rate: float = Field(default=0.01, gt=0.0)
    epochs: int = Field(default=12, gt=0)
    save_to: Path = DEFAULT_DATA_PATH / "checkpoints"
    dataset_path: Path = DEFAULT_DATA_PATH
    image_size: int = Field(gt=0, default=128)
    batch_size: int = Field(gt=0, default=32)
    data_perturbation: PerturbationConfig = Field(
        default_factory=PerturbationConfig
    )
