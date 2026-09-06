from pathlib import Path

from pydantic import BaseModel, Field

from ..constants import DEFAULT_DATA_PATH


class TrainingSettings(BaseModel):
    """
    Settings for model training
    """

    learning_rate: float = Field(default=0.01, gt=0.0)
    epochs: int = Field(default=12, gt=0)
    save_to: Path = DEFAULT_DATA_PATH / "checkpoints"
    dataset_path: Path = DEFAULT_DATA_PATH
    image_size: int = Field(gt=0, default=128)
    batch_size: int = Field(gt=0, default=32)
