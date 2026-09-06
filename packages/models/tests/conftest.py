from pathlib import Path

import pytest
from models import convolutional as conv

ASSETS_DIR: Path = Path(__file__).parent / "assets"


@pytest.fixture(scope="function")
def model_test_cfg() -> conv.ModelSettings:
    """
    Load a sample model configuration for testing.
    """
    file = ASSETS_DIR / "model-conf.json"
    with file.open("r") as f:
        return conv.ModelSettings.model_validate_json(f.read())
