from pathlib import Path
from random import shuffle

from .cistercian import make_image, train_test_numbers


def split_numbers(
    test_proportion: float = 0.075, save_to: Path | None = None
) -> tuple[list[int], list[int]]:
    """
    Generate a list of numbers for training and a list for testing. These
    will be split from the numbers from 1 to 9999. This can be done since the
    objective is to classify the 4 digits in the glyph instead of the whole
    number at once.
    :param test_proportion:
    :param save_to: If given, save to this directory.
    """
    train, test = train_test_numbers(test_proportion)

    if not save_to:
        return train, test
    elif save_to.is_dir():
        pass
    elif save_to.exists():
        raise FileExistsError(
            "save_to should point to a directory. Found a file."
        )
    else:
        save_to.mkdir(parents=True, exist_ok=True)

    train_path = save_to / "train.txt"
    test_path = save_to / "test.txt"

    train_path.write_text("\n".join(map(str, train)) + "\n")
    test_path.write_text("\n".join(map(str, test)) + "\n")
    return train, test


def train_dataset():
    pass
