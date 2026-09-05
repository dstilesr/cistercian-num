from pathlib import Path

from .cistercian import save_images_batch, train_test_numbers


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


def generate_dataset(
    save_to: Path,
    size: int = 96,
    margin: int = 6,
    line_thickness: int = 3,
    test_proportion: float = 0.075,
):
    """
    Generate dataset for training the model. This is only a starter with
    "prototype" starter images, but it serves to get started with examples.
    :param save_to: Directory to save in.
    :param size: Image width (images will be squares).
    :param margin:
    :param line_thickness:
    :param test_proportion: Proportion of examples to put in test set.
    """
    if save_to.exists():
        if not save_to.is_dir():
            raise FileExistsError(
                f"The path {save_to} existis and is not a directory!"
            )
    else:
        save_to.mkdir(parents=True, exist_ok=True)

    train_nums, test_nums = split_numbers(test_proportion)
    train_dir = save_to / "train"
    train_dir.mkdir(exist_ok=True)
    save_images_batch(
        size=size,
        line_thickness=line_thickness,
        margin=margin,
        numbers=train_nums,
        dir_path=str(train_dir),
    )

    test_dir = save_to / "test"
    test_dir.mkdir(exist_ok=True)
    save_images_batch(
        size=size,
        line_thickness=line_thickness,
        margin=margin,
        numbers=test_nums,
        dir_path=str(test_dir),
    )
