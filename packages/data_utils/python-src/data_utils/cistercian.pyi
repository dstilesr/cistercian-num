import numpy as np

def make_image(
    size: int, line_thickness: int, margin: int, number: int
) -> np.ndarray: ...
def train_test_numbers(
    test_proportion: float,
) -> tuple[list[int], list[int]]: ...
def generate_and_save_image(
    size: int, line_thickness: int, margin: int, number: int, filepath: str
) -> None: ...
