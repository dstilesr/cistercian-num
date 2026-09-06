import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from numbers import Number
from pathlib import Path
from typing import cast

import torch
import torchvision.io
from torchvision.datasets import VisionDataset
from torchvision.transforms.v2 import (
    Compose,
    GaussianBlur,
    GaussianNoise,
    RandomAffine,
    Resize,
    ToDtype,
    Transform,
)

from data_utils.cistercian import number_to_digits


def get_train_test_datasets(
    root_path: Path,
    image_size: int = 128,
    perturb_cfg: PerturbationConfig | None = None,
    perturb_train: bool = True,
    perturb_test: bool = False,
) -> tuple[CistercianDataset, CistercianDataset]:
    """
    Get the training and test datasets of numeral images.
    :param root_path: Root path of the dataset. Must contain a `train` and a
        `test` subfolder with the images within them. Check `CistercianDataset`
        for the file name format to use within each.
    :param image_size: Size of images to load. Images will be square, so this
        refers to the side length in pixels.
    :param perturb_cfg: Configurations to add perturbations to the data when loading.
    :param perturb_train: Whether to apply the perturbations to the training set.
    :param perturb_test: Whether to apply perturbations to the test set.
    :return: (train dataset, test dataset) pair.
    """
    perturb_cfg = perturb_cfg or PerturbationConfig()
    train = CistercianDataset(
        root_path / "train",
        transform=perturb_cfg.make_transform() if perturb_train else None,
        image_size=image_size,
    )
    test = CistercianDataset(
        root_path / "test",
        transform=perturb_cfg.make_transform() if perturb_test else None,
        image_size=image_size,
    )
    return train, test


@dataclass(slots=True, frozen=True, kw_only=True)
class PerturbationConfig:
    """
    Configuration to create perturbations to add noise to images while
    training.
    """

    add_gaussian_noise: bool = True
    noise_std_dev: float = 0.1
    noise_mean: float = 0.0

    add_affine_transforms: bool = True
    rotation_degrees: float | Number = 15.0
    translation: tuple[float, float] | None = (0.025, 0.025)
    shear: float | tuple[float, float] | None = 3.0

    add_gaussian_blur: bool = True
    blur_kernel_size: int | Sequence[int] = 3
    blur_sigma: float | tuple[float, float] = (0.1, 2.0)

    def make_transform(self) -> Transform:
        """
        Instantiate the transformation from this configuration.
        """
        seq = []
        if self.add_affine_transforms:
            seq.append(
                RandomAffine(
                    degrees=cast(Number, self.rotation_degrees),
                    translate=self.translation,
                    shear=self.shear,
                    fill=255,  #: Fill with white
                )
            )
        if self.add_gaussian_blur:
            seq.append(
                GaussianBlur(
                    kernel_size=self.blur_kernel_size, sigma=self.blur_sigma
                )
            )
        if self.add_gaussian_noise:
            seq.append(
                GaussianNoise(sigma=self.noise_std_dev, mean=self.noise_mean)
            )

        return Compose(seq)


class CistercianDataset(VisionDataset):
    """
    Dataset to load Cistercian Numbers for train or test from a folder. The
    folder should contain the images of the numbers following the naming
    pattern: `{label}{optional-suffix}.png`. The label should be 4 digits
    representing the number depicted. The `optional-suffix` must be
    separated from the label by a hyphen and be in kebab case.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Transform | Callable | None = None,
        image_size: int = 128,
    ):
        self.__imsize = image_size

        transformation = [
            Resize(size=self.image_size),
            ToDtype(torch.float32, scale=True),
        ]
        if transform:
            transformation = Compose([transform, *transformation])
        else:
            transformation = Compose(transformation)

        super().__init__(root=root, transform=transformation)

    @property
    def image_size(self) -> tuple[int, int]:
        """
        Size of images loaded in the dataset.
        """
        return (self.__imsize, self.__imsize)

    @cached_property
    def image_paths(self) -> list[Path]:
        """
        Get the paths for all images in the dataset.
        """
        path = Path(self.root).resolve()
        images = []
        for im in path.iterdir():
            if im.is_file() and re.fullmatch(
                r"^(\d{4})(?:-\w+)*.png$", im.name
            ):
                images.append(im)

        if len(images) == 0:
            raise FileNotFoundError(
                "Found no dataset images! Check the folder or file naming."
            )

        images.sort()
        return images

    def __len__(self) -> int:
        """
        The total number of image paths in the folder.
        """
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the (image, label) tensor pair of an image in the dataset by its
        index.
        """
        fp = self.image_paths[index]
        pattern = re.search(r"(\d{4})(?:-\w+)*.png$", fp.name)
        if not pattern:
            raise ValueError("Filepath does not match expected pattern!")

        number = int(pattern.group(1))
        y = torch.from_numpy(number_to_digits(number)).to(torch.long)
        x = torchvision.io.decode_image(str(fp))
        if self.transform:
            x = self.transform(x)
        return x, y
