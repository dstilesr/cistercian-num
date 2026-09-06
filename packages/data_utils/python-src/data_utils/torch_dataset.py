import re
from collections.abc import Callable
from functools import cached_property
from pathlib import Path

import torch
import torchvision.io
from torchvision.datasets import VisionDataset
from torchvision.transforms.v2 import Compose, Resize, Transform

from data_utils.cistercian import number_to_digits


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
        resize = Resize(size=self.image_size)
        super().__init__(
            root=root,
            transform=Compose([transform, resize]) if transform else resize,
        )

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
        y = torch.from_numpy(number_to_digits(number))
        x = torchvision.io.decode_image(str(fp))
        if self.transform:
            x = self.transform(x)
        return x, y
