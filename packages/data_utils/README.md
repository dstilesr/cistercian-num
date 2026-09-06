# Data Utilities

This package contains utilities for creating data for training and evaluating the model that reads cistercian numerals.
You may use some functionality from Rust if you wish, but it is more straightforward to use this with python.

## Structure
The Rust code for this package is in the `src` folder, and is configured as a Rust cdylib to use as Python extension. The
Python source code is in the `python-src` directory, contained in the `data_utils` module.

## Development
To install a development version of the package in your local environment, you can run `uv run maturin develop`. You can also build the full package wheel
with `uv run maturin build --release`. You can test the Rust code with `cargo test --no-default-features`. This flag is needed because there can be some linking
issues due to the python versions and the virtual environment confusing Cargo.

## Usage Examples

### Generate and View Images
The package can generate sample images as prototypes. They may not be the prettiest but they get the job done for now.
```python
from matplotlib import pyplot as plt

from data_utils.cistercian import make_image

size = 128
thickness = 3
margin = 6

# Generate and show an image with the given number
img = make_image(size, thickness, margin, 5943)
plt.imshow(img, cmap="gray")
```

### Generate Full Dataset
You can also generate the full training and test datasets of prototype images. Here a random fraction of the images will be
saved in the test set. The images will be saved in two folders: `train` and `test` within your desired location. These folders
will contain the prototype images stored as `.png`.

In order for the images to be used later by the dataloader, they must follow
the naming convention: `{label}-{optional-suffix}.png`. Label refers to the actual number represented on the image, zero-padded to
length 4. The optional suffix is there to allow you to add more images (e.g. handwritten versions) to the train and evaluation
sets. Some examples of valid names are:
- `0012.png`
- `9812-nice-image.png`
- `0790-interesting.png`

```python
from pathlib import Path

from data_utils.dataset_gen import generate_dataset

size = 128
thickness = 3
margin = 6
# Save 7.5% of images for test set
train_set_proportion = 0.075

out_path = Path("data").resolve()

generate_dataset(
    save_to=out_path,
    size=size,
    margin=margin,
    line_thickness=thickness,
    test_proportion=test_proportion,
)
```

### Using the PyTorch Datasets
The package also contains a dataset class made for using the dataset as specified and generated above. The dataloader can be pointed at a directory
with `.png` images following the naming convention explained above. The basic dataloader can be used like this:
```python
from pathlib import Path

from matplotlib import pyplot as plt

from data_utils.torch_dataset import CistercianDataset

# All images will be resized to this dimension on loading. Ideally
# it will match that in which they were originally generated.
image_size = 128
data_path = Path("data").resolve() / "train"

# Instantiate the dataset. This is a regular Torchvision dataset
ds = CistercianDataset(root=data_path, image_size=image_size)

image, label = ds[123]
print(label)  # Array with the 4 digits of the numeral

plt.imshow(image[0, :, :], cmap="gray")
```

You can also get the datasets including some configurable transforms / distortions as data augmentation
by using the following:
```python
from pathlib import Path

from matplotlib import pyplot as plt


from data_utils.torch_dataset import get_train_test_datasets, PerturbationConfig

image_size = 128

# This path must have a 'train' and a 'test' directory following the 
# conventions above.
data_path = Path("data").resolve()

# Check this to see the options you can add
config = PerturbationConfig()
train_ds, test_ds = get_train_test_datasets(
    data_path,
    image_size,
    perturb_cfg=config,
    perturb_train=True,
    perturb_test=False,
)

# Check a distorted image
image, label = train_ds[123]
plt.imshow(image[0, :, :], cmap="gray");
```
