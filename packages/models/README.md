# Models
This package contains the code for the actual modelling to read the number from a Cistercian numeral file.

## Model Configuration
The model definition is in the `models.convolutional` subpackage along with the configuration schema. The models
there are basic Convolutional Neural Network models that can be configured to some extent with
[this schema](./src/models/convolutional/settings.py).

The configuration schema is a Pydantic model, and you must create one before training and instantiating a model.
Here is an example for a model that will carry out predictions in `128 x 128` images:
```json
{
  "dtype": "float32",
  "dropout": 0.1,
  "conv_layers": [
    {
      "input_channels": 1,
      "output_channels": 4,
      "add_padding": true,
      "activation": "relu"
    },
    {
      "input_channels": 4,
      "output_channels": 8,
      "add_padding": true,
      "activation": "relu"
    },
    {
      "input_channels": 8,
      "output_channels": 16,
      "add_padding": true,
      "activation": "relu"
    }
  ],
  "linear_layers": [
    {
      "input_dim": 4096,
      "output_dim": 1024,
      "activation": "relu",
      "normalise": true
    },
    {
      "input_dim": 1024,
      "output_dim": 512,
      "activation": "relu",
      "normalise": true
    },
    {
      "input_dim": 512,
      "output_dim": 40,
      "activation": "linear"
    }
  ]
}
```

## Training
The module also contains code to run model training. In order to do this, you must first prepare your dataset as
specified in the [`data-utils`](../data_utils/) package. You can then create a training configuration file with the schema
specified in the [training](./src/models/training/settings.py) module. An example training configuration schema is
```json
{
  "learning_rate": 0.004,
  "epochs": 10,
  "dataset_path": "path/to/data/dir",
  "save_to": "save/checkpoints/here"
}
```
The `dataset_path` must contain a `train` and a `test` subdirectory as specified in the data package's README to ensure the dataloaders
can find the images you want to use. Once you have the model and training configurations ready, you can do a training run by going to the
`src` subdirectory and running
```sh
uv run python -m models.training \
    path/to/model-cfg.json \
    --train-cfg=path/to/train-cfg.json
```
