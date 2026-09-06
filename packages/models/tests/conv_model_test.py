import numpy as np
import pytest
import torch
from models import convolutional as conv


def test_model_instantiation(model_test_cfg):
    """
    Test that the model can be instantiated correctly
    """
    _ = conv.ConvolutionalModel(model_test_cfg)


def test_model_instantiation_fail(model_test_cfg):
    """
    Test that the model instantiation fails with invalid parameters.
    """
    model_test_cfg.linear_layers[-1].output_dim = 64
    with pytest.raises(ValueError):
        _ = conv.ConvolutionalModel(model_test_cfg)


def test_forward_pass(model_test_cfg):
    """
    Test that the model forward pass returns data in the correct dimension
    """
    model = conv.ConvolutionalModel(model_test_cfg)
    model.eval()

    with torch.no_grad():
        # Batch of 5 random inputs
        inputs = torch.rand((5, 1, 32, 32), dtype=torch.float32)
        out = model(inputs)
        assert len(out.shape) == 3
        assert out.shape[0] == 5
        assert out.shape[1] == 4
        assert out.shape[2] == 10


def test_classification(model_test_cfg):
    """
    Test that the classification head works correclty on the logits.
    """
    model = conv.ConvolutionalModel(model_test_cfg)
    model.eval()

    with torch.no_grad():
        # Batch of 5 random inputs
        inputs = torch.rand((31, 1, 32, 32), dtype=torch.float32)
        probas = model.get_digits(inputs, probabilities=True)

        assert probas.shape == (31, 4, 10)
        probas = probas.numpy()
        sums = probas.sum(axis=2)
        assert np.allclose(sums, 1.0)

        labels_1 = probas.argmax(axis=2).astype(np.uint32)
        labels_2 = model.get_digits(inputs, probabilities=False).numpy()

        assert np.all(labels_1 == labels_2)
