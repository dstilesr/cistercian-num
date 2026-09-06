import os
from pathlib import Path

import streamlit as st
import torch
from models import convolutional as conv
from torchvision.io import decode_image
from torchvision.transforms.v2 import Compose, Resize, ToDtype


@st.cache_data
def load_model(state_path: str, cfg_path: str) -> conv.ConvolutionalModel:
    """
    Load a trained classifier model.
    :param cfg_path: Path to configuration used to instantiate model
    :param state_path: Path to model state dictionary (.pth file)
    """
    with open(cfg_path, "r") as f:
        model_config = conv.ModelSettings.model_validate_json(f.read())

    model = conv.ConvolutionalModel(model_config)
    model.load_state_dict(torch.load(state_path, weights_only=True))
    model.eval()
    st.info("Loaded model!")
    st.session_state.image_size = model_config.input_image_size
    return model


@st.dialog("Select an Image")
def select_image():
    """
    Load an image from a file in the filesystem
    """
    if "curr_img_dir" not in st.session_state:
        dirpath = Path(__file__).parents[3]
        st.session_state.curr_img_dir = dirpath
    else:
        dirpath = Path(st.session_state.curr_img_dir).resolve()

    st.write(f"Current directory: {dirpath!s}")

    if not dirpath.is_dir():
        dirpath = dirpath.parent

    files = [f for f in os.listdir(dirpath) if not f.startswith(".")]
    files.sort()
    files = files[:24]
    if st.button("Move Up"):
        st.session_state.curr_img_dir = str(dirpath.parent)
        st.rerun()

    selected = st.selectbox("img-file", options=files)
    path = dirpath / selected
    if st.button("Confirm"):
        if path.is_dir():
            st.session_state.curr_img_dir = str(path)
            st.rerun()
        else:
            st.session_state.selected_file = str(path)
            st.rerun()


def load_image() -> torch.Tensor | None:
    """
    Load the image if a file has been selected.
    """
    if not (
        "selected_file" in st.session_state and "image_size" in st.session_state
    ):
        return None

    size = st.session_state.image_size
    transform = Compose(
        [Resize(size=(size, size)), ToDtype(dtype=torch.float32, scale=True)]
    )
    image = decode_image(st.session_state.selected_file)
    return transform(image)


def model_loaded() -> bool:
    """
    Whether the model has been loaded
    """
    return (
        "model_cfg_path" in st.session_state
        and "model_path" in st.session_state
    )
