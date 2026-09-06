# Cistercian Numbers Project

This is a small project to train a Machine Learning model to read images of
[Cistercian numerals](https://en.wikipedia.org/wiki/Cistercian_numerals).
These are a number system devised by Cistercian Monks which allows one to write a number
between 1 and 9999 with a single glyph. This repository contains code to generate a
sample dataset of images of the numbers for parsing, and for training a
Neural Network model to read the digits from the images.

## Tech Stack
The project is mostly written in Python, but I also used some Rust and PyO3 for the image generation
code, just as an excuse to try another language and practice creating extension modules. The Deep Learning
framework used is PyTorch.

## Repository Structure
The repository is structured as follows:
```
- packages: This contains the source code, organised as UV workspaces
  - data_utils: This contains utilities to generate the image data and the PyTorch datasets to handle it. Part of
    this module is written in Rust.
  - models: This contains code to instantiate the Machine Learning models used, as well as the code to run the training loop.
```

## Setup
In order to set this up, you must first have the following already configured in your computer:
- `uv` for Python package management.
- The Rust toolchain (`rustc`, `cargo`) in order to compile the Rust package.

Once you have the above ready, you can set up the environment as follows:
```sh
# Set up Python Virtual environment
uv venv --python 3.14
source .venv/bin/activate

# Install all packages and dependencies
uv sync --all-groups --all-packages --all-extras
```
