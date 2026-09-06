# Visualisation

This package contains some basic visualisation with Streamlit to allow you to run the model on
images of your choice somewhat interactively. In order to run the streamlit app, go to the `src` directory
and run
```sh
uv run streamlit run main.py
```

You will be prompted to input the paths of a trained model and its configuration file
(so goes without saying you should train a model first!), and then you can select an 
image from your filesystem on which to run the model.
