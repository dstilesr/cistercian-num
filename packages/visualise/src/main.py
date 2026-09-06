import json

import streamlit as st
import utils

st.set_page_config(page_title="Cistercian Numbers Reader", page_icon=":one:")
st.title("Cistercian Numeral Classification")

model = None
img_tensor = None

with st.form(key="model-select"):
    model_cfg_path = st.text_input(
        label="Input the path of the trained model configuration (JSON):"
    )
    model_path = st.text_input(
        label="Input the path of the trained model weights:"
    )
    submitted = st.form_submit_button()

if submitted:
    st.session_state.update(
        model_cfg_path=model_cfg_path, model_path=model_path
    )
    model = utils.load_model(model_path, model_cfg_path)


with st.expander("Image Selector"):
    if st.button("Select File") or "selected_file" not in st.session_state:
        utils.select_image()

    st.write(
        "Selected file: {}".format(st.session_state.get("selected_file", "<>"))
    )


with st.expander("Image Show", expanded="selected_file" in st.session_state):
    if st.button("Show Image") or "selected_file" in st.session_state:
        try:
            img_tensor = utils.load_image()
            st.image(st.session_state.selected_file, caption="Selected Image")

        except Exception as e:  # noqa
            st.error(f"Could not load image: {e}")

    if (
        st.button("Get Number")
        and img_tensor is not None
        and utils.model_loaded()
    ):
        model = utils.load_model(
            st.session_state["model_path"], st.session_state["model_cfg_path"]
        )
        prediction = model.get_digits(img_tensor.unsqueeze(0))[0, :].tolist()

        st.write("Prediction:")
        st.code(json.dumps(prediction), language="json")
