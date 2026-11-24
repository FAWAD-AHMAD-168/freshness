"""Simple optional Streamlit demo: upload a fruit photo for a prediction."""
import io
import streamlit as st
import tensorflow as tf
from PIL import Image
from src.config import MODELS
from src.predict import predict_image

st.set_page_config(page_title="Fruit Freshness Classifier", page_icon="🍎")
st.title("🍎 Fruit Freshness Classifier")
st.write("Upload a photo of an apple, banana, or orange. The CNN predicts **Fresh** or **Rotten**.")
st.caption("Educational demonstration only. An image cannot determine whether fruit is safe to eat.")

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model(MODELS / "fruit_freshness_model.keras")

if not (MODELS / "fruit_freshness_model.keras").is_file():
    st.warning("First train the model: python -m src.train")
    st.stop()

uploaded = st.file_uploader("Choose a fruit image", type=["jpg", "jpeg", "png"])
if uploaded:
    raw_bytes = uploaded.getvalue()
    try:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except (OSError, ValueError):
        st.error("This file is not a valid image.")
        st.stop()
    st.image(img, caption="Uploaded image", width=300)
    with st.spinner("Predicting..."):
        label, confidence, _ = predict_image(io.BytesIO(raw_bytes), load_trained_model())
    st.subheader(f"Prediction: {label}")
    st.write(f"Model score: {confidence:.1%}")
