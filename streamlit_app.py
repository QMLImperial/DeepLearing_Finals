import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
import os

CLASS_NAMES = ['Cargo', 'Military', 'Carrier', 'Cruise', 'Tankers']

MODEL_URL = "https://raw.githubusercontent.com/QMLImperial/DeepLearing_Finals/master/ship_classifier.onnx"
MODEL_PATH = "ship_classifier.onnx"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)

@st.cache_resource
def load_model():
    download_model()
    return ort.InferenceSession(MODEL_PATH)

model = load_model()

st.title("🚢 Ship Classifier")
st.write("Upload an image of a ship to predict its type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ONNX Prediction
    input_name = model.get_inputs()[0].name
    outputs = model.run(None, {input_name: img_array})
    predictions = outputs[0][0]

    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    st.subheader("🔍 Prediction")
    st.write(f"**Class:** `{predicted_class}`")
    st.write(f"**Confidence:** `{confidence * 100:.2f}%`")

    st.subheader("📊 Class Probabilities")
    for i, prob in enumerate(predictions):
        st.write(f"{CLASS_NAMES[i]}: `{prob * 100:.2f}%`")
