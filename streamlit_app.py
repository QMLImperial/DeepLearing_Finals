import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

# Class names
CLASS_NAMES = ['Cargo', 'Military', 'Carrier', 'Cruise', 'Tankers']

# Download model from GitHub if not already present
MODEL_URL = "https://raw.githubusercontent.com/your-username/your-repo/main/ship_classifier.h5"
MODEL_PATH = "ship_classifier.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)

# Load model
@st.cache_resource
def load_model():
    download_model()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Streamlit App
st.title("🚢 Ship Classifier")
st.write("Upload a ship image and get the predicted class.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.subheader("🔍 Prediction")
    st.write(f"**Class:** `{predicted_class}`")
    st.write(f"**Confidence:** `{confidence * 100:.2f}%`")

    # Show all class probabilities
    st.subheader("📊 Class Probabilities")
    for i, prob in enumerate(predictions[0]):
        st.write(f"{CLASS_NAMES[i]}: `{prob * 100:.2f}%`")
