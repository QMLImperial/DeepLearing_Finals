%%writefile streamlit_app.py
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("ship_classifier.h5")
class_names = ['Cargo', 'Military', 'Carrier', 'Cruise', 'Tankers', 'Other']

st.title("Ship Classifier")
st.write("Upload a ship image to classify its type.")

uploaded_file = st.file_uploader("Choose a ship image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    st.write(f"Prediction: **{class_names[class_idx]}**")
