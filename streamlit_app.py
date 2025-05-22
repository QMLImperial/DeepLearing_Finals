import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

class_names = ['Cargo', 'Military', 'Carrier', 'Cruise', 'Tankers']

# Load ONNX model
@st.cache_resource
def load_model():
    session = ort.InferenceSession("ship_classifier.onnx", providers=["CPUExecutionProvider"])
    return session

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("🚢 Ship Type Classifier")

uploaded_file = st.file_uploader("Upload a ship image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).numpy()

    input_name = model.get_inputs()[0].name
    outputs = model.run(None, {input_name: img_tensor})

    logits = torch.tensor(outputs[0][0])
    probs = torch.nn.functional.softmax(logits, dim=0).numpy()
    pred_index = np.argmax(probs)
    pred_label = class_names[pred_index]

    st.subheader(f"Predicted: {pred_label}")
    st.write("Prediction confidence:")
    st.bar_chart(probs)
