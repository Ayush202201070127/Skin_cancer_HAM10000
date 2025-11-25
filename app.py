import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np

# ------------------------------
# Load Model
# ------------------------------
MODEL_PATH = "resnet_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Label mapping
label_names = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']

label_fullname = {
    'akiec': 'Actinic keratoses',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions'
}

# ------------------------------
# Streamlit Page Configuration
# ------------------------------
st.set_page_config(page_title="Skin Cancer Classification", layout="centered")

# Title
st.markdown("## 🩺 Skin Cancer Classification (ResNet50)")
st.write("Upload a dermatoscopic image to classify the lesion type.")

# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess Image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)
    scores = preds[0]
    class_index = np.argmax(scores)
    class_code = label_names[class_index]
    class_name = label_fullname[class_code]
    confidence = scores[class_index] * 100

    # ------------------------------
    # Prediction Output (TEXT ONLY)
    # ------------------------------
    st.markdown("### 🔍 Prediction Result")
    st.success(f"**Predicted Class:** {class_name} ({class_code.upper()})")
    st.info(f"**Confidence:** {confidence:.2f}%")
