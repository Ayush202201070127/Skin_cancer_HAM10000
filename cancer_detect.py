import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
import requests
import os

# ------------------------------
# Load Model from Google Drive
# ------------------------------
MODEL_PATH = "resnet_model.h5"
file_id = "119ForaTDWBQOoLk9maqaXkEvOYlFsX6Z"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

# Download file if not exists
if not os.path.exists(MODEL_PATH):
    session = requests.Session()
    response = session.get(url, stream=True)
    token = None

    # Get confirmation token for large files
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value

    if token:
        url_confirm = f"{url}&confirm={token}"
        response = session.get(url_confirm, stream=True)

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# ------------------------------
# Label mapping
# ------------------------------
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
st.markdown("## 🩺 Skin Cancer Classification")
st.write("Upload one or multiple dermatoscopic images to classify the lesion type.")

# ------------------------------
# Multiple File Upload
# ------------------------------
uploaded_files = st.file_uploader(
    "Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown("---")
        st.subheader(f"📌 File: {uploaded_file.name}")

        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array)
        scores = preds[0]
        class_index = np.argmax(scores)
        class_code = label_names[class_index]
        class_name = label_fullname[class_code]
        confidence = scores[class_index] * 100

        st.markdown("### 🔍 Prediction Result")
        st.success(f"**Predicted Class:** {class_name} ({class_code.upper()})")
        st.info(f"**Confidence:** {confidence:.2f}%")
