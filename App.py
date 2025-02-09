import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Google Drive model file ID
file_id = "1OsPx6VZAxSgJgV18ULNNFvG3cUC3y39G"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "trained_plant_disease_model.keras"

# Download model from Google Drive if not present
if not os.path.exists(model_path):
    with st.spinner("Downloading model... This may take a while ⏳"):
        gdown.download(url, model_path, quiet=False)

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

# Class Labels
class_names = ['Potato - Early Blight', 'Potato - Late Blight', 'Potato - Healthy']

# Disease Information and Remedies
disease_info = {
    "Potato - Early Blight": "Early blight is a fungal disease caused by Alternaria solani. It results in dark spots with concentric rings on leaves.",
    "Potato - Late Blight": "Late blight, caused by Phytophthora infestans, is a severe disease leading to dark lesions and decay.",
    "Potato - Healthy": "No disease detected. Keep monitoring your plants for any changes."
}

remedies = {
    "Potato - Early Blight": "Use fungicides like chlorothalonil and copper-based treatments. Ensure proper crop rotation.",
    "Potato - Late Blight": "Apply fungicides like mancozeb and copper oxychloride. Remove infected plants to prevent spread.",
    "Potato - Healthy": "Maintain proper watering, use quality fertilizers, and ensure good air circulation."
}

# Image Prediction Function
def model_prediction(image):
    image = image.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    confidence = np.max(predictions)
    return np.argmax(predictions), confidence

# Sidebar Navigation
st.sidebar.title("Plant Disease Detection System 🌱")
app_mode = st.sidebar.radio("Navigation", ['Home', 'Disease Recognition'])

# Home Page
if app_mode == 'Home':
    st.markdown("""
    <h1 style='text-align: center;'>🌾 Plant Disease Detection System for Sustainable Agriculture 🌿</h1>
    <p style='text-align: center;'>Upload an image of a potato plant leaf to detect diseases and receive treatment recommendations.</p>
    """, unsafe_allow_html=True)
    st.image("Diseases.png", use_column_width=True)

# Disease Recognition Page
elif app_mode == 'Disease Recognition':
    st.header("📸 Plant Disease Detection")
    test_image = st.file_uploader("Upload an Image (JPG, PNG):", type=["jpg", "png", "jpeg"])
    
    if test_image is not None:
        image = Image.open(test_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Predict Disease'):
            with st.spinner('Analyzing Image... ⏳'):
                result_index, confidence = model_prediction(image)
                disease_name = class_names[result_index]
                
                st.success(f'✅ Model Prediction: {disease_name}')
                st.write(f'🔍 **Confidence Score:** {confidence:.2%}')
                
                st.subheader("📝 Disease Information")
                st.info(disease_info[disease_name])
                
                st.subheader("💡 Suggested Remedies")
                st.warning(remedies[disease_name])
                
                st.balloons()

# Footer
st.sidebar.markdown("""
📌 **Project Features:**
- AI-based plant disease detection 🌾
- Provides disease details & treatment suggestions 💊
- Uses Convolutional Neural Networks (CNNs) 🤖
""")
