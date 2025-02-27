import streamlit as st  
import tensorflow as tf
import numpy as np
import cv2
import os
import time
from PIL import Image
from io import BytesIO

# Getting the absolute path of the current directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__) if "__file__" in locals() else os.getcwd())

# Constructing the correct model path
MODEL_PATH = os.path.join(BASE_DIR, "image_selection_model.keras")

# Loading my trained model with caching
@st.cache_resource
def load_model():
    with st.spinner("🔄 Loading Model... Please Wait."):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None

model = load_model()

# Image size & class labels
IMG_SIZE = (224, 224)
CATEGORIES = ["not_selected", "selected"]

def predict_image(image):
    """Preprocess image & predict category."""
    img = np.array(image)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0  
    img = np.expand_dims(img, axis=0) 

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    return CATEGORIES[class_index], confidence

def play_sound(label):
    """Play different sound effects based on selection result."""
    selected_audio = "/Users/mattgutierrez80/Downloads/yippee.mp3"  
    not_selected_audio = "/Users/mattgutierrez80/Downloads/fart.mp3"  

    # Choose the correct audio file
    audio_file = selected_audio if label == "selected" else not_selected_audio

    if os.path.exists(audio_file):
        with open(audio_file, "rb") as file:
            audio_bytes = BytesIO(file.read()) 
            st.audio(audio_bytes, format="audio/mp3")  
            st.markdown("🔊 **Click play to hear the sound!**")
    else:
        st.error(f"⚠️ Sound file not found: {audio_file}")

# Custom Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e1e1e;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .stTitle {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #FFD700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Layout
st.title("📷 Photo of The Year?")
st.write(
    "Upload an image and the model will classify it as **selected** or **not selected** for a "
    "photo of the year catalog. This model was trained on images from CNN, AP, and TIME. Will your image make the cut?"
)

# File Upload
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display Uploaded Image
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    # Make Prediction
    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image... 🔍"):
            time.sleep(2) 
            label, confidence = predict_image(image)

        # Display Result
        if label == "selected":
            st.balloons()
            st.success(f"🏆 **Your image made the cut!** 🎯 (Confidence: {confidence:.2f})")
            play_sound(label) 
        else:
            st.warning(f"🚫 Not selected this time... Try another! ({confidence:.2f} confidence)")
            play_sound(label) 


st.cache_data.clear()  # Clears previous cache
st.cache_resource.clear()  # Clears loaded resources



