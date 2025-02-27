import streamlit as st  
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Loading the trained model
import os
import streamlit as st
import tensorflow as tf

# ✅ Get the absolute path of the current directory (Works in Streamlit!)
BASE_DIR = os.path.abspath(os.path.dirname(__file__) if "__file__" in locals() else os.getcwd())

# ✅ Construct the correct model path
MODEL_PATH = os.path.join(BASE_DIR, "image_selection_model.keras")

# ✅ Load the trained model with caching
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()



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

# Making the app
st.title("📷 Photo of The Year?")
st.write("Upload an image and the model will classify it as **selected** or **not selected** for a photo of the year catalog. This model was tranied off images from CNN, AP, and TIME. Will your image make the cut?")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Displaying Uploaded Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Making the Prediction
    if st.button("🔍 Predict"):
        label, confidence = predict_image(image)
        st.success(f"**Prediction:** {label} 🎯 (Confidence: {confidence:.2f})")


