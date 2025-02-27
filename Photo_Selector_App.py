import streamlit as st  
import tensorflow as tf
import numpy as np
import cv2
import os
import time
from PIL import Image
import streamlit.components.v1 as components

# ✅ Get the absolute path of the current directory (Works in Streamlit!)
BASE_DIR = os.path.abspath(os.path.dirname(__file__) if "__file__" in locals() else os.getcwd())

# ✅ Construct the correct model path
MODEL_PATH = os.path.join(BASE_DIR, "image_selection_model.keras")

# ✅ Load the trained model with caching
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

# ✅ Image size & class labels
IMG_SIZE = (224, 224)
CATEGORIES = ["not_selected", "selected"]

def predict_image(image):
    """Preprocess image & predict category."""
    img = np.array(image)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    return CATEGORIES[class_index], confidence

def play_sound():
    """Play sound effect if selected."""
    components.html(
        """
        <audio autoplay>
        <source src="https://www.myinstants.com/media/sounds/tada-fanfare-a.mp3" type="audio/mpeg">
        </audio>
        """,
        height=0,
    )

# ✅ Custom CSS Styling
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

# ✅ Structured Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.image("📷")  # Placeholder for a logo (Optional)

with col2:
    st.title("📷 Photo of The Year?")
    st.write(
        "Upload an image and the model will classify it as **selected** or **not selected** for a "
        "photo of the year catalog. This model was trained on images from CNN, AP, and TIME."
    )

# ✅ File Upload
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # ✅ Display Uploaded Image
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # ✅ Make Prediction
    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image... 🔍"):
            time.sleep(2)  # Simulate loading effect
            label, confidence = predict_image(image)

        # ✅ Display Result
        if label == "selected":
            play_sound()
            st.balloons()
            st.success(f"🏆 **Your image made the cut!** 🎯 (Confidence: {confidence:.2f})")
        else:
            st.warning(f"🚫 Not selected this time... Try another! ({confidence:.2f} confidence)")


