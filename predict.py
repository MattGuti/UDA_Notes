import tensorflow as tf
import numpy as np
import cv2
import sys

# Loading the trained model
try:
    model = tf.keras.models.load_model("image_selection_model.keras")
    print("Model loaded successfully.")
except Exception as e:
    print(f" Error loading model: {e}")
    sys.exit(1)
model.summary()
# Image size & class labels
IMG_SIZE = (224, 224)
CATEGORIES = ["not_selected", "selected"]

def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Making prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"🔍 Prediction: {CATEGORIES[class_index]} (Confidence: {confidence:.2f})")

# Test with test image
image_path = "/Users/mattgutierrez80/Desktop/koolaid.jpeg"
predict_image(image_path)

