import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


# Set paths
IMAGE_DIR = "images"
OUTPUT_CSV = "features.csv"

# Load pre-trained VGG16 model
vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Function to extract deep learning features
def extract_vgg16_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = vgg_model.predict(img)
    return features.flatten()

# Function to extract color histogram features
def extract_color_histogram(img_path, bins=32):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    hist = cv2.calcHist([img], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Function to extract edge detection features
def extract_edge_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    edges = cv2.Canny(img, 100, 200)
    return edges.flatten()

# Process images
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
feature_vectors = []

for img_file in image_files:
    img_path = os.path.join(IMAGE_DIR, img_file)
    print(f"Processing: {img_file}")

    vgg_features = extract_vgg16_features(img_path)
    color_hist_features = extract_color_histogram(img_path)
    edge_features = extract_edge_features(img_path)
    
    combined_features = np.hstack([vgg_features, color_hist_features, edge_features])
    feature_vectors.append(combined_features)

# Save to CSV
feature_df = pd.DataFrame(feature_vectors)
feature_df.to_csv(OUTPUT_CSV, index=False)
print(f"Feature extraction completed! Saved to {OUTPUT_CSV}")