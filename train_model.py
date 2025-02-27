import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
import cv2
from sklearn.model_selection import train_test_split
import hashlib
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Define dataset path
DATASET_PATH = "/Users/mattgutierrez80/Desktop/UDA_Notes/resized_images"

# ✅ Define constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CATEGORIES = ["not_selected", "selected"]

# ✅ Function to load and preprocess images
def load_images():
    data, labels, hashes = [], [], set()
    for category in CATEGORIES:
        class_index = CATEGORIES.index(category)
        category_path = os.path.join(DATASET_PATH, category)

        if not os.path.exists(category_path):
            continue

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)

            # ✅ Remove duplicates
            img_hash = hashlib.md5(img.tobytes()).hexdigest()
            if img_hash in hashes:
                continue

            hashes.add(img_hash)
            data.append(img)
            labels.append(class_index)

    return np.array(data), np.array(labels)

# ✅ Load dataset and normalize
X, y = load_images()
X = X / 255.0  # Normalize images (0-255 → 0-1)

# ✅ Train-Test Split (60% train, 20% validation, 20% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training images: {len(X_train)}")
print(f"Validation images: {len(X_val)}")
print(f"Test images: {len(X_test)}")

# ✅ Data Augmentation
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2,
                             height_shift_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode='nearest')
datagen.fit(X_train)

# ✅ Optimized Model with MobileNetV2 (Transfer Learning)
def create_model():
    base_model = applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False  # Freeze base layers

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),  # ✅ Reduced dropout
        layers.Dense(len(CATEGORIES), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ✅ Train the model
print("🚀 Training model...")
model = create_model()
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_val, y_val),
                    epochs=25)  # ✅ Increased epochs

# ✅ Save the model
model.save("/Users/mattgutierrez80/image_selection_model.keras")
print("✅ Model saved as image_selection_model.keras")

# ✅ Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

