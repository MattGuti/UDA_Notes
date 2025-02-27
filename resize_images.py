import os
import cv2

# Paths
ORIGINAL_PATH = "/Users/mattgutierrez80/Desktop/UDA_Notes/images"
NEW_PATH = "/Users/mattgutierrez80/Desktop/UDA_Notes/resized_images"
IMG_SIZE = (224, 224)  # Standardized image size

# Create a new directory for resized images
if not os.path.exists(NEW_PATH):
    os.makedirs(NEW_PATH)

# Loop through 'selected' and 'not_selected' folders
for category in ["selected", "not_selected"]:
    folder_path = os.path.join(ORIGINAL_PATH, category)
    new_folder_path = os.path.join(NEW_PATH, category)

    if not os.path.isdir(folder_path):
        print(f"⚠️ Skipping {folder_path}, folder not found")
        continue

    # Create the resized folder if it does not exist
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    # Resize each image
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        new_img_path = os.path.join(new_folder_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Skipping {img_path}, invalid image")
            continue

        img = cv2.resize(img, IMG_SIZE)
        cv2.imwrite(new_img_path, img)

print("✅ All images resized and saved in:", NEW_PATH)

