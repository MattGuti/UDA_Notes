import os
import cv2

DATASET_PATH = "/Users/mattgutierrez80/Desktop/UDA_Notes/images"

def check_image_sizes():
    sizes = {}
    for folder in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder)
        if not os.path.isdir(folder_path):
            continue

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Skipping {img_path}, invalid image")
                continue

            height, width, _ = img.shape
            size_key = f"{width}x{height}"
            sizes[size_key] = sizes.get(size_key, 0) + 1

    print("Image Size Distribution:", sizes)

# Run script
check_image_sizes()
