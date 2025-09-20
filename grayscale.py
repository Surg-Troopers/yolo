import os
import cv2
import shutil

# Input paths
image_dir = "/home/kuo/yolo/extracted_7_fps1/train/images"
label_dir = "/home/kuo/yolo/extracted_7_fps1/train/labels"

# Output paths
output_base = "/home/kuo/yolo/gray_scale/extracted_7_fps1_gray/train"
output_images = os.path.join(output_base, "images")
output_labels = os.path.join(output_base, "labels")

# Make sure output directories exist
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# Process images
for img_name in os.listdir(image_dir):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert back to 3 channels (YOLO usually expects 3-channel images)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Save grayscale image
    cv2.imwrite(os.path.join(output_images, img_name), gray_3ch)

    # Copy corresponding label
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(label_dir, label_name)

    if os.path.exists(label_path):
        shutil.copy(label_path, os.path.join(output_labels, label_name))

print("✅ All images converted to grayscale and saved in:", output_base)
