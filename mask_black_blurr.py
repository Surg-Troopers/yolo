# import os
# import random
# import shutil
# import cv2

# # Input paths
# image_dir = "/home/kuo/yolo/extracted_1_fps1/train/images"
# label_dir = "/home/kuo/yolo/extracted_1_fps1/train/labels"

# # Output paths
# output_base = "/home/kuo/yolo/50_50_augment/extracted_1_fps1_augmented/train"
# output_images = os.path.join(output_base, "images")
# output_labels = os.path.join(output_base, "labels")

# # Make sure output directories exist
# os.makedirs(output_images, exist_ok=True)
# os.makedirs(output_labels, exist_ok=True)

# def yolo_to_xyxy(line, img_w, img_h):
#     """Convert YOLO format to pixel (x_min, y_min, x_max, y_max)."""
#     parts = line.strip().split()
#     cls, x_center, y_center, w, h = map(float, parts)
#     x_center, y_center, w, h = x_center * img_w, y_center * img_h, w * img_w, h * img_h
#     x_min = int(x_center - w / 2)
#     y_min = int(y_center - h / 2)
#     x_max = int(x_center + w / 2)
#     y_max = int(y_center + h / 2)
#     return cls, x_min, y_min, x_max, y_max

# # Process all images
# for img_name in os.listdir(image_dir):
#     if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
#         continue

#     img_path = os.path.join(image_dir, img_name)
#     img = cv2.imread(img_path)
#     if img is None:
#         continue
#     h, w, _ = img.shape

#     label_name = os.path.splitext(img_name)[0] + ".txt"
#     label_path = os.path.join(label_dir, label_name)

#     # Copy labels no matter what
#     if os.path.exists(label_path):
#         shutil.copy(label_path, os.path.join(output_labels, label_name))

#     # Choose action: 0 = original, 1 = black mask, 2 = blur mask
#     r = random.random()
#     if r < 0.50 and os.path.exists(label_path):
#         # Black background (mask)
#         with open(label_path, "r") as f:
#             lines = f.readlines()

#         masked_img = img.copy()
#         masked_img[:] = 0

#         for line in lines:
#             cls, x_min, y_min, x_max, y_max = yolo_to_xyxy(line, w, h)
#             x_min, y_min = max(0, x_min), max(0, y_min)
#             x_max, y_max = min(w, x_max), min(h, y_max)
#             masked_img[y_min:y_max, x_min:x_max] = img[y_min:y_max, x_min:x_max]

#         cv2.imwrite(os.path.join(output_images, img_name), masked_img)

#     # elif r < 0.50 and os.path.exists(label_path):
#     else:
#         # Blurred background
#         with open(label_path, "r") as f:
#             lines = f.readlines()

#         blurred_img = cv2.GaussianBlur(img, (51, 51), 0)  # adjust kernel size if needed

#         # Copy bbox area back to sharp
#         for line in lines:
#             cls, x_min, y_min, x_max, y_max = yolo_to_xyxy(line, w, h)
#             x_min, y_min = max(0, x_min), max(0, y_min)
#             x_max, y_max = min(w, x_max), min(h, y_max)
#             blurred_img[y_min:y_max, x_min:x_max] = img[y_min:y_max, x_min:x_max]

#         cv2.imwrite(os.path.join(output_images, img_name), blurred_img)

#     # else:
#     #     # Keep original
#     #     shutil.copy(img_path, os.path.join(output_images, img_name))

# print("✅ Processing completed. Results saved to:", output_base)

import os
import shutil
import cv2

# Input paths
image_dir = "/home/kuo/yolo/extracted_7_fps1/train/images"
label_dir = "/home/kuo/yolo/extracted_7_fps1/train/labels"

# Output paths
output_base = "/home/kuo/yolo/100_blur_mask_set/extracted_7_fps1/train"
output_images = os.path.join(output_base, "images")
output_labels = os.path.join(output_base, "labels")

# Make sure output directories exist
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

def yolo_to_xyxy(line, img_w, img_h):
    """Convert YOLO format to pixel (x_min, y_min, x_max, y_max)."""
    parts = line.strip().split()
    cls, x_center, y_center, w, h = map(float, parts)
    x_center, y_center, w, h = x_center * img_w, y_center * img_h, w * img_w, h * img_h
    x_min = int(x_center - w / 2)
    y_min = int(y_center - h / 2)
    x_max = int(x_center + w / 2)
    y_max = int(y_center + h / 2)
    return cls, x_min, y_min, x_max, y_max

# Collect all image filenames
images = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
images.sort()  # make order stable

# Split into first half (mask) and second half (blur)
half = 0
mask_images = images[:half]
blur_images = images[half:]

for idx, img_name in enumerate(images):
    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w, _ = img.shape

    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(label_dir, label_name)

    # Copy labels (always needed)
    if os.path.exists(label_path):
        shutil.copy(label_path, os.path.join(output_labels, label_name))
    else:
        # Skip if no labels
        continue

    # Decide operation
    if img_name in mask_images:
        # Black background (mask)
        with open(label_path, "r") as f:
            lines = f.readlines()

        masked_img = img.copy()
        masked_img[:] = 0

        for line in lines:
            cls, x_min, y_min, x_max, y_max = yolo_to_xyxy(line, w, h)
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            masked_img[y_min:y_max, x_min:x_max] = img[y_min:y_max, x_min:x_max]

        cv2.imwrite(os.path.join(output_images, img_name), masked_img)

    else:  # blur_images
        # Blurred background
        with open(label_path, "r") as f:
            lines = f.readlines()

        blurred_img = cv2.GaussianBlur(img, (51, 51), 0)

        for line in lines:
            cls, x_min, y_min, x_max, y_max = yolo_to_xyxy(line, w, h)
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            blurred_img[y_min:y_max, x_min:x_max] = img[y_min:y_max, x_min:x_max]

        cv2.imwrite(os.path.join(output_images, img_name), blurred_img)

print("✅ Done! First half masked, second half blurred. Results saved to:", output_base)
