import os
import shutil
import random

# Input root directory
input_root = "/mnt/c/SurgVU_25/extracted_frames"

# Output base directory
output_root = "/home/kuo/yolo/for_pseudo_ltr"
os.makedirs(output_root, exist_ok=True)

# Loop from case_000 to case_154
for i in range(155):  # 0 → 154
    case_name = f"case_{i:03d}"
    case_path = os.path.join(input_root, case_name)

    if not os.path.exists(case_path):
        print(f"⚠️ Skipping {case_name}, folder not found.")
        continue

    # Determine subfolder index (1-based)
    subfolder_idx = i // 20 + 1
    subfolder = os.path.join(output_root, str(subfolder_idx))
    os.makedirs(subfolder, exist_ok=True)

    # Collect all jpg/png images
    images = [f for f in os.listdir(case_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(images) < 50:
        print(f"⚠️ {case_name} has only {len(images)} images, taking all of them.")
        selected = images  # take all if less than 50
    else:
        selected = random.sample(images, 50)

    # Copy to output with case prefix to avoid name conflicts
    for img in selected:
        src = os.path.join(case_path, img)
        dst = os.path.join(subfolder, f"{case_name}_{img}")
        shutil.copy(src, dst)

    print(f"✅ Copied {len(selected)} images from {case_name} into subfolder {subfolder_idx}")

print("🎉 Done! All sampled images saved in subfolders under", output_root)
