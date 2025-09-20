# Split between train and val folders (Ultralytics-style: train/images, train/labels, validation/images, validation/labels)

from pathlib import Path
import random
import os
import sys
import shutil
import argparse

# -------- Args --------
parser = argparse.ArgumentParser()
parser.add_argument('--datapath', help='Path to data folder containing image and annotation files', required=True)
parser.add_argument('--train_pct', help='Ratio of images to go to train folder (rest to validation). Example: 0.8',
                    default=.8, type=float)
parser.add_argument('--outname', help='Name of OUTPUT root folder to create under CWD (e.g., "data_obb"). Default: "data"',
                    default='data')
parser.add_argument('--seed', help='Random seed for reproducible splits', type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

data_path = args.datapath
train_percent = float(args.train_pct)

# -------- Validate --------
if not os.path.isdir(data_path):
    print('Directory specified by --datapath not found. Verify the path and try again.')
    sys.exit(0)
if not (0.01 <= train_percent <= 0.99):
    print('Invalid entry for --train_pct. Please enter a number between 0.01 and 0.99.')
    sys.exit(0)

val_percent = 1 - train_percent

# -------- Input folders --------
input_image_path = os.path.join(data_path, 'images')
input_label_path = os.path.join(data_path, 'labels')

if not os.path.isdir(input_image_path) or not os.path.isdir(input_label_path):
    print('Expected subfolders "images" and "labels" inside --datapath. Please create them and retry.')
    sys.exit(0)

# -------- Output folders --------
cwd = os.getcwd()
root_out = os.path.join(cwd, args.outname)

train_img_path = os.path.join(root_out, 'train', 'images')
train_txt_path = os.path.join(root_out, 'train', 'labels')
val_img_path   = os.path.join(root_out, 'validation', 'images')
val_txt_path   = os.path.join(root_out, 'validation', 'labels')

for dir_path in [train_img_path, train_txt_path, val_img_path, val_txt_path]:
    os.makedirs(dir_path, exist_ok=True)

# -------- Gather files --------
img_file_list = [p for p in Path(input_image_path).rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
txt_file_list = [p for p in Path(input_label_path).rglob('*.txt')]

print(f'Number of image files: {len(img_file_list)}')
print(f'Number of annotation files: {len(txt_file_list)}')

# -------- Split counts --------
file_num = len(img_file_list)
train_num = int(file_num * train_percent)
val_num = file_num - train_num
print('Images moving to train: %d' % train_num)
print('Images moving to validation: %d' % val_num)

# -------- Copy files --------
for i, set_num in enumerate([train_num, val_num]):
    for _ in range(set_num):
        img_path = random.choice(img_file_list)
        img_fn = img_path.name
        base_fn = img_path.stem
        txt_fn = base_fn + '.txt'
        txt_path = os.path.join(input_label_path, txt_fn)

        if i == 0:
            new_img_path, new_txt_path = train_img_path, train_txt_path
        else:
            new_img_path, new_txt_path = val_img_path, val_txt_path

        shutil.copy2(img_path, os.path.join(new_img_path, img_fn))
        if os.path.exists(txt_path):  # background images allowed
            shutil.copy2(txt_path, os.path.join(new_txt_path, txt_fn))

        img_file_list.remove(img_path)

print(f'Done. Output written to: {root_out}')

# python3 utils/train_val_split.py --datapath="./extracted_1_fps1" --train_pct 0.8 --outname "extracted_1_fps1"

