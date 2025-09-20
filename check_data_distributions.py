import os
from collections import defaultdict

def count_yolo_classes(labels_dir):
    """
    Count YOLO class occurrences across all .txt files in a directory.
    
    Args:
        labels_dir (str): Path to the directory containing YOLO txt label files.
    
    Returns:
        dict: {class_id: count}
    """
    class_counts = defaultdict(int)

    # Loop through all txt files
    for folder_name in labels_dir:
        for file_name in os.listdir(folder_name):
            if file_name.endswith(".txt"):
                file_path = os.path.join(folder_name, file_name)
                with open(file_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:  # skip empty lines
                            class_id = int(parts[0])  # first column = class number
                            class_counts[class_id] += 1

    return dict(class_counts)


if __name__ == "__main__":
    labels_dir = [
    "/home/kuo/yolo/new_data_sept8_after_mask/train/labels",
    "/home/kuo/yolo/extracted_1_fps1/train/labels",
    "/home/kuo/yolo/extracted_2_fps1/train/labels",
    "/home/kuo/yolo/extracted_3_fps1/train/labels",
    "/home/kuo/yolo/extracted_4_fps1/train/labels",
    "/home/kuo/yolo/extracted_5_fps1/train/labels",
    "/home/kuo/yolo/extracted_6_fps1/train/labels",
    "/home/kuo/yolo/extracted_7_fps1/train/labels",
      ]  # <-- change this to your directory
    counts = count_yolo_classes(labels_dir)
    
    print("Bounding box counts per class:")
    for class_id, count in sorted(counts.items()):
        print(f"Class {class_id}: {count}")
