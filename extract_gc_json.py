import os
import json

tool_names = [
    "bipolar_forceps",
    "cadiere_forceps",
    "clip_applier",
    "force_bipolar",
    "grasping_retractor",
    "monopolar_curved_scissor",
    "needle_driver",
    "permanent_cautery_hook_spatula",
    "prograsp_forceps",
    "stapler",
    "tip_up_fenestrated_grasper",
    "vessel_sealer",
]

def convert_json_to_yolo(json_path, output_dir, image_width, image_height):
    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # Group boxes by frame number
    frame_boxes = {}

    for box in data["boxes"]:
        name = box["name"].lower()
        corners = box["corners"]

        # extract frame number from name (slice_nr_X)
        parts = name.split("_")
        frame_id = parts[2]  # "1" from "slice_nr_1_bipolar_forceps"
        frame_key = f"frame_{frame_id}"

        # map tool name to class id
        tool_label = "_".join(parts[3:])  # e.g., "bipolar_forceps"
        if tool_label not in tool_names:
            continue  # skip unknown tool
        class_id = tool_names.index(tool_label)

        # corners -> bbox
        xs = [pt[0] for pt in corners]
        ys = [pt[1] for pt in corners]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # normalize to YOLO format
        x_center = ((x_min + x_max) / 2) / image_width
        y_center = ((y_min + y_max) / 2) / image_height
        w = (x_max - x_min) / image_width
        h = (y_max - y_min) / image_height

        # store per frame
        if frame_key not in frame_boxes:
            frame_boxes[frame_key] = []
        frame_boxes[frame_key].append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    # write YOLO .txt files
    os.makedirs(output_dir, exist_ok=True)
    for frame_key, labels in frame_boxes.items():
        txt_path = os.path.join(output_dir, f"{frame_key}.txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(labels))
        print(f"Wrote {txt_path}")
    
    return