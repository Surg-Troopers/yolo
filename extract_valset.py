from extract_frame import (extract_frames, remove_unmatched_images)
from extract_gc_json import convert_json_to_yolo

video_path = "/mnt/d/SurgVU 25/cat1_test_set_public/7_fps1.mp4"
json_path = "/mnt/d/SurgVU 25/cat1_test_set_public/7_fps1_gc.json"
output_dir = "extracted_7_fps1/images/"
json_output = "extracted_7_fps1/labels/"

width, height = extract_frames(video_path=video_path, output_dir=output_dir)

convert_json_to_yolo(json_path=json_path, output_dir=json_output, 
                image_width=width, image_height=height)

remove_unmatched_images(images_dir=output_dir, labels_dir=json_output)