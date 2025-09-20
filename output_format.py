import re
import json
import os
from ultralytics.utils.ops import scale_boxes

def get_slice_number(file_path, FPS):
    """Extract the frame number from the file path.

    Args:
        file_path (str): The path to the file.

    Returns:
        int: The slice number extracted from the file name.
    """
    print(file_path)
    file_name = file_path.split('/')[-1]
    #/home/kuo/yolo/data_masked/cat1_test_validation/frame_00000.jpg
    # format = <prefix>_<part_number>_<frame_number>.<ext>
    frame_number = re.split(r'[_\.]', file_name)[1]
    
    return int(frame_number)//FPS

def output_format_as_json(results):
    """Format the json output to GrandChallenge format.

    Args:
        results (any): The prediction results from YOLO model.
        FPS (int, optional): The frequency of the frames per second. Defaults to 1.

    Returns:
        str (json): The formatted output in GrandChallenge format.
    """
    boxes_list = []
    
    # Convert results to GrandChallenge format
    for i, result in enumerate(results):
        boxes = result.boxes  # Boxes object
        tools_dict = result.names
        slice_number = get_slice_number(result.path, 1)

        xyxy = boxes.xyxy.cpu().numpy() # Get the first box as an example
        # xyxy = scale_boxes((640, 640), xyxy , result.orig_shape).round()

        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy() # confidence scores

        
        for j in range(len(xyxy)):
            box = {}
            print(xyxy[j])
            x_min, y_min, x_max, y_max = xyxy[j]
            
            box["corners"] = [[float(x_min), float(y_min), 0.5], [float(x_max), float(y_min), 0.5], [float(x_max), float(y_max), 0.5], [float(x_min), float(y_max), 0.5]]
            
            box["name"] = f"slice_nr_{slice_number}_{tools_dict[cls[j]]}" #changer later?

            box["probability"] = float(conf[j])
            
            boxes_list.append(box)

    # Add FPS information
    formatted_results = {
        "type": "Multiple 2D bounding boxes",
        "boxes": boxes_list,
        "version": { "major": 1, "minor": 0 }
    }

    output_path = "output.json"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)  # create folder if needed

    # Save JSON to file
    with open(output_path, "w") as f:
        json.dump(formatted_results, f, indent=4)


    return json.dumps(formatted_results, indent = 4)
