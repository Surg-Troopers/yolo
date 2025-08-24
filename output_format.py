import re
import json

def get_slice_number(file_path, FPS=60):
    """Extract the frame number from the file path.

    Args:
        file_path (str): The path to the file.

    Returns:
        int: The slice number extracted from the file name.
    """
    file_name = file_path.split('/')[-1]
    # format = <prefix>_<part_number>_<frame_number>.<ext>
    frame_number = re.split(r'[_\.]', file_name)[5]
    
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
        slice_number = get_slice_number(result.path, 60)
        
        xyxy = boxes.xyxy[0].cpu().numpy() # Get the first box as an example
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy() # confidence scores

        
        for j in range(len(xyxy)):
            box = {}
            x_min, y_min, x_max, y_max = xyxy[j]
            
            box["corners"] = [[x_min, y_min, 0.5], [x_max, y_min, 0.5], [x_max, y_max, 0.5], [x_min, y_max, 0.5]]
            
            box["name"] = f"slice_nr_{slice_number}_{tools_dict[cls[j]]}"

            box["probability"] = conf[j]
            
            boxes_list.append(box)

    # Add FPS information
    formatted_results = {
        "type": "Multiple 2D bounding boxes",
        "boxes": boxes_list,
        "version": { "major": 1, "minor": 0 }
    }

    return json.dumps(formatted_results, indent=4)
