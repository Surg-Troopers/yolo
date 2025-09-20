# extract frame from input video from GrandChallenge
import cv2
import os

def extract_frames(video_path, output_dir, fps_interval=1):
    """
    Extract frames from a video at specified FPS.

    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save extracted frames.
        fps (int): Number of frames to extract per second.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video resolution: {width}x{height}")

    count = 0
    saved_count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        if count % fps_interval == 0:
            # Save frame
            frame_name = os.path.join(output_dir, f"frame_{int(saved_count)}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_count += 1

        count += 1

    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")

    return width, height

def remove_unmatched_images(images_dir, labels_dir):
    """
    Remove images from images_dir if a matching .txt file does not exist in labels_dir.
    Assumes filenames (without extension) must match.
    """
    removed = 0
    for img_file in os.listdir(images_dir):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            base_name = os.path.splitext(img_file)[0]
            txt_file = base_name + ".txt"
            txt_path = os.path.join(labels_dir, txt_file)

            if not os.path.exists(txt_path):
                img_path = os.path.join(images_dir, img_file)
                os.remove(img_path)
                removed += 1
                print(f"Removed {img_path}")

    print(f"Cleanup done. Removed {removed} images without labels.")
    return