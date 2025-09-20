import torch, ultralytics
from ultralytics import YOLO
print("PyTorch:", torch.__version__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device available:", DEVICE)

print("Ultralytics:", ultralytics.__version__)

model = YOLO("runs/detect/train_notebook/weights/best.pt")
# video_path = "/mnt/d/SurgVU 25/surgvu24_videos_only/surgvu24/case_001/case_001_video_part_001.mp4""
pred = model.predict(
    source="data_masked/cat1_test_validation",
    imgsz=512,
    conf=0.25,
    save=True
)