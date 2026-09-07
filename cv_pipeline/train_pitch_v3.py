"""
train_pitch_v3.py — retrain YOLOv8-pose pitch keypoint model (v3).

Saves best weights to: pose/pitch_keypoints_v3/weights/best.pt
"""
import time
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

print("Starting v3 training: epochs=150, imgsz=1280, batch=8, lr0=0.001")
t0 = time.time()

results = model.train(
    data='pitch_keypoints_data/data.yaml',
    epochs=150,
    imgsz=1280,
    batch=8,
    patience=30,
    augment=True,
    lr0=0.001,
    project='pose',
    name='pitch_keypoints_v3',
    exist_ok=True,
    device='cpu',
    workers=4,
    verbose=True,
)

elapsed = time.time() - t0
print(f"\nTraining complete in {elapsed/3600:.1f} h")
print(f"Best weights: pose/pitch_keypoints_v3/weights/best.pt")
print(f"mAP50-pose: {results.results_dict.get('metrics/mAP50(P)', 'N/A')}")
