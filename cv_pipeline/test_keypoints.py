from ultralytics import YOLO
import cv2

model = YOLO(r'C:\UCLA\yolo model 2\pose\pitch_keypoints_model2\weights\best.pt')

cap = cv2.VideoCapture(r'C:\UCLA\yolo model 2\121364_0.mp4')
ret, frame = cap.read()
cap.release()

print(f"Frame shape: {frame.shape}")

# Try very low confidence threshold
results = model.predict(source=frame, conf=0.1, imgsz=1280, save=False)

annotated = results[0].plot()
cv2.imwrite(r'C:\UCLA\yolo model 2\keypoint_test.jpg', annotated)

if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
    kpts = results[0].keypoints
    print(f"Keypoints detected: {len(kpts.xy[0])}")
    print(f"Confidence scores: {kpts.conf[0]}")
else:
    print("No keypoints detected - model may need more training")