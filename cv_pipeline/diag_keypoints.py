"""
Diagnostic: analyze training label files to compute the average normalized
position of each of the 48 keypoints.  Then run the v3 model on a test frame
and draw every detected keypoint with its index so we can visually verify
which index belongs to which pitch feature.
"""

import os
import glob
import numpy as np
import cv2
from collections import defaultdict

# ── Step 1: analyse all label files ─────────────────────────────────────────
label_dirs = [
    r'C:\UCLA\yolo model 2\pitch_keypoints_data\test\labels',
    r'C:\UCLA\yolo model 2\train\labels',
]

sum_x  = defaultdict(float)
sum_y  = defaultdict(float)
counts = defaultdict(int)

for ldir in label_dirs:
    for lpath in glob.glob(os.path.join(ldir, '*.txt')):
        with open(lpath) as f:
            line = f.read().strip()
        tokens = line.split()
        # format: class cx cy w h  [kpx kpy kpc] x48
        if len(tokens) < 5 + 48 * 3:
            continue
        kp_tokens = tokens[5:]
        for i in range(48):
            kpx = float(kp_tokens[i * 3])
            kpy = float(kp_tokens[i * 3 + 1])
            kpc = float(kp_tokens[i * 3 + 2])
            if kpc > 0:          # 1 = visible but occluded, 2 = visible
                sum_x[i] += kpx
                sum_y[i] += kpy
                counts[i] += 1

print("\n=== Average normalised position per keypoint index ===")
print(f"{'idx':>4}  {'avg_x':>7}  {'avg_y':>7}  {'n':>6}")
for i in range(48):
    n = counts[i]
    if n == 0:
        print(f"{i:>4}  {'--':>7}  {'--':>7}  {0:>6}")
    else:
        ax = sum_x[i] / n
        ay = sum_y[i] / n
        print(f"{i:>4}  {ax:>7.4f}  {ay:>7.4f}  {n:>6}")

# ── Step 2: run the v3 model on a test frame and draw keypoints ──────────────
from ultralytics import YOLO

MODEL_PATH = r'C:\UCLA\yolo model 2\pose\pitch_keypoints_v3\weights\best.pt'
VIDEO_PATH = r'C:\UCLA\yolo model 2\Match_videos\121364_0.mp4'

print("\nLoading v3 model ...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
# jump to frame ~300 (10 s) for a good mid-pitch view
cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Could not read frame from video.")
    exit()

print(f"Frame shape: {frame.shape}")
results = model.predict(source=frame, conf=0.10, imgsz=1280, save=False, verbose=False)

out = frame.copy()

if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
    kpts = results[0].keypoints
    xy   = kpts.xy[0].cpu().numpy()    # (48, 2)
    conf = kpts.conf[0].cpu().numpy()  # (48,)

    print("\n=== Detected keypoints (conf > 0.10) ===")
    print(f"{'idx':>4}  {'px_x':>6}  {'px_y':>6}  {'conf':>6}")
    for i, (pt, c) in enumerate(zip(xy, conf)):
        if c > 0.10:
            px, py = int(pt[0]), int(pt[1])
            print(f"{i:>4}  {px:>6}  {py:>6}  {c:>6.3f}")
            # Draw dot and label
            cv2.circle(out, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(out, str(i), (px + 4, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
else:
    print("No keypoints detected.")

save_path = r'C:\UCLA\yolo model 2\kp_label_debug.jpg'
cv2.imwrite(save_path, out)
print(f"\nAnnotated frame saved to: {save_path}")
print("Open it to see which index is at which pitch location.")
