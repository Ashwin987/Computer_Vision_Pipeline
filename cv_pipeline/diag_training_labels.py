"""
Draw ground-truth keypoint labels (from the training data label files) onto
the corresponding images so we can visually verify which index = which pitch
feature.  Picks the label file with the most visible (conf=2) keypoints.
"""

import os, glob
import numpy as np
import cv2

LABEL_DIR = r'C:\UCLA\yolo model 2\train\labels'
IMAGE_DIR = r'C:\UCLA\yolo model 2\train\images'

best_file, best_count = None, 0
for lpath in glob.glob(os.path.join(LABEL_DIR, '*.txt')):
    with open(lpath) as f:
        tokens = f.read().split()
    if len(tokens) < 5 + 48*3:
        continue
    kp = tokens[5:]
    n_vis = sum(1 for i in range(48) if float(kp[i*3+2]) == 2)
    if n_vis > best_count:
        best_count = n_vis
        best_file = lpath

print(f"Best label file: {best_file} ({best_count} visible keypoints)")

# Find matching image
stem = os.path.splitext(os.path.basename(best_file))[0]
img_path = None
for ext in ('.jpg', '.jpeg', '.png'):
    p = os.path.join(IMAGE_DIR, stem + ext)
    if os.path.exists(p):
        img_path = p
        break

if img_path is None:
    print("Could not find matching image")
    exit()

img = cv2.imread(img_path)
h, w = img.shape[:2]
print(f"Image: {img_path}  ({w}x{h})")

with open(best_file) as f:
    tokens = f.read().split()
kp_toks = tokens[5:]

for i in range(48):
    kpx = float(kp_toks[i*3])
    kpy = float(kp_toks[i*3+1])
    kpc = float(kp_toks[i*3+2])
    if kpc < 1:
        continue
    px = int(kpx * w)
    py = int(kpy * h)
    col = (0, 255, 0) if kpc == 2 else (0, 128, 255)
    cv2.circle(img, (px, py), 6, col, -1)
    cv2.circle(img, (px, py), 6, (0,0,0), 1)
    cv2.putText(img, str(i), (px+5, py-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(img, str(i), (px+5, py-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

out_path = r'C:\UCLA\yolo model 2\kp_ground_truth.jpg'
cv2.imwrite(out_path, img)
print(f"Saved: {out_path}")

# Also print pixel positions of visible keypoints
print("\n=== Visible keypoints in best image ===")
with open(best_file) as f:
    tokens = f.read().split()
kp_toks = tokens[5:]
for i in range(48):
    kpx = float(kp_toks[i*3])
    kpy = float(kp_toks[i*3+1])
    kpc = float(kp_toks[i*3+2])
    if kpc < 1:
        continue
    px = int(kpx * w)
    py = int(kpy * h)
    print(f"  kp{i:2d}: px=({px:4d},{py:4d}) conf={int(kpc)}")
