"""Print RANSAC inliers for the PROPOSED table in diag_homography2.py."""
import cv2, numpy as np, sys
sys.path.insert(0, r'C:\UCLA\yolo model 2')
from ultralytics import YOLO

MODEL_PATH = r'C:\UCLA\yolo model 2\pose\pitch_keypoints_v3\weights\best.pt'
VIDEO_PATH = r'C:\UCLA\yolo model 2\Match_videos\121364_0.mp4'
CONF_THR   = 0.15

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
ret, frame = cap.read(); cap.release()
fh, fw = frame.shape[:2]

model = YOLO(MODEL_PATH)
res = model.predict(source=frame, conf=0.1, imgsz=1280, save=False, verbose=False)
kp_xy   = res[0].keypoints.xy[0].cpu().numpy()
kp_conf = res[0].keypoints.conf[0].cpu().numpy()

PROPOSED = np.array([
    [ 16.5,  13.84],  #  0
    [ 11.0,  13.84],  #  1
    [  5.5,  24.84],  #  2
    [  0.0,  13.84],  #  3
    [  0.0,   0.00],  #  4
    [  0.0,  68.00],  #  5
    [  5.5,  43.16],  #  6
    [  5.5,  24.84],  #  7
    [ 16.5,  54.16],  #  8
    [ 20.15, 34.00],  #  9
    [ 11.0,  34.00],  # 10
    [  5.5,  43.16],  # 11
    [ 16.5,  13.84],  # 12
    [ 52.5,   0.00],  # 13
    [ 52.5,  34.00],  # 14
    [ 43.35, 34.00],  # 15
    [ 52.5,  68.00],  # 16
    [ 61.65, 34.00],  # 17
    [  0.0,  43.16],  # 18
    [105.0,   0.00],  # 19
    [105.0,  68.00],  # 20
    [105.0,  54.16],  # 21
    [ 88.5,  13.84],  # 22
    [ 88.5,  54.16],  # 23
    [ 61.65, 34.00],  # 24
    [ 99.5,  13.84],  # 25
    [ 88.5,  13.84],  # 26
    [ 52.5,  24.85],  # 27
    [ 52.5,  43.15],  # 28
    [ 88.5,  54.16],  # 29
    [ 99.5,  43.16],  # 30
    [ 88.5,  54.16],  # 31
    [ 16.5,  26.69],  # 32
    [ 52.5,  24.85],  # 33
    [ 52.5,   0.00],  # 34
    [ 88.5,   0.00],  # 35
    [ 16.5,  68.00],  # 36
    [ 16.5,  68.00],  # 37
    [ 88.5,  68.00],  # 38
    [ 16.5,  68.00],  # 39
    [ 52.5,   0.00],  # 40
    [ 88.5,   0.00],  # 41
    [ 88.5,  68.00],  # 42
    [  5.5,   0.00],  # 43
    [  5.5,  68.00],  # 44
    [  0.0,  30.34],  # 45
    [ 99.5,  43.16],  # 46
    [ 99.5,  24.84],  # 47
], dtype=np.float32)

idx_list, pix_list, world_list = [], [], []
for i, (pt, c) in enumerate(zip(kp_xy, kp_conf)):
    if c >= CONF_THR and i < len(PROPOSED):
        idx_list.append(i); pix_list.append([pt[0],pt[1]]); world_list.append(PROPOSED[i].tolist())

pa = np.array(pix_list, dtype=np.float32)
wa = np.array(world_list, dtype=np.float32)
H, mask = cv2.findHomography(pa, wa, cv2.RANSAC, 8.0)
print(f"RANSAC inliers: {int(mask.sum())}/{len(pa)}")

CONFIRMED = {13:(52.5,0), 14:(52.5,34), 15:(43.35,34), 16:(52.5,68)}
print("\nStatus of each correspondence:")
print(f"{'kp':>4}  {'px_x':>6} {'px_y':>6}  {'w_x':>7} {'w_y':>7}  status   correct?")
for k, (i, pt, wd) in enumerate(zip(idx_list, pa, wa)):
    status = "INLIER " if mask[k] else "outlier"
    correct = CONFIRMED.get(i)
    note = " ← confirmed ✓" if (correct and abs(wd[0]-correct[0])<0.01 and abs(wd[1]-correct[1])<0.01) else ""
    print(f"{i:>4}  {pt[0]:>6.0f} {pt[1]:>6.0f}  {wd[0]:>7.2f} {wd[1]:>7.2f}  {status}{note}")

if H is not None:
    pt = np.array([[[fw/2, fh/2]]], dtype=np.float32)
    r = cv2.perspectiveTransform(pt, H)
    print(f"\nFrame centre → ({r[0,0,0]:.1f}, {r[0,0,1]:.1f})")
    print("Expected for left-half view: x≈20-35, y≈34")
