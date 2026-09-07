"""
Prints RANSAC inliers and tests a proposed corrected PITCH_KP_WORLD table.

The key finding is that only 9/39 correspondences are inliers, meaning ~30 world
coordinate assignments are wrong.  This script identifies the inliers and then
tests a corrected table derived from careful geometric analysis of the training
label data and visual inspection.
"""
import cv2, numpy as np, sys
sys.path.insert(0, r'C:\UCLA\yolo model 2')
from ultralytics import YOLO
from pitch_calibrator import PITCH_KP_WORLD as CURRENT

MODEL_PATH = r'C:\UCLA\yolo model 2\pose\pitch_keypoints_v3\weights\best.pt'
VIDEO_PATH = r'C:\UCLA\yolo model 2\Match_videos\121364_0.mp4'
CONF_THR   = 0.15

# ── Load frame ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
ret, frame = cap.read()
cap.release()
fh, fw = frame.shape[:2]

model = YOLO(MODEL_PATH)
res = model.predict(source=frame, conf=0.1, imgsz=1280, save=False, verbose=False)
kp_xy   = res[0].keypoints.xy[0].cpu().numpy()
kp_conf = res[0].keypoints.conf[0].cpu().numpy()

# ── Current table inlier analysis ────────────────────────────────────────────
idx_list, pix_list, world_list = [], [], []
for i, (pt, c) in enumerate(zip(kp_xy, kp_conf)):
    if c >= CONF_THR and i < len(CURRENT):
        idx_list.append(i)
        pix_list.append([pt[0], pt[1]])
        world_list.append(CURRENT[i].tolist())

pix_arr   = np.array(pix_list,   dtype=np.float32)
world_arr = np.array(world_list, dtype=np.float32)

H, mask = cv2.findHomography(pix_arr, world_arr, cv2.RANSAC, 8.0)
print("=== CURRENT TABLE: RANSAC inliers ===")
for k, (i, pt, wd) in enumerate(zip(idx_list, pix_arr, world_arr)):
    status = "INLIER " if mask[k] else "outlier"
    print(f"  kp{i:2d}  pix=({pt[0]:6.0f},{pt[1]:6.0f})  "
          f"world=({wd[0]:6.2f},{wd[1]:5.2f})  {status}")

if H is not None:
    pt = np.array([[[fw/2, fh/2]]], dtype=np.float32)
    r = cv2.perspectiveTransform(pt, H)
    print(f"\nFrame centre → ({r[0,0,0]:.1f}, {r[0,0,1]:.1f})  (expect ~26–35, 34 for left-half view)")

# ── Proposed corrected table ──────────────────────────────────────────────────
# Derived by:
#  - Confirming kp13=(52.5,0), kp14=(52.5,34), kp16=(52.5,68) from vertical-line analysis
#  - kp15=(43.35,34) center circle left tangent confirmed from centre position
#  - Mapping avg_x statistics to world_x for left-half and right-half visible points
#  - Using avg_y statistics: low y = far touchline (y≈0), high y = near touchline (y≈68)
#  - Cross-checking with known pitch geometry
#
# avg_x → world_x rough guide (for left-half-only visible points): world_x ≈ avg_x * 70
# avg_y → world_y rough guide: world_y ≈ avg_y * 68
#
# Keypoints 0-12 are left-side features (avg_x < 0.37)
# Keypoints 13-17 are center features
# Keypoints 18-47 are a mix of right-side, center, and near-touchline
#
# Based on analysis of avg positions and visual inspection:
PROPOSED = np.array([
    # ── 0-12: left side ──────────────────────────────────────────────────────
    # avg_x=0.253,avg_y=0.198 → x≈17.7m(≈16.5), y≈13.5m(≈13.84) → penalty area TRC
    [ 16.5,  13.84],  #  0  left penalty-area top-right corner
    # avg_x=0.210,avg_y=0.249 → x≈14.7m, y≈16.9m → near penalty spot depth
    [ 11.0,  13.84],  #  1  penalty-spot x × penalty-area top
    # avg_x=0.180,avg_y=0.300 → x≈12.6m, y≈20.4m → goal-area top-right
    [  5.5,  24.84],  #  2  left goal-area top-right corner
    # avg_x=0.036,avg_y=0.425 → x≈2.5m, y≈28.9m → goal-line × goal-area area
    [  0.0,  13.84],  #  3  left goal line × penalty-area top  ← very left
    # avg_x=N/A     → not visible → keep as TL corner
    [  0.0,   0.00],  #  4  TL corner (left goal line × far touchline)
    # avg_x=N/A     → not visible → keep as BL corner
    [  0.0,  68.00],  #  5  BL corner (left goal line × near touchline)
    # avg_x=0.217,avg_y=0.328 → x≈15.2m, y≈22.3m → near goal area
    [  5.5,  43.16],  #  6  left goal-area bottom-right corner
    # avg_x=0.101,avg_y=0.446 → x≈7.1m, y≈30.3m → near goal area
    [  5.5,  24.84],  #  7  goal area inner top (dup; will refine)
    # avg_x=0.365,avg_y=0.306 → x≈25.6m, y≈20.8m → left of penalty arc
    [ 16.5,  54.16],  #  8  left penalty-area bottom-right corner
    # avg_x=0.289,avg_y=0.385 → x≈20.2m, y≈26.2m → penalty arc area
    [ 20.15, 34.00],  #  9  left penalty arc apex
    # avg_x=0.206,avg_y=0.490 → x≈14.4m, y≈33.3m → near penalty spot height
    [ 11.0,  34.00],  # 10  left penalty spot
    # avg_x=0.138,avg_y=0.638 → x≈9.7m, y≈43.4m → goal area level
    [  5.5,  43.16],  # 11  left goal-area bottom-right (dup of 6)
    # avg_x=0.206,avg_y=0.411 → x≈14.4m, y≈28.0m → penalty area interior
    [ 16.5,  13.84],  # 12  (dup of 0 — will need refinement)
    # ── 13-17: center/halfway ────────────────────────────────────────────────
    [ 52.5,   0.00],  # 13  halfway × far touchline  ← CONFIRMED
    [ 52.5,  34.00],  # 14  halfway center            ← CONFIRMED
    [ 43.35, 34.00],  # 15  centre-circle left tangent ← CONFIRMED
    [ 52.5,  68.00],  # 16  halfway × near touchline  ← CONFIRMED
    [ 61.65, 34.00],  # 17  centre-circle right tangent (avg_x=0.551 → just right of halfway)
    # ── 18-47: right side + misc ─────────────────────────────────────────────
    [  0.0,  43.16],  # 18  left goal line × goal-area bottom (avg_x=0.426,avg_y=0.967 → near touchline left)
    [105.0,   0.00],  # 19  TR corner (avg_x=N/A → invisible mostly)
    [105.0,  68.00],  # 20  BR corner (avg_x=N/A)
    [105.0,  54.16],  # 21  right goal line × penalty-area bottom (avg_x=0.933)
    [ 88.5,  13.84],  # 22  right penalty-area top-left corner (avg_x=0.816)
    [ 88.5,  54.16],  # 23  right penalty-area bottom-left corner (avg_x=0.730)
    [ 61.65, 34.00],  # 24  centre-circle right tangent (avg_x=0.623) ← dup of 17
    [ 99.5,  13.84],  # 25  right goal-area top-left (avg_x=0.901)
    [ 88.5,  13.84],  # 26  right penalty-area top (avg_x=0.743) ← similar to 22
    [ 52.5,  24.85],  # 27  halfway × centre-circle top (avg_x=0.594,avg_y=0.301)
    [ 52.5,  43.15],  # 28  halfway × centre-circle bottom (avg_x=0.612,avg_y=0.394)
    [ 88.5,  54.16],  # 29  right penalty-area bottom (avg_x=0.715)
    [ 99.5,  43.16],  # 30  right goal-area bottom (avg_x=0.843)
    [ 88.5,  54.16],  # 31  dup (avg_x=0.739)
    [ 16.5,  26.69],  # 32  left penalty arc × penalty-area top (avg_x=0.428,avg_y=0.236)
    [ 52.5,  24.85],  # 33  halfway × circle top (avg_x=0.539,avg_y=0.254)
    [ 52.5,   0.00],  # 34  halfway × far touchline (avg_x=0.542,avg_y=0.239) ← dup of 13
    [ 88.5,   0.00],  # 35  right penalty-area × far touchline (avg_x=0.615,avg_y=0.230)
    [ 16.5,  68.00],  # 36  left penalty-area × near touchline (avg_x=N/A)
    [ 16.5,  68.00],  # 37  dup (avg_x=0.428,avg_y=0.953 → near touchline left area)
    [ 88.5,  68.00],  # 38  right penalty-area × near touchline (avg_x=0.639,avg_y=0.941)
    [ 16.5,  68.00],  # 39  dup (avg_x=N/A)
    [ 52.5,   0.00],  # 40  halfway × far touchline (avg_x=0.510,avg_y=0.254) ← likely same as 13
    [ 88.5,   0.00],  # 41  right penalty-area × far touchline (avg_x=0.565,avg_y=0.224)
    [ 88.5,  68.00],  # 42  right penalty-area × near touchline (avg_x=0.147,avg_y=0.895) ← contradicts
    [  5.5,   0.00],  # 43  left goal-area × far touchline (avg_x=0.903,avg_y=0.920) ← contradicts
    [  5.5,  68.00],  # 44  left goal-area × near touchline (avg_x=0.149,avg_y=0.332)
    [  0.0,  30.34],  # 45  left goal post (avg_x=0.081,avg_y=0.381)
    [ 99.5,  43.16],  # 46  right goal-area bottom (avg_x=0.836,avg_y=0.403)
    [ 99.5,  24.84],  # 47  right goal-area top (avg_x=0.908,avg_y=0.455)
], dtype=np.float32)

print("\n\n=== PROPOSED TABLE sanity check (same frame) ===")
idx2, pix2, world2 = [], [], []
for i, (pt, c) in enumerate(zip(kp_xy, kp_conf)):
    if c >= CONF_THR and i < len(PROPOSED):
        idx2.append(i)
        pix2.append([pt[0], pt[1]])
        world2.append(PROPOSED[i].tolist())
p2 = np.array(pix2,   dtype=np.float32)
w2 = np.array(world2, dtype=np.float32)
H2, mask2 = cv2.findHomography(p2, w2, cv2.RANSAC, 8.0)
inliers2 = int(mask2.sum()) if mask2 is not None else 0
print(f"RANSAC inliers: {inliers2}/{len(p2)}")
if H2 is not None:
    pt = np.array([[[fw/2, fh/2]]], dtype=np.float32)
    r = cv2.perspectiveTransform(pt, H2)
    print(f"Frame centre → ({r[0,0,0]:.1f}, {r[0,0,1]:.1f})")

    debug = frame.copy()
    H2_inv = np.linalg.inv(H2)
    c_w = np.array([[[0,0],[105,0],[105,68],[0,68]]], dtype=np.float32)
    c_px = cv2.perspectiveTransform(c_w, H2_inv).reshape(-1,2).astype(np.int32)
    cv2.polylines(debug, [c_px], True, (0,255,0), 3)
    for p, lbl in zip(c_px, ["(0,0)","(105,0)","(105,68)","(0,68)"]):
        cv2.putText(debug, lbl, (p[0]+5, p[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imwrite(r'C:\UCLA\yolo model 2\diag_hom_proposed.jpg', debug)
    print("Saved: diag_hom_proposed.jpg")
