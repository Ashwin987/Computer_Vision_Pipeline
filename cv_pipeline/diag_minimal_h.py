"""
Compute H from the 4 keypoints I'm geometrically confident about and use that
to back-project ALL detected keypoints to world space.  The resulting world
positions tell me what each keypoint index corresponds to on the pitch.
"""
import cv2, numpy as np, sys
sys.path.insert(0, r'C:\UCLA\yolo model 2')
from ultralytics import YOLO

MODEL_PATH = r'C:\UCLA\yolo model 2\pose\pitch_keypoints_v3\weights\best.pt'
VIDEO_PATH = r'C:\UCLA\yolo model 2\Match_videos\121364_0.mp4'
CONF_THR   = 0.10

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
ret, frame = cap.read()
cap.release()
fh, fw = frame.shape[:2]

model = YOLO(MODEL_PATH)
res = model.predict(source=frame, conf=0.1, imgsz=1280, save=False, verbose=False)
kp_xy   = res[0].keypoints.xy[0].cpu().numpy()
kp_conf = res[0].keypoints.conf[0].cpu().numpy()

# ── Seed correspondences I'm geometrically confident about ────────────────────
# kp13: consistently at far-right edge of frame in LEFT-HALF view → halfway × far touchline
# kp14: same vertical line, middle height → halfway center
# kp16: same vertical line, lower → halfway × near touchline
# kp15: center of frame, mid-height → centre-circle left tangent
# kp27: upper-center area (avg_x=0.594, avg_y=0.301) → halfway × circle top
# kp28: center area (avg_x=0.612, avg_y=0.394) → halfway × circle bottom
# kp11: far-left area, lower (avg_x=0.138, avg_y=0.638) → near the goal-line area

SEED = {
    13: (52.5,  0.00),   # halfway × far touchline  ✓ confirmed
    14: (52.5, 34.00),   # halfway center            ✓ confirmed
    16: (52.5, 68.00),   # halfway × near touchline  ✓ confirmed
    15: (43.35, 34.00),  # centre-circle left tangent  ✓ confirmed
}

pix_seed  = np.array([[kp_xy[i][0], kp_xy[i][1]] for i in SEED], dtype=np.float32)
world_seed = np.array([SEED[i] for i in SEED], dtype=np.float32)

H_seed, _ = cv2.findHomography(pix_seed, world_seed, 0)  # no RANSAC, use all 4 pts

if H_seed is None:
    print("findHomography returned None!")
    exit()

cx, cy = fw/2.0, fh/2.0
centre_w = cv2.perspectiveTransform(np.array([[[cx, cy]]], dtype=np.float32), H_seed)
print(f"Frame centre → world ({centre_w[0,0,0]:.1f}, {centre_w[0,0,1]:.1f})")
print(f"(pitch is 0–105 × 0–68;  left-half view centre should be ~20–35, 34)")

# Back-project all detected keypoints
print("\n=== All detected kp → back-projected world coords ===")
print(f"{'idx':>4}  {'px_x':>6} {'px_y':>6}  {'wx':>7} {'wy':>7}  {'conf':>5}")
all_wx, all_wy = [], []
for i, (pt, c) in enumerate(zip(kp_xy, kp_conf)):
    if c < CONF_THR:
        continue
    px, py = float(pt[0]), float(pt[1])
    wp = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), H_seed)
    wx, wy = float(wp[0,0,0]), float(wp[0,0,1])
    all_wx.append(wx); all_wy.append(wy)
    note = ""
    if i in SEED: note = " ← seed"
    print(f"{i:>4}  {px:>6.0f} {py:>6.0f}  {wx:>7.2f} {wy:>7.2f}  {c:>5.3f}{note}")

# Draw pitch boundary
debug = frame.copy()
H_inv = np.linalg.inv(H_seed)
c_w = np.array([[[0,0],[105,0],[105,68],[0,68]]], dtype=np.float32)
c_px = cv2.perspectiveTransform(c_w, H_inv).reshape(-1,2).astype(np.int32)
cv2.polylines(debug, [c_px], True, (0,255,0), 3)
for p, lbl in zip(c_px, ["TL(0,0)","TR(105,0)","BR(105,68)","BL(0,68)"]):
    cv2.putText(debug, lbl, (p[0]+5, p[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

# Draw each keypoint with its back-projected world coords
for i, (pt, c) in enumerate(zip(kp_xy, kp_conf)):
    if c < CONF_THR:
        continue
    px, py = int(pt[0]), int(pt[1])
    wp = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), H_seed)
    wx, wy = float(wp[0,0,0]), float(wp[0,0,1])
    col = (0, 255, 255) if i in SEED else (0, 100, 255)
    cv2.circle(debug, (px, py), 5, col, -1)
    cv2.putText(debug, f"{i}:({wx:.0f},{wy:.0f})", (px+3, py-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

cv2.imwrite(r'C:\UCLA\yolo model 2\diag_backproject.jpg', debug)
print("\nSaved: diag_backproject.jpg")
print(f"wx range: {min(all_wx):.1f} – {max(all_wx):.1f}")
print(f"wy range: {min(all_wy):.1f} – {max(all_wy):.1f}")
