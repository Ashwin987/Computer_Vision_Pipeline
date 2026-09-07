"""
6-point least-squares H from well-spread seeds:
  kp0  (883,247) → (16.5,  0.0)  left penalty-area × far touchline
  kp8  (744,328) → (16.5, 13.84) left penalty-area top-right corner
  kp13 (1402,274)→ (52.5,  0.0)  halfway × far touchline   [confirmed]
  kp14 (1407,399)→ (52.5, 34.0)  halfway center             [confirmed]
  kp15 (836, 446)→ (43.35,34.0)  centre-circle left tangent [confirmed]
  kp16 (1403,569)→ (52.5, 68.0)  halfway × near touchline   [confirmed]

kp0 and kp8 are on the same world line (x=16.5m) → provides strong x-constraint.
6 points → least-squares H, better conditioned than 4-point exact.
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

SEED = {
     0: (16.5,   0.00),   # left penalty-area line × far touchline [assumed]
     8: (16.5,  13.84),   # left penalty-area top-right corner     [assumed]
    13: (52.5,   0.00),   # halfway × far touchline                [confirmed]
    14: (52.5,  34.00),   # halfway center                         [confirmed]
    15: (43.35, 34.00),   # centre-circle left tangent             [confirmed]
    16: (52.5,  68.00),   # halfway × near touchline               [confirmed]
}

pix_seed   = np.array([[kp_xy[i][0], kp_xy[i][1]] for i in SEED], dtype=np.float32)
world_seed = np.array([SEED[i]                    for i in SEED], dtype=np.float32)

# Method 0 with 6 pts = least-squares (over-determined DLT)
H, _ = cv2.findHomography(pix_seed, world_seed, 0)
if H is None:
    print("ERROR: findHomography returned None"); sys.exit(1)

def proj(px, py):
    wp = cv2.perspectiveTransform(np.array([[[float(px), float(py)]]], dtype=np.float32), H)
    return float(wp[0,0,0]), float(wp[0,0,1])

print("=== Seed back-projection residuals ===")
for i, (wx, wy) in SEED.items():
    px, py = kp_xy[i]
    ex, ey = proj(px, py)
    print(f"  kp{i:2d}: ({px:.0f},{py:.0f}) → ({ex:.2f},{ey:.2f})  "
          f"expected=({wx},{wy})  err=({ex-wx:.2f},{ey-wy:.2f})")

cx, cy = fw/2.0, fh/2.0
fcx, fcy = proj(cx, cy)
print(f"\nFrame centre → ({fcx:.1f}, {fcy:.1f})")
print(f"(left-half view, expect x≈20–35, y≈34)")

print("\n=== All detected kp → back-projected world ===")
print(f"{'idx':>4}  {'px_x':>6} {'px_y':>6}  {'wx':>7} {'wy':>7}  {'conf':>5}  snap_y")
PITCH_Y = [0, 13.84, 24.84, 30.34, 34, 37.66, 43.16, 54.16, 68]
PITCH_X = [0, 5.5, 11, 16.5, 20.15, 43.35, 52.5, 61.65, 84.85, 88.5, 94, 99.5, 105]

def snap(v, vals):
    return min(vals, key=lambda x: abs(x-v))

for i, (pt, c) in enumerate(zip(kp_xy, kp_conf)):
    if c < CONF_THR: continue
    px, py = float(pt[0]), float(pt[1])
    wx, wy = proj(px, py)
    sy = snap(wy, PITCH_Y)
    sx = snap(wx, PITCH_X)
    note = " ← seed" if i in SEED else ""
    print(f"{i:>4}  {px:>6.0f} {py:>6.0f}  {wx:>7.2f} {wy:>7.2f}  {c:>5.3f}  "
          f"snap=({sx},{sy}){note}")

# Debug image
debug = frame.copy()
H_inv = np.linalg.inv(H)
corners_w = np.array([[[0,0],[105,0],[105,68],[0,68]]], dtype=np.float32)
corners_px = cv2.perspectiveTransform(corners_w, H_inv).reshape(-1,2).astype(np.int32)
cv2.polylines(debug, [corners_px], True, (0,255,0), 2)
for p, lbl in zip(corners_px, ["(0,0)","(105,0)","(105,68)","(0,68)"]):
    cv2.putText(debug, lbl, (p[0]+4,p[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
for i, (pt, c) in enumerate(zip(kp_xy, kp_conf)):
    if c < CONF_THR: continue
    px, py = int(pt[0]), int(pt[1])
    wx, wy = proj(px, py)
    col = (0,255,255) if i in SEED else (0,100,255)
    cv2.circle(debug, (px, py), 5, col, -1)
    cv2.putText(debug, f"{i}:({wx:.0f},{wy:.0f})", (px+3,py-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255,255,255), 1)
cv2.imwrite(r'C:\UCLA\yolo model 2\diag_backproject3.jpg', debug)
print("\nSaved: diag_backproject3.jpg")
