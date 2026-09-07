"""
Use 4 non-degenerate seed points (no 3 collinear in pixel OR world space)
to compute H and back-project all detected keypoints.

Seeds:
  kp0  (883,247) → (16.5,  0.0)  left penalty-area line × far touchline
  kp13 (1402,274) → (52.5,  0.0)  halfway × far touchline   [confirmed]
  kp14 (1407,399) → (52.5, 34.0)  halfway center             [confirmed]
  kp15 (836, 445) → (43.35,34.0)  centre-circle left tangent [confirmed]

Sanity check: kp16 (not in seeds) should back-project to ~(52.5, 68).
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

# avg_y calibration reference points (halfway-line, visible from both sides → clean avg):
#   kp13 (52.5, 0)  avg_y=0.244
#   kp14 (52.5, 34) avg_y=0.408
#   kp16 (52.5, 68) avg_y=0.615
AVG_Y_TABLE = {
    0: 0.1982, 1: 0.2490, 2: 0.2998, 3: 0.4245, 6: 0.3280, 7: 0.4462,
    8: 0.3055, 9: 0.3849, 10: 0.4895, 11: 0.6383, 12: 0.4106,
    13: 0.2438, 14: 0.4079, 15: 0.5117, 16: 0.6147, 17: 0.4868,
    18: 0.9665, 21: 0.5671, 22: 0.3497, 23: 0.2658, 24: 0.1932,
    25: 0.5549, 26: 0.3590, 27: 0.3014, 28: 0.3940, 29: 0.5608,
    30: 0.8216, 31: 0.4600, 32: 0.2361, 33: 0.2540, 34: 0.2389,
    35: 0.2303, 37: 0.9532, 38: 0.9412, 40: 0.2539, 41: 0.2244,
    42: 0.8947, 43: 0.9197, 44: 0.3317, 45: 0.3813, 46: 0.4033, 47: 0.4552,
}
# avg_y → world_y linear fit from confirmed (y=0→0.244), (y=34→0.408), (y=68→0.615)
def avg_y_to_world(ay):
    return (ay - 0.244) / (0.615 - 0.244) * 68.0

# Non-degenerate seed set: no 3 collinear in pixel OR world space
SEED = {
     0: (16.5,  0.00),   # left penalty-area line × far touchline
    13: (52.5,  0.00),   # halfway × far touchline          [confirmed]
    14: (52.5, 34.00),   # halfway center                   [confirmed]
    15: (43.35, 34.00),  # centre-circle left tangent       [confirmed]
}

pix_seed   = np.array([[kp_xy[i][0], kp_xy[i][1]] for i in SEED], dtype=np.float32)
world_seed = np.array([SEED[i]                    for i in SEED], dtype=np.float32)

H, _ = cv2.findHomography(pix_seed, world_seed, 0)  # exact 4-point, no RANSAC
if H is None:
    print("ERROR: findHomography returned None")
    sys.exit(1)

def proj(px, py):
    wp = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), H)
    return float(wp[0,0,0]), float(wp[0,0,1])

# Sanity checks
print("=== Seed back-projection (should match exactly) ===")
for i, (wx, wy) in SEED.items():
    px, py = kp_xy[i]
    ex, ey = proj(px, py)
    print(f"  kp{i:2d}: px=({px:.0f},{py:.0f}) → ({ex:.2f},{ey:.2f})  expected=({wx},{wy})")

print("\n=== Sanity checks (not in seeds) ===")
for i, expected in [(16, (52.5, 68.0))]:
    px, py = kp_xy[i]
    if kp_conf[i] >= CONF_THR:
        ex, ey = proj(px, py)
        print(f"  kp{i:2d}: px=({px:.0f},{py:.0f}) → ({ex:.2f},{ey:.2f})  expected≈{expected}")

cx, cy = fw/2.0, fh/2.0
fcx, fcy = proj(cx, cy)
print(f"\nFrame centre → ({fcx:.1f}, {fcy:.1f})")
print(f"(left-half view: expect ~20–35 in x, ~34 in y)")

print("\n=== All detected kp → back-projected world coords ===")
print(f"{'idx':>4}  {'px_x':>6} {'px_y':>6}  {'wx':>7} {'wy':>7}  {'avg_y_wy':>9}  {'conf':>5}")
for i, (pt, c) in enumerate(zip(kp_xy, kp_conf)):
    if c < CONF_THR:
        continue
    px, py = float(pt[0]), float(pt[1])
    wx, wy = proj(px, py)
    awy = avg_y_to_world(AVG_Y_TABLE.get(i, float('nan')))
    note = " ← seed" if i in SEED else ""
    print(f"{i:>4}  {px:>6.0f} {py:>6.0f}  {wx:>7.2f} {wy:>7.2f}  {awy:>9.2f}  {c:>5.3f}{note}")

# Draw debug image
debug = frame.copy()
H_inv = np.linalg.inv(H)
corners_w = np.array([[[0,0],[105,0],[105,68],[0,68]]], dtype=np.float32)
corners_px = cv2.perspectiveTransform(corners_w, H_inv).reshape(-1,2).astype(np.int32)
cv2.polylines(debug, [corners_px], True, (0,255,0), 2)
for p, lbl in zip(corners_px, ["(0,0)","(105,0)","(105,68)","(0,68)"]):
    cv2.putText(debug, lbl, (p[0]+4, p[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

for i, (pt, c) in enumerate(zip(kp_xy, kp_conf)):
    if c < CONF_THR: continue
    px, py = int(pt[0]), int(pt[1])
    wx, wy = proj(px, py)
    col = (0,255,255) if i in SEED else (0,100,255)
    cv2.circle(debug, (px, py), 5, col, -1)
    cv2.putText(debug, f"{i}:({wx:.0f},{wy:.0f})", (px+3, py-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255,255,255), 1)

cv2.imwrite(r'C:\UCLA\yolo model 2\diag_backproject2.jpg', debug)
print("\nSaved: diag_backproject2.jpg")
