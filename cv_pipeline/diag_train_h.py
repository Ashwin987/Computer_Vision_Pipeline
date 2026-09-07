"""
Compute H from 4 ground-truth label points in the Man City training image
(right-half view), then back-project all 16 visible keypoints to world space.

Seeds (training image pixel positions, confirmed):
  kp13 (  8,  77) → (52.5,  0.0)  halfway × far touchline  [confirmed]
  kp14 ( 20, 196) → (52.5, 34.0)  halfway center             [confirmed]
  kp16 ( 34, 341) → (52.5, 68.0)  halfway × near touchline   [confirmed]
  kp35 (767,  50) → (88.5,  0.0)  right penalty-area × far touchline [assumed]

kp35 assumption: in right-half view (1280px wide), the halfway line is at px_x=8
and the right goal line appears near px_x=1220 (from kp23 at 1220,71).
kp35 at px_x=767: (767-8)/(1220-8)=62.6% of the way from halfway to right goal.
world_x = 52.5 + 0.626*52.5 ≈ 85m → snap to 88.5 (right penalty-area line).
"""
import cv2, numpy as np

IMG_PATH   = r'C:\UCLA\yolo model 2\train\images\00-54-22-00-55-28-000850_jpg.rf.8742e214ed2d98be77e1166179af02c6.jpg'
LABEL_PATH = r'C:\UCLA\yolo model 2\train\labels\00-54-22-00-55-28-000850_jpg.rf.8742e214ed2d98be77e1166179af02c6.txt'

img = cv2.imread(IMG_PATH)
ih, iw = img.shape[:2]

with open(LABEL_PATH) as f:
    tokens = f.read().split()
kp_toks = tokens[5:]
kp_px = {}  # index → (px, py)
for i in range(48):
    kpx, kpy, kpc = float(kp_toks[i*3]), float(kp_toks[i*3+1]), float(kp_toks[i*3+2])
    if kpc >= 2:
        kp_px[i] = (kpx*iw, kpy*ih)

print(f"Image: {iw}x{ih},  visible keypoints: {sorted(kp_px.keys())}")

SEED = {
    14: (52.5,  34.0),   # halfway center            [confirmed]
    16: (52.5,  68.0),   # halfway x near touchline  [confirmed]
    17: (61.65, 34.0),   # centre-circle right tangent [assumed, avg_y->43 but perspective bias ok]
    35: (88.5,   0.0),   # right penalty-area x far touchline [assumed]
}
# sanity check: kp13=(52.5,0) not in seeds — should back-project near (52.5,0)
pix_seed   = np.array([[kp_px[i][0], kp_px[i][1]] for i in SEED], dtype=np.float32)
world_seed = np.array([SEED[i]                    for i in SEED], dtype=np.float32)

H, _ = cv2.findHomography(pix_seed, world_seed, 0)
if H is None:
    print("ERROR: None H"); exit()

def proj(px, py):
    wp = cv2.perspectiveTransform(np.array([[[float(px),float(py)]]], dtype=np.float32), H)
    return float(wp[0,0,0]), float(wp[0,0,1])

PITCH_Y = [0, 13.84, 24.84, 30.34, 34, 37.66, 43.16, 54.16, 68]
PITCH_X = [0, 5.5, 11, 16.5, 20.15, 43.35, 52.5, 61.65, 84.85, 88.5, 94, 99.5, 105]
def snap(v, vals): return min(vals, key=lambda x: abs(x-v))

print("\nSeed residuals:")
for i, (wx,wy) in SEED.items():
    ex, ey = proj(*kp_px[i])
    print(f"  kp{i:2d}: px=({kp_px[i][0]:.0f},{kp_px[i][1]:.0f}) -> ({ex:.1f},{ey:.1f}) expected=({wx},{wy})")

print(f"\nAll visible keypoints back-projected:")
print(f"{'kp':>4}  {'px_x':>6} {'px_y':>6}  {'wx':>7} {'wy':>7}  snap_x snap_y  (note)")
for i in sorted(kp_px.keys()):
    px, py = kp_px[i]
    wx, wy = proj(px, py)
    sx, sy = snap(wx, PITCH_X), snap(wy, PITCH_Y)
    note = " <seed>" if i in SEED else ""
    print(f"{i:>4}  {px:>6.0f} {py:>6.0f}  {wx:>7.2f} {wy:>7.2f}  {sx:>6.2f} {sy:>6.2f}{note}")

# Draw debug image
debug = img.copy()
H_inv = np.linalg.inv(H)
corners_w = np.array([[[52.5,0],[105,0],[105,68],[52.5,68]]], dtype=np.float32)  # right half
corners_px = cv2.perspectiveTransform(corners_w, H_inv).reshape(-1,2).astype(np.int32)
cv2.polylines(debug, [corners_px], True, (0,255,0), 2)
for p, lbl in zip(corners_px, ["(52.5,0)","(105,0)","(105,68)","(52.5,68)"]):
    cv2.putText(debug, lbl, (p[0]+3,p[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
for i in sorted(kp_px.keys()):
    px, py = int(kp_px[i][0]), int(kp_px[i][1])
    wx, wy = proj(px, py)
    col = (0,255,255) if i in SEED else (0,100,255)
    cv2.circle(debug, (px,py), 5, col, -1)
    cv2.putText(debug, f"{i}:({wx:.0f},{wy:.0f})", (px+3,py-3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255,255,255), 1)
cv2.imwrite(r'C:\UCLA\yolo model 2\diag_train_backproject.jpg', debug)
print("\nSaved: diag_train_backproject.jpg")
