"""
Diagnostic: validate candidate PITCH_KP_WORLD assignments for the 48 keypoints.

Uses the v3 model output on frame 300 of the match video, plus analysis of
the training label files, to test if the current or a proposed PITCH_KP_WORLD
produces a sensible homography for the match video.

Steps:
1. Run model on frame 300 → get detected kp pixel positions
2. Try current PITCH_KP_WORLD → compute H, check frame-centre world coords
3. Draw the projected pitch boundary on the frame and save for inspection
4. Also dump the detected kp pixel positions alongside the world coords in the
   PROPOSED table so we can cross-check them visually.
"""

import cv2, numpy as np, sys
sys.path.insert(0, r'C:\UCLA\yolo model 2')
from ultralytics import YOLO

MODEL_PATH  = r'C:\UCLA\yolo model 2\pose\pitch_keypoints_v3\weights\best.pt'
VIDEO_PATH  = r'C:\UCLA\yolo model 2\Match_videos\121364_0.mp4'
FRAME_IDX   = 300
CONF_THR    = 0.15

# ── Current PITCH_KP_WORLD ────────────────────────────────────────────────────
from pitch_calibrator import PITCH_KP_WORLD as CURRENT_TABLE

# ── Load frame ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
ret, frame = cap.read()
cap.release()
assert ret, "cannot read frame"
fh, fw = frame.shape[:2]
print(f"Frame {FRAME_IDX}: {fw}x{fh}")

# ── Run model ─────────────────────────────────────────────────────────────────
model = YOLO(MODEL_PATH)
res = model.predict(source=frame, conf=0.1, imgsz=1280, save=False, verbose=False)
kp_xy   = res[0].keypoints.xy[0].cpu().numpy()    # (48,2)
kp_conf = res[0].keypoints.conf[0].cpu().numpy()  # (48,)

# ── Build point correspondences ───────────────────────────────────────────────
pix_pts   = []
world_pts = []
for i, (pt, c) in enumerate(zip(kp_xy, kp_conf)):
    if c < CONF_THR or i >= len(CURRENT_TABLE):
        continue
    pix_pts.append([pt[0], pt[1]])
    world_pts.append(CURRENT_TABLE[i].tolist())

pix_arr   = np.array(pix_pts,   dtype=np.float32)
world_arr = np.array(world_pts, dtype=np.float32)

print(f"\n{len(pix_pts)} correspondences above conf={CONF_THR}")
print(f"{'idx':>4}  {'px_x':>6} {'px_y':>6}  {'w_x':>7} {'w_y':>7}")
for i, (kp_idx, pt, wd) in enumerate(zip(
        [i for i,c in enumerate(kp_conf) if c >= CONF_THR and i < len(CURRENT_TABLE)],
        pix_arr, world_arr)):
    print(f"{kp_idx:>4}  {pt[0]:>6.0f} {pt[1]:>6.0f}  {wd[0]:>7.2f} {wd[1]:>7.2f}")

# ── Solve homography ──────────────────────────────────────────────────────────
if len(pix_arr) >= 4:
    H, mask = cv2.findHomography(pix_arr, world_arr, cv2.RANSAC, 8.0)
    inliers = int(mask.sum()) if mask is not None else 0
    print(f"\nRANSAC inliers: {inliers}/{len(pix_arr)}")
    if H is not None:
        # Frame-centre world coords
        cx, cy = fw/2.0, fh/2.0
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        res_w = cv2.perspectiveTransform(pt, H)
        rx, ry = float(res_w[0,0,0]), float(res_w[0,0,1])
        print(f"Frame centre ({cx:.0f},{cy:.0f}) → world ({rx:.1f}, {ry:.1f})  (expect ~52.5, 34 for left-half)")

        # Pitch corners projected back to pixels
        H_inv = np.linalg.inv(H)
        corners_w = np.array([[[0,0],[105,0],[105,68],[0,68]]], dtype=np.float32)
        corners_px = cv2.perspectiveTransform(corners_w, H_inv).reshape(-1,2).astype(np.int32)
        debug = frame.copy()
        cv2.polylines(debug, [corners_px], True, (0,255,0), 3)
        # Label corners
        labels = ["(0,0)","(105,0)","(105,68)","(0,68)"]
        for p, lbl in zip(corners_px, labels):
            cv2.putText(debug, lbl, (p[0]+5, p[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        # Draw keypoint dots
        for i, (pt, c) in enumerate(zip(kp_xy, kp_conf)):
            if c < CONF_THR:
                continue
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(debug, (px, py), 5, (0,0,255), -1)
            cv2.putText(debug, str(i), (px+4, py-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        cv2.imwrite(r'C:\UCLA\yolo model 2\diag_hom_debug.jpg', debug)
        print("Saved: diag_hom_debug.jpg")
    else:
        print("findHomography returned None — correspondences degenerate")
else:
    print("Not enough correspondences to compute H")
