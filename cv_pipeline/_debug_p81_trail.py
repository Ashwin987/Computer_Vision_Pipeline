"""
Before/after debug images for player 81's trail (render_output6 fix).
Renders frame 492 twice: once with old logic, once with the three fixes applied.
"""
import sys, os, pickle
import numpy as np
import cv2
from collections import deque
sys.path.insert(0, r'c:\UCLA\yolo model 2')

from trackers import Tracker
from camera_movement_estimator import CameraMovementEstimator

TRACK_STUB  = 'stubs/track_stubs_121364.pkl'
CAM_STUB    = 'stubs/camera_movement_stub_121364.pkl'
RAW_VIDEO   = 'Match_videos/121364_0.mp4'
SRC_VIDEO   = 'output_videos/output_video.avi'
TARGET_PID  = 81
VIEW_FRAME  = 450       # frame where fn=412 spike is still in the 72-frame window
FPS         = 24
TRAIL_LEN   = FPS * 3  # 72 frames

# -- constants matching render_output6 -----------------------------------------
T1_COLOR            = (0, 0, 220)
T2_COLOR            = (0, 220, 0)
JUMP_PX             = 50
RECORD_JUMP_PX      = 25
OUTLIER_NEIGHBOR_PX = 35

# ── Load stubs and compute positions ──────────────────────────────────────────
print("Loading stubs / computing positions...")
with open(TRACK_STUB, 'rb') as f: tracks = pickle.load(f)
with open(CAM_STUB,   'rb') as f: cam_mv = pickle.load(f)
n = len(tracks['players'])

tracker = Tracker('models/best.pt')
tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
tracker.add_position_to_tracks(tracks)

cap = cv2.VideoCapture(RAW_VIDEO)
ret, frame0 = cap.read(); cap.release()
cam_est = CameraMovementEstimator(frame0)
cam_est.add_adjust_positions_to_tracks(tracks, cam_mv)

# Prefix sums
prefix_x = np.zeros(n, dtype=np.float64)
prefix_y = np.zeros(n, dtype=np.float64)
for fn in range(1, n):
    dx, dy = float(cam_mv[fn][0]), float(cam_mv[fn][1])
    prefix_x[fn] = prefix_x[fn-1] + dx
    prefix_y[fn] = prefix_y[fn-1] + dy

# ── Source frame ──────────────────────────────────────────────────────────────
cap2 = cv2.VideoCapture(SRC_VIDEO)
cap2.set(cv2.CAP_PROP_POS_FRAMES, VIEW_FRAME)
ret2, base_frame = cap2.read(); cap2.release()
if not ret2:
    print(f"Cannot read frame {VIEW_FRAME} from {SRC_VIDEO}")
    sys.exit(1)
orig_h, orig_w = base_frame.shape[:2]

# ── Helper: build trail up to a given frame, optionally with recording filter ─
def build_trail(frame_num, use_record_filter):
    trail = deque(maxlen=TRAIL_LEN)
    for fn in range(max(0, frame_num - TRAIL_LEN + 1), frame_num + 1):
        if TARGET_PID not in tracks['players'][fn]:
            continue
        info = tracks['players'][fn][TARGET_PID]
        adj  = info.get('position_adjusted')
        if adj is None:
            continue
        x, y = float(adj[0]), float(adj[1])
        if x == 0.0 and y == 0.0:
            continue
        if use_record_filter and trail:
            _, lx, ly = trail[-1]
            if (x-lx)**2 + (y-ly)**2 > RECORD_JUMP_PX**2:
                continue
        trail.append((fn, x, y))
    return list(trail)


def compute_cam_pts(trail, frame_num):
    cam_pts = []
    for fn, x_adj, y_adj in trail:
        cum_dx = prefix_x[frame_num] - prefix_x[fn]
        cum_dy = prefix_y[frame_num] - prefix_y[fn]
        draw_x = int(round(x_adj - cum_dx))
        draw_y = int(round(y_adj - cum_dy))
        cam_pts.append((fn, x_adj, y_adj, draw_x, draw_y))
    return cam_pts


def apply_outlier_rejection(cam_pts):
    if len(cam_pts) < 3:
        return cam_pts
    clean = [cam_pts[0]]
    for i in range(1, len(cam_pts) - 1):
        _, _, _, cx, cy = cam_pts[i]
        _, _, _, px, py = cam_pts[i-1]
        _, _, _, nx, ny = cam_pts[i+1]
        avg_x = (px + nx) * 0.5
        avg_y = (py + ny) * 0.5
        if (cx-avg_x)**2 + (cy-avg_y)**2 <= OUTLIER_NEIGHBOR_PX**2:
            clean.append(cam_pts[i])
    clean.append(cam_pts[-1])
    return clean


def draw_trail(frame, cam_pts, color, draw_all_pts=True):
    """Draw the trail; return count of segments drawn vs filtered."""
    drawn = filtered = 0
    n_pts = len(cam_pts)
    for i in range(1, n_pts):
        _, _, _, dx1, dy1 = cam_pts[i-1]
        _, _, _, dx2, dy2 = cam_pts[i]
        if not (0 <= dx1 < orig_w and 0 <= dy1 < orig_h
                and 0 <= dx2 < orig_w and 0 <= dy2 < orig_h):
            continue
        disp = np.sqrt((dx2-dx1)**2 + (dy2-dy1)**2)
        if disp > JUMP_PX:
            filtered += 1
            continue
        t     = i / n_pts
        alpha = 0.20 + 0.80 * t
        col   = tuple(int(c * alpha) for c in color)
        cv2.line(frame, (dx1, dy1), (dx2, dy2), col, 3, cv2.LINE_AA)
        drawn += 1

    # Mark each stored trail point
    if draw_all_pts:
        for _, _, _, dx, dy in cam_pts:
            if 0 <= dx < orig_w and 0 <= dy < orig_h:
                cv2.circle(frame, (dx, dy), 3, (255, 255, 0), -1)

    # Highlight the current (newest) point
    if cam_pts:
        _, _, _, cx, cy = cam_pts[-1]
        cv2.circle(frame, (cx, cy), 8, (0, 255, 255), 2)
    return drawn, filtered


# ── BEFORE — no recording filter, no outlier rejection ───────────────────────
trail_before = build_trail(VIEW_FRAME, use_record_filter=False)
cam_pts_before = compute_cam_pts(trail_before, VIEW_FRAME)

# Find the stray segment in the before trail and highlight it
stray_i = None
for i in range(1, len(cam_pts_before)):
    _, _, _, dx1, dy1 = cam_pts_before[i-1]
    _, _, _, dx2, dy2 = cam_pts_before[i]
    disp = np.sqrt((dx2-dx1)**2 + (dy2-dy1)**2)
    if 25 < disp <= JUMP_PX:
        stray_i = i

frame_before = base_frame.copy()
drawn_b, filtered_b = draw_trail(frame_before, cam_pts_before, T2_COLOR)
# Highlight the stray segment in red
if stray_i is not None:
    _, _, _, sx1, sy1 = cam_pts_before[stray_i-1]
    _, _, _, sx2, sy2 = cam_pts_before[stray_i]
    cv2.line(frame_before, (sx1, sy1), (sx2, sy2), (0, 0, 255), 4, cv2.LINE_AA)
    cv2.circle(frame_before, (sx1, sy1), 6, (0, 0, 255), -1)
    cv2.circle(frame_before, (sx2, sy2), 6, (0, 0, 255), -1)
    mid_x, mid_y = (sx1+sx2)//2, (sy1+sy2)//2
    cv2.putText(frame_before, f"stray {int(np.sqrt((sx2-sx1)**2+(sy2-sy1)**2))}px",
                (mid_x+5, mid_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

# Labels
for thick, col in ((3,(0,0,0)),(1,(255,255,255))):
    cv2.putText(frame_before, f"BEFORE  Player 81 trail  frame {VIEW_FRAME}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, thick)
    cv2.putText(frame_before,
                f"trail pts={len(cam_pts_before)}  drawn={drawn_b}  filtered={filtered_b}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, thick)
    cv2.putText(frame_before, "Red segment = drawn stray (cam spike leaked through)",
                (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, thick)
    cv2.putText(frame_before, "Yellow dots = stored trail pts | Cyan ring = player now",
                (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, thick)

path_before = "output_videos/output6_dbg_p81_before.jpg"
cv2.imwrite(path_before, frame_before)
print(f"Saved: {path_before}  ({os.path.getsize(path_before)//1024} KB)")
print(f"  before: {len(cam_pts_before)} trail pts, {drawn_b} drawn, {filtered_b} filtered")
if stray_i:
    _, _, _, sx1, sy1 = cam_pts_before[stray_i-1]
    _, _, _, sx2, sy2 = cam_pts_before[stray_i]
    print(f"  stray segment: ({sx1},{sy1}) -> ({sx2},{sy2})  "
          f"disp={np.sqrt((sx2-sx1)**2+(sy2-sy1)**2):.1f} px")

# ── AFTER — recording filter + outlier rejection ──────────────────────────────
trail_after  = build_trail(VIEW_FRAME, use_record_filter=True)
cam_pts_after = compute_cam_pts(trail_after, VIEW_FRAME)
cam_pts_after = apply_outlier_rejection(cam_pts_after)

frame_after = base_frame.copy()
drawn_a, filtered_a = draw_trail(frame_after, cam_pts_after, T2_COLOR)

for thick, col in ((3,(0,0,0)),(1,(255,255,255))):
    cv2.putText(frame_after, f"AFTER   Player 81 trail  frame {VIEW_FRAME}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, thick)
    cv2.putText(frame_after,
                f"trail pts={len(cam_pts_after)}  drawn={drawn_a}  filtered={filtered_a}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, thick)
    cv2.putText(frame_after,
                f"Record filter {RECORD_JUMP_PX}px + outlier rejection {OUTLIER_NEIGHBOR_PX}px",
                (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, thick)

path_after = "output_videos/output6_dbg_p81_after.jpg"
cv2.imwrite(path_after, frame_after)
print(f"Saved: {path_after}  ({os.path.getsize(path_after)//1024} KB)")
print(f"  after : {len(cam_pts_after)} trail pts, {drawn_a} drawn, {filtered_a} filtered")

# Print which points were removed by recording filter
removed = set(fn for fn, _, _ in trail_before) - set(fn for fn, _, _ in trail_after)
print(f"  points removed by recording filter: {sorted(removed)}")
for fn in sorted(removed):
    cdx = float(cam_mv[fn][0]) if fn < len(cam_mv) else 0
    cdy = float(cam_mv[fn][1]) if fn < len(cam_mv) else 0
    print(f"    fn={fn}  cam=({cdx:.2f},{cdy:.2f})  cam_mag={np.sqrt(cdx**2+cdy**2):.2f}")
