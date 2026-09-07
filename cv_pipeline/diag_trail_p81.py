"""
Diagnostic: render_output6 trail glitch for player 81.

Simulates the exact trail accumulation from render_output6 and prints
every stored trail point (frame, position_adjusted, camera-compensated
draw position) at the frame where player 81's trail is longest / worst.
Also identifies which segment is the outlier and prints its camera data.
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
FPS         = 24
TRAIL_LEN   = FPS * 3   # 72 frames, matching render_output6
JUMP_PX     = 50        # current draw-time segment filter

# ── Load stubs and compute positions ─────────────────────────────────────────
print("Loading stubs...")
with open(TRACK_STUB, 'rb') as f: tracks = pickle.load(f)
with open(CAM_STUB,   'rb') as f: cam_mv = pickle.load(f)
n = len(tracks['players'])
print(f"  frames: {n}")

tracker = Tracker('models/best.pt')
tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
tracker.add_position_to_tracks(tracks)

cap = cv2.VideoCapture(RAW_VIDEO)
ret, frame0 = cap.read(); cap.release()
cam_est = CameraMovementEstimator(frame0)
cam_est.add_adjust_positions_to_tracks(tracks, cam_mv)

# ── Prefix sums (same as render_output6) ─────────────────────────────────────
prefix_x = np.zeros(n, dtype=np.float64)
prefix_y = np.zeros(n, dtype=np.float64)
for fn in range(1, n):
    dx, dy = float(cam_mv[fn][0]), float(cam_mv[fn][1])
    prefix_x[fn] = prefix_x[fn-1] + dx
    prefix_y[fn] = prefix_y[fn-1] + dy

# ── Simulate trail accumulation for player 81 ─────────────────────────────────
# Exactly mirrors render_output6: append (frame_num, x_adj, y_adj) if non-zero.
trail_p81 = deque(maxlen=TRAIL_LEN)
first_fn_81 = None
last_fn_81  = None

for fn in range(n):
    if TARGET_PID not in tracks['players'][fn]:
        continue
    info = tracks['players'][fn][TARGET_PID]
    adj  = info.get('position_adjusted')
    if adj is None:
        continue
    x, y = float(adj[0]), float(adj[1])
    if x == 0.0 and y == 0.0:
        continue
    trail_p81.append((fn, x, y))
    if first_fn_81 is None:
        first_fn_81 = fn
    last_fn_81 = fn

print(f"\nPlayer {TARGET_PID}: first frame={first_fn_81}  last frame={last_fn_81}  "
      f"total positions={len(trail_p81)}")

if not trail_p81:
    print(f"ERROR: no valid positions found for player {TARGET_PID}")
    sys.exit(1)

# ── Find the "current" frame that shows the worst draw-space outlier ──────────
# We want the frame where trail points show the biggest stray segment.
# Try last_fn_81 first (full trail), then search neighbourhood.
def worst_segment_disp(trail_list, cur_fn):
    """Return max draw-space segment displacement for this trail at cur_fn."""
    cam_pts = []
    for fn, x_adj, y_adj in trail_list:
        cum_dx = prefix_x[cur_fn] - prefix_x[fn]
        cum_dy = prefix_y[cur_fn] - prefix_y[fn]
        cam_pts.append((fn, x_adj, y_adj,
                        int(round(x_adj - cum_dx)),
                        int(round(y_adj - cum_dy))))
    worst = 0.0
    for i in range(1, len(cam_pts)):
        _, _, _, dx1, dy1 = cam_pts[i-1]
        _, _, _, dx2, dy2 = cam_pts[i]
        d = np.sqrt((dx2-dx1)**2 + (dy2-dy1)**2)
        if d > worst:
            worst = d
    return worst, cam_pts

# Search from first appearance up to last, re-simulating the trail
best_cur = last_fn_81
best_disp = 0.0

# Quick scan: rebuild trail up to each frame and check
trail_scan = deque(maxlen=TRAIL_LEN)
for fn in range(n):
    if TARGET_PID not in tracks['players'][fn]:
        continue
    info = tracks['players'][fn][TARGET_PID]
    adj  = info.get('position_adjusted')
    if adj is None:
        continue
    x, y = float(adj[0]), float(adj[1])
    if x == 0.0 and y == 0.0:
        continue
    trail_scan.append((fn, x, y))
    if len(trail_scan) < 3:
        continue
    disp, _ = worst_segment_disp(list(trail_scan), fn)
    if disp > best_disp:
        best_disp = disp
        best_cur  = fn
        best_trail = list(trail_scan)

print(f"  Worst draw-space segment: {best_disp:.1f} px  at current frame {best_cur}")

# ── Full trail report at worst frame ─────────────────────────────────────────
_, cam_pts = worst_segment_disp(best_trail, best_cur)

print(f"\n{'='*95}")
print(f"PLAYER {TARGET_PID}  TRAIL AT FRAME {best_cur}  ({len(cam_pts)} points)")
print(f"{'='*95}")
print(f"\n  {'i':>3}  {'fn':>5}  {'x_adj':>8}  {'y_adj':>8}  "
      f"{'draw_x':>7}  {'draw_y':>7}  {'seg_disp':>9}  "
      f"{'cam_dx':>7}  {'cam_dy':>7}  {'cam_mag':>8}")
print(f"  {'-'*93}")

draw_xs = np.array([p[3] for p in cam_pts])
draw_ys = np.array([p[4] for p in cam_pts])

for i, (fn, x_adj, y_adj, draw_x, draw_y) in enumerate(cam_pts):
    # Segment displacement to next point
    if i < len(cam_pts) - 1:
        _, _, _, nx, ny = cam_pts[i+1]
        seg = np.sqrt((nx-draw_x)**2 + (ny-draw_y)**2)
        seg_str = f"{seg:>9.1f}"
    else:
        seg_str = f"{'':>9}"

    cdx = float(cam_mv[fn][0]) if fn < len(cam_mv) else 0.0
    cdy = float(cam_mv[fn][1]) if fn < len(cam_mv) else 0.0
    cmag = np.sqrt(cdx**2 + cdy**2)

    flag = ""
    if i > 0:
        _, _, _, px_prev, py_prev = cam_pts[i-1]
        back_seg = np.sqrt((draw_x-px_prev)**2 + (draw_y-py_prev)**2)
        if back_seg > JUMP_PX:
            flag = " <<FILTERED>>"
        elif back_seg > 25:
            flag = " <medium>"

    print(f"  {i:>3}  {fn:>5}  {x_adj:>8.1f}  {y_adj:>8.1f}  "
          f"{draw_x:>7}  {draw_y:>7}  {seg_str}  "
          f"{cdx:>7.2f}  {cdy:>7.2f}  {cmag:>8.2f}{flag}")

# ── Identify outlier point ────────────────────────────────────────────────────
print(f"\n--- Outlier detection (point far from neighbor average) ---")
outliers = []
for i in range(1, len(cam_pts) - 1):
    _, _, _, px, py = cam_pts[i]
    _, _, _, px_prev, py_prev = cam_pts[i-1]
    _, _, _, px_next, py_next = cam_pts[i+1]
    avg_x = (px_prev + px_next) / 2
    avg_y = (py_prev + py_next) / 2
    dist_from_avg = np.sqrt((px - avg_x)**2 + (py - avg_y)**2)
    if dist_from_avg > 20:
        fn_out = cam_pts[i][0]
        cdx = float(cam_mv[fn_out][0]) if fn_out < len(cam_mv) else 0
        cdy = float(cam_mv[fn_out][1]) if fn_out < len(cam_mv) else 0
        print(f"  point {i:>3} at frame {fn_out:>5}: draw=({px},{py})  "
              f"neighbor_avg=({avg_x:.0f},{avg_y:.0f})  "
              f"dist_from_avg={dist_from_avg:.1f} px  "
              f"cam=({cdx:.2f},{cdy:.2f})")
        outliers.append((i, fn_out, px, py, dist_from_avg, cdx, cdy))

if not outliers:
    print("  No single-point outliers found (outlier may be in trail entry or exit).")
    # Check first and last points
    for i, (fn, x_adj, y_adj, draw_x, draw_y) in enumerate(cam_pts[:3] + cam_pts[-3:]):
        cdx = float(cam_mv[fn][0]) if fn < len(cam_mv) else 0
        cdy = float(cam_mv[fn][1]) if fn < len(cam_mv) else 0
        print(f"  i={i:>3}  fn={fn}  draw=({draw_x},{draw_y})  "
              f"x_adj={x_adj:.1f}  y_adj={y_adj:.1f}  cam=({cdx:.2f},{cdy:.2f})")

# ── Camera data for the outlier frame ────────────────────────────────────────
if outliers:
    i_out, fn_out, px, py, dist, cdx, cdy = outliers[0]
    print(f"\n--- Camera data for outlier frame {fn_out} ---")
    print(f"  camera_movement[{fn_out}]     = ({cdx:.4f}, {cdy:.4f})  "
          f"mag={np.sqrt(cdx**2+cdy**2):.2f}")
    print(f"  prefix_x[{fn_out}]            = {prefix_x[fn_out]:.4f}")
    print(f"  prefix_y[{fn_out}]            = {prefix_y[fn_out]:.4f}")
    print(f"  prefix_x[{best_cur}] (current) = {prefix_x[best_cur]:.4f}")
    print(f"  prefix_y[{best_cur}] (current) = {prefix_y[best_cur]:.4f}")
    cum_dx = prefix_x[best_cur] - prefix_x[fn_out]
    cum_dy = prefix_y[best_cur] - prefix_y[fn_out]
    print(f"  cumulative drift applied:    ({cum_dx:.4f}, {cum_dy:.4f})")
    fn_out = int(fn_out)
    x_adj_out = cam_pts[i_out][1]
    y_adj_out = cam_pts[i_out][2]
    print(f"  position_adjusted[{fn_out}]   = ({x_adj_out:.1f}, {y_adj_out:.1f})")
    print(f"  draw position (after comp):   ({px}, {py})")
    if i_out > 0:
        _, _, _, ppx, ppy = cam_pts[i_out-1]
        print(f"  prev draw position (i={i_out-1}):  ({ppx}, {ppy})  "
              f"dist={np.sqrt((px-ppx)**2+(py-ppy)**2):.1f} px")
    if i_out < len(cam_pts)-1:
        _, _, _, npx, npy = cam_pts[i_out+1]
        print(f"  next draw position (i={i_out+1}):  ({npx}, {npy})  "
              f"dist={np.sqrt((px-npx)**2+(py-npy)**2):.1f} px")

    # Diagnosis: filter a or b?
    print(f"\n--- DIAGNOSIS ---")
    if np.sqrt(cdx**2+cdy**2) > 5:
        print(f"  The outlier frame {fn_out} has cam_mag={np.sqrt(cdx**2+cdy**2):.2f}.")
        print(f"  position_adjusted is shifted by the camera spike BEFORE entering trail.")
        print(f"  The cumulative drift compensation ({cum_dx:.1f},{cum_dy:.1f}) does not")
        print(f"  fully cancel this because the spike is a single-frame anomaly.")
        print(f"  -> Type (a): the outlier entered the trail despite large camera movement.")
        print(f"     Fix: recording-time filter that checks position_adjusted jump > threshold.")
    else:
        print(f"  The outlier frame {fn_out} has cam_mag={np.sqrt(cdx**2+cdy**2):.2f} (small).")
        print(f"  The cumulative drift compensation is ({cum_dx:.1f},{cum_dy:.1f}).")
        x_adj_raw = cam_pts[i_out][1] + (prefix_x[fn_out] - prefix_x[best_cur])
        print(f"  The camera compensation itself may be wrong (cumulative error).")
        print(f"  -> Type (b): the cumulative camera offset for this frame is wrong.")

# ── Current draw-time jump filter check ──────────────────────────────────────
print(f"\n--- Current JUMP_PX={JUMP_PX} filter status ---")
filtered = []
drawn = []
for i in range(1, len(cam_pts)):
    _, _, _, dx1, dy1 = cam_pts[i-1]
    _, _, _, dx2, dy2 = cam_pts[i]
    disp = np.sqrt((dx2-dx1)**2 + (dy2-dy1)**2)
    fn_i = cam_pts[i][0]
    if disp > JUMP_PX:
        filtered.append((i, fn_i, disp))
    else:
        drawn.append((i, fn_i, disp))

print(f"  Segments drawn  (disp <= {JUMP_PX}px): {len(drawn)}")
print(f"  Segments filtered (disp >  {JUMP_PX}px): {len(filtered)}")
for i, fn_i, d in filtered:
    print(f"    segment {i-1}->{i}  fn={fn_i}  disp={d:.1f} px  <- FILTERED")

# ── Existing recording-time filter check ────────────────────────────────────
print(f"\n--- Recording-time filter ---")
print(f"  No recording-time displacement filter exists in render_output6.py.")
print(f"  All non-zero position_adjusted values enter the trail unconditionally.")
print(f"  The position_adjusted jump at the outlier frame entered the trail because")
print(f"  there is no threshold check at append time.")
