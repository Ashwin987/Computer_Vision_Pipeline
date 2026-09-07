"""
Diagnostic: trail glitch root-cause analysis for render_output6.

Finds the player + 30-frame window with the worst position jumps,
then prints raw position_adjusted, camera_movement, and frame-by-frame
displacement so we can tell whether glitches come from tracker jitter
or camera compensation errors.
"""
import sys, os, pickle
import numpy as np
sys.path.insert(0, r'c:\UCLA\yolo model 2')

from trackers import Tracker
from camera_movement_estimator import CameraMovementEstimator
import cv2

TRACK_STUB = 'stubs/track_stubs_121364.pkl'
CAM_STUB   = 'stubs/camera_movement_stub_121364.pkl'
RAW_VIDEO  = 'Match_videos/121364_0.mp4'

# ── Load stubs ────────────────────────────────────────────────────────────────
print("Loading stubs...")
with open(TRACK_STUB, 'rb') as f: tracks = pickle.load(f)
with open(CAM_STUB,   'rb') as f: cam_mv = pickle.load(f)
total_frames = len(tracks['players'])
print(f"  track frames : {total_frames}")
print(f"  cam_mv len   : {len(cam_mv)}")

# ── Add positions and camera-adjusted positions ───────────────────────────────
print("Computing positions...")
tracker = Tracker('models/best.pt')
tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
tracker.add_position_to_tracks(tracks)

# Load one frame for CameraMovementEstimator constructor
cap = cv2.VideoCapture(RAW_VIDEO)
ret, first_frame = cap.read()
cap.release()
if not ret:
    print("ERROR: cannot read first frame from raw video")
    sys.exit(1)

cam_est = CameraMovementEstimator(first_frame)
cam_est.add_adjust_positions_to_tracks(tracks, cam_mv)

# ── Build per-player position_adjusted time series ───────────────────────────
print("Building position histories...")
# player_history[pid][fn] = (x_adj, y_adj) or None
player_history = {}  # pid -> list of (fn, x_adj, y_adj)

for fn in range(total_frames):
    for pid, info in tracks['players'][fn].items():
        adj = info.get('position_adjusted')
        if adj is None:
            continue
        x, y = float(adj[0]), float(adj[1])
        if x == 0.0 and y == 0.0:
            continue
        if pid not in player_history:
            player_history[pid] = []
        player_history[pid].append((fn, x, y))

print(f"  tracked players: {len(player_history)}")

# ── Compute per-player frame-to-frame jumps ───────────────────────────────────
# Find the single worst jump (largest consecutive-frame displacement) per player,
# then pick the top-N players and show their 30-frame window around the worst jump.
print("Computing frame-to-frame jumps...")

worst_per_player = {}  # pid -> (max_jump_px, worst_fn, window_start, window_end)

for pid, hist in player_history.items():
    if len(hist) < 4:
        continue
    # hist is sorted by fn (iteration order of frames)
    hist_arr = np.array([(fn, x, y) for fn, x, y in hist])
    fns  = hist_arr[:, 0].astype(int)
    xs   = hist_arr[:, 1]
    ys   = hist_arr[:, 2]

    # Only consider consecutive frames (fn+1 -> fn)
    worst_jump = 0.0
    worst_idx  = 0
    for i in range(1, len(fns)):
        if fns[i] != fns[i-1] + 1:
            continue  # skip non-consecutive frames
        dx = xs[i] - xs[i-1]
        dy = ys[i] - ys[i-1]
        jump = np.sqrt(dx*dx + dy*dy)
        if jump > worst_jump:
            worst_jump = jump
            worst_idx  = i

    if worst_jump < 10:
        continue  # no interesting jumps

    worst_fn = int(fns[worst_idx])
    # 30-frame window centered on worst jump
    w_start = max(0, worst_idx - 14)
    w_end   = min(len(hist)-1, worst_idx + 15)
    worst_per_player[pid] = (worst_jump, worst_fn, hist, w_start, w_end)

# Sort by worst jump descending
ranked = sorted(worst_per_player.items(), key=lambda kv: kv[1][0], reverse=True)
print(f"  players with consecutive-frame jump > 10 px: {len(ranked)}")

# ── Camera movement overview ──────────────────────────────────────────────────
cam_arr  = np.array([[float(v[0]), float(v[1])] for v in cam_mv], dtype=np.float64)
cam_mag  = np.sqrt(cam_arr[:,0]**2 + cam_arr[:,1]**2)

nonzero_frames = np.where(cam_mag > 0.5)[0]
print(f"\n--- Camera movement overview ---")
print(f"  frames with |cam_mv| > 0.5 px : {len(nonzero_frames)} / {len(cam_mv)}")
if len(nonzero_frames) > 0:
    print(f"  max camera movement            : {cam_mag.max():.2f} px at frame {cam_mag.argmax()}")
    print(f"  non-zero frames sample         : {nonzero_frames[:20].tolist()}")
else:
    print("  camera movement is (0,0) for all frames")

# Prefix sums for compensation check
prefix_x = np.zeros(total_frames, dtype=np.float64)
prefix_y = np.zeros(total_frames, dtype=np.float64)
for fn in range(1, min(total_frames, len(cam_mv))):
    dx, dy = float(cam_mv[fn][0]), float(cam_mv[fn][1])
    prefix_x[fn] = prefix_x[fn-1] + dx
    prefix_y[fn] = prefix_y[fn-1] + dy

# ── Detailed report for top-3 glitching players ───────────────────────────────
TOP_N = 3
print(f"\n{'='*75}")
print(f"TOP {TOP_N} GLITCHING PLAYERS (worst consecutive-frame position jump)")
print(f"{'='*75}")

for rank, (pid, (worst_jump, worst_fn, hist, w_start, w_end)) in enumerate(ranked[:TOP_N]):
    print(f"\n--- RANK {rank+1}: PID {pid}  worst jump = {worst_jump:.1f} px at frame {worst_fn} ---")

    window = hist[w_start : w_end + 1]
    print(f"\n  30-frame window: history index {w_start}-{w_end}  "
          f"(frame {window[0][0]} to {window[-1][0]})")
    print()
    print(f"  {'fn':>5}  {'x_adj':>8}  {'y_adj':>8}  {'jump_px':>9}  "
          f"{'cam_dx':>7}  {'cam_dy':>7}  {'cam_mag':>8}  analysis")
    print(f"  {'-'*77}")

    prev_x, prev_y = None, None
    for fn, x, y in window:
        fn = int(fn)
        # Frame-to-frame jump in position_adjusted
        if prev_x is not None:
            jump_px = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
        else:
            jump_px = 0.0
        prev_x, prev_y = x, y

        # Camera movement at this frame
        if fn < len(cam_mv):
            cdx, cdy = float(cam_mv[fn][0]), float(cam_mv[fn][1])
            cmag = np.sqrt(cdx**2 + cdy**2)
        else:
            cdx, cdy, cmag = 0.0, 0.0, 0.0

        # Classify this frame
        analysis = ""
        if jump_px > 50:
            analysis += " <<BIG_JUMP>>"
            if cmag > 2.0:
                analysis += "+CAM_SPIKE"
            else:
                analysis += "+CAM_STEADY"
        elif jump_px > 20:
            analysis += " <medium>"
        if cmag > 2.0 and jump_px < 5:
            analysis += " cam_spike_no_jump"

        print(f"  {fn:>5}  {x:>8.1f}  {y:>8.1f}  {jump_px:>9.1f}  "
              f"{cdx:>7.2f}  {cdy:>7.2f}  {cmag:>8.2f}  {analysis}")

    # Summary for this player
    window_jumps = []
    pw_x, pw_y = None, None
    for fn, x, y in window:
        if pw_x is not None:
            window_jumps.append(np.sqrt((x-pw_x)**2 + (y-pw_y)**2))
        pw_x, pw_y = x, y

    if window_jumps:
        big_jumps = [(i+1, window[i+1][0], j) for i, j in enumerate(window_jumps) if j > 20]
        cam_spikes = [(fn, np.sqrt(cam_mv[fn][0]**2+cam_mv[fn][1]**2))
                      for fn in [int(window[i+1][0]) for i, j in enumerate(window_jumps) if j > 20]
                      if fn < len(cam_mv)]
        print(f"\n  Frames with jump > 20 px : {len(big_jumps)}")
        for (idx, fn, j) in big_jumps:
            fn = int(fn)
            cmag_here = np.sqrt(cam_mv[fn][0]**2+cam_mv[fn][1]**2) if fn < len(cam_mv) else 0
            verdict = "CAMERA SPIKE" if cmag_here > 2.0 else "tracker jitter (cam steady)"
            print(f"    frame {fn:>4}  jump={j:>7.1f} px  cam_mag={cmag_here:.2f}  -> {verdict}")

# ── Camera compensation contribution ─────────────────────────────────────────
print(f"\n{'='*75}")
print(f"CAMERA COMPENSATION ANALYSIS")
print(f"{'='*75}")

# For the top player, show what prefix_x compensation adds/removes at draw time
pid, (worst_jump, worst_fn, hist, w_start, w_end) = ranked[0]
print(f"\nFor PID {pid} (worst glitcher), cumulative camera displacement")
print(f"during their glitch window (how much compensation is applied):")
window = hist[w_start : w_end + 1]
ref_fn = int(window[0][0])
print(f"\n  {'fn':>5}  {'prefix_x':>10}  {'prefix_y':>10}  "
      f"{'comp_from_ref':>15}  {'cam_mag':>8}")
print(f"  {'-'*60}")
for fn, x, y in window:
    fn = int(fn)
    px = prefix_x[fn] if fn < total_frames else 0
    py = prefix_y[fn] if fn < total_frames else 0
    comp_x = px - prefix_x[ref_fn] if ref_fn < total_frames else px
    comp_y = py - prefix_y[ref_fn] if ref_fn < total_frames else py
    cmag = np.sqrt(cam_mv[fn][0]**2+cam_mv[fn][1]**2) if fn < len(cam_mv) else 0
    print(f"  {fn:>5}  {px:>10.2f}  {py:>10.2f}  "
          f"  ({comp_x:>+6.2f},{comp_y:>+6.2f})  {cmag:>8.2f}")

# ── Overall jump statistics ───────────────────────────────────────────────────
print(f"\n{'='*75}")
print(f"GLOBAL JUMP STATISTICS (all players, all consecutive frames)")
print(f"{'='*75}")

all_jumps = []
all_cam_at_jump = []
for pid, hist in player_history.items():
    for i in range(1, len(hist)):
        fn_prev, xp, yp = hist[i-1]
        fn_cur,  xc, yc = hist[i]
        if int(fn_cur) != int(fn_prev) + 1:
            continue
        j = np.sqrt((xc-xp)**2 + (yc-yp)**2)
        fn_cur = int(fn_cur)
        cmag = np.sqrt(cam_mv[fn_cur][0]**2+cam_mv[fn_cur][1]**2) if fn_cur < len(cam_mv) else 0
        all_jumps.append(j)
        all_cam_at_jump.append(cmag)

all_jumps       = np.array(all_jumps)
all_cam_at_jump = np.array(all_cam_at_jump)

print(f"\n  Total consecutive-frame pairs: {len(all_jumps)}")
for thresh in [20, 50, 100, 200]:
    mask = all_jumps > thresh
    if mask.sum() == 0:
        print(f"  Jumps > {thresh:>3} px : 0")
        continue
    corr = all_cam_at_jump[mask]
    print(f"  Jumps > {thresh:>3} px : {mask.sum():>5}  "
          f"of those, cam_mag > 2.0: {(corr>2.0).sum():>4}  "
          f"({100*(corr>2.0).mean():.0f}%)  "
          f"median_cam={np.median(corr):.2f}  max_cam={corr.max():.2f}")

print(f"\n  jump percentiles: "
      f"p50={np.percentile(all_jumps,50):.1f}  "
      f"p90={np.percentile(all_jumps,90):.1f}  "
      f"p95={np.percentile(all_jumps,95):.1f}  "
      f"p99={np.percentile(all_jumps,99):.1f}  "
      f"max={all_jumps.max():.1f}")

# ── Conclusion ────────────────────────────────────────────────────────────────
print(f"\n{'='*75}")
print("PRELIMINARY CONCLUSION")
print(f"{'='*75}")

big_mask = all_jumps > 50
if big_mask.sum() == 0:
    print("  No jumps > 50 px found. Glitch may be a rendering issue, not position data.")
else:
    cam_corr = (all_cam_at_jump[big_mask] > 2.0).mean()
    if cam_corr > 0.6:
        print(f"  {cam_corr*100:.0f}% of big jumps (>50 px) co-occur with camera spikes.")
        print("  Likely cause: (b) camera compensation error — camera_movement spikes")
        print("  cause incorrect position_adjusted values that trail drawing amplifies.")
    elif cam_corr < 0.2:
        print(f"  Only {cam_corr*100:.0f}% of big jumps (>50 px) co-occur with camera spikes.")
        print("  Likely cause: (a) raw tracker jitter — ByteTrack assigns same PID to")
        print("  different detections, causing position_adjusted to jump discontinuously.")
    else:
        print(f"  Mixed: {cam_corr*100:.0f}% of big jumps co-occur with camera spikes.")
        print("  Both tracker jitter AND camera compensation errors likely contribute.")
