"""
Diagnostic: what does render_output3 actually feed into Voronoi at frame 50?
Loads the video selectively to run team assignment, then shows exact Voronoi inputs/outputs.
Does NOT change any existing code.
"""
import pickle, sys, os, time
import numpy as np
import cv2
from scipy.spatial import Voronoi

sys.path.insert(0, r'c:\UCLA\yolo model 2')
from utils import get_center_of_bbox, get_foot_position
from team_assigner import TeamAssigner

TRACK_STUB = 'stubs/track_stubs_121364.pkl'
CAM_STUB   = 'stubs/camera_movement_stub_121364.pkl'
SRC_VIDEO  = 'output_videos/output_video.avi'
RAW_VIDEO  = 'Match_videos/121364_0.mp4'
FRAME      = 50

# ── 1. Load stubs ────────────────────────────────────────────────────────────
print("Loading stubs...")
with open(TRACK_STUB, 'rb') as f:
    tracks = pickle.load(f)
with open(CAM_STUB, 'rb') as f:
    camera_movement_per_frame = pickle.load(f)
print(f"  player frames in stub: {len(tracks['players'])}")

# ── 2. Add positions ─────────────────────────────────────────────────────────
for obj, obj_tracks in tracks.items():
    for frame_num, track in enumerate(obj_tracks):
        for tid, tinfo in track.items():
            bbox = tinfo['bbox']
            pos  = get_center_of_bbox(bbox) if obj == 'ball' else get_foot_position(bbox)
            tracks[obj][frame_num][tid]['position'] = pos

for obj, obj_tracks in tracks.items():
    for frame_num, track in enumerate(obj_tracks):
        cam = camera_movement_per_frame[frame_num]
        for tid, tinfo in track.items():
            p = tinfo['position']
            tracks[obj][frame_num][tid]['position_adjusted'] = (
                p[0] - cam[0], p[1] - cam[1])

# ── 3. Load only the frames needed for team assignment ───────────────────────
# CLUSTER_FRAMES = [0,10,20,30,40], plus frame 50 for classification.
# Load frames 0-50 from the raw video (or output_video.avi as fallback).
NEED_FRAMES = set([0, 10, 20, 30, 40, FRAME])
print(f"\nLoading frames {sorted(NEED_FRAMES)} from video...")
video_src = RAW_VIDEO if os.path.exists(RAW_VIDEO) else SRC_VIDEO
cap = cv2.VideoCapture(video_src)
video_frames_sparse = {}
for fn in sorted(NEED_FRAMES):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frm = cap.read()
    if ret:
        video_frames_sparse[fn] = frm
cap.release()
print(f"  Loaded {len(video_frames_sparse)} frames from {video_src}")

# Wrap sparse dict as a list-like object (index → frame)
class SparseFrameList:
    def __init__(self, d, total):
        self.d = d; self.total = total
    def __len__(self):
        return self.total
    def __getitem__(self, i):
        return self.d[i]   # KeyError if not loaded

total_frames = len(tracks['players'])
vf = SparseFrameList(video_frames_sparse, total_frames)

# ── 4. Get source video dimensions ───────────────────────────────────────────
cap2 = cv2.VideoCapture(SRC_VIDEO)
ret2, f0 = cap2.read()
cap2.release()
if ret2:
    orig_h, orig_w = f0.shape[:2]
else:
    orig_w, orig_h = 1920, 1080
w_f, h_f = float(orig_w), float(orig_h)
print(f"  output_video.avi dimensions: {orig_w}x{orig_h}")

# ── 5. Run team assignment ────────────────────────────────────────────────────
print("\nRunning team assignment...")
ta = TeamAssigner()
# build sparse video_frames list for assign_team_color
# (only needs frames in CLUSTER_FRAMES, so SparseFrameList works)
ta.assign_team_color(vf, tracks['players'])

# Assign teams for ALL frames up to FRAME (to build the cache correctly)
for fn in range(min(FRAME + 1, len(tracks['players']))):
    if fn not in video_frames_sparse:
        continue
    for pid, info in tracks['players'][fn].items():
        team = ta.get_player_team(video_frames_sparse[fn], info['bbox'], pid)
        tracks['players'][fn][pid]['team'] = team

# ── 6. QUESTION 1: Players at frame 50 with team and position_adjusted ───────
print(f"\n{'='*70}")
print(f"Q1: FRAME {FRAME} — all players, team, position_adjusted")
print(f"{'PID':>5}  {'team':>5}  {'adj_x':>10}  {'adj_y':>10}  "
      f"{'x_ok':>6}  {'y_ok':>6}  {'passes_filter':>14}")
print("-" * 70)

player_data      = tracks['players'][FRAME]
player_pos_filt  = {}   # team-filtered positions for Voronoi

for pid in sorted(player_data.keys()):
    info = player_data[pid]
    team = info.get('team', 0)
    adj  = info.get('position_adjusted')
    if adj is not None:
        x, y = float(adj[0]), float(adj[1])
        x_ok   = 0 <= x <= orig_w
        y_ok   = 0 <= y <= orig_h
        passes = team in (1, 2) and (x != 0.0 or y != 0.0)
        print(f"{pid:>5}  {str(team):>5}  {x:>10.1f}  {y:>10.1f}  "
              f"{str(x_ok):>6}  {str(y_ok):>6}  {str(passes):>14}")
        if passes:
            player_pos_filt[pid] = (x, y)
    else:
        print(f"{pid:>5}  {str(team):>5}  {'None':>10}  {'None':>10}  "
              f"{'?':>6}  {'?':>6}  {'?':>14}")

print(f"\n  Total players in frame: {len(player_data)}")
print(f"  Passing team filter (team in 1,2): {len(player_pos_filt)}")

# ── Q2: Confirm coordinate range ─────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Q2: Are position_adjusted coords valid for a {orig_w}x{orig_h} frame?")
if player_pos_filt:
    xs = [v[0] for v in player_pos_filt.values()]
    ys = [v[1] for v in player_pos_filt.values()]
    print(f"  adj_x range: {min(xs):.1f} – {max(xs):.1f}   (frame width  = {orig_w})")
    print(f"  adj_y range: {min(ys):.1f} – {max(ys):.1f}   (frame height = {orig_h})")
    all_in = all(0 <= x <= orig_w and 0 <= y <= orig_h
                 for x, y in player_pos_filt.values())
    print(f"  All within frame bounds: {all_in}")
else:
    print("  No players passed the team filter — cannot check.")

# Also show full range including unfiltered players
all_adj = [(float(info['position_adjusted'][0]), float(info['position_adjusted'][1]))
           for info in player_data.values()
           if info.get('position_adjusted') is not None]
if all_adj:
    xs_all = [v[0] for v in all_adj]
    ys_all = [v[1] for v in all_adj]
    print(f"\n  All {len(all_adj)} players (unfiltered):")
    print(f"  adj_x range: {min(xs_all):.1f} – {max(xs_all):.1f}")
    print(f"  adj_y range: {min(ys_all):.1f} – {max(ys_all):.1f}")

# ── Q3: Guard points ─────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Q3: Guard points  (w_f={w_f}, h_f={h_f})")
guard_names = ['N','NE','E','SE','S','SW','W','NW']
guard = [
    ( w_f*0.5, -2*h_f),
    ( 3*w_f,   -2*h_f),
    ( 3*w_f,    h_f*0.5),
    ( 3*w_f,    3*h_f),
    ( w_f*0.5,  3*h_f),
    (-2*w_f,    3*h_f),
    (-2*w_f,    h_f*0.5),
    (-2*w_f,   -2*h_f),
]
for name, (gx, gy) in zip(guard_names, guard):
    print(f"  {name:3s}: ({gx:>10.1f}, {gy:>10.1f})")

# ── Q4 & Q5: Voronoi region count and mapping ────────────────────────────────
print(f"\n{'='*70}")
print(f"Q4/Q5: Voronoi ({len(player_pos_filt)} filtered players + {len(guard)} guard)")

if len(player_pos_filt) < 2:
    print(f"  ERROR: only {len(player_pos_filt)} player(s) pass the team filter.")
    print(f"  Voronoi requires at least 2 points — SKIPPED.")
    print(f"\n  >>> ROOT CAUSE: team assignment giving too few valid players at frame {FRAME}.")
else:
    pids_list = list(player_pos_filt.keys())
    coords    = [player_pos_filt[p] for p in pids_list]
    pts_arr   = np.array(coords + guard, dtype=np.float64)

    vor = Voronoi(pts_arr)

    print(f"  Input total:          {len(pts_arr)} points")
    print(f"  vor.point_region len: {len(vor.point_region)}")
    print(f"  vor.regions count:    {len(vor.regions)}")
    print(f"  vor.vertices shape:   {vor.vertices.shape}")

    print(f"\n  Q5: Region-to-player mapping (i -> pid -> region):")
    print(f"  {'i':>4}  {'PID':>5}  {'pos':<22}  {'ridx':>5}  "
          f"{'nvert':>6}  {'has-1':>6}  {'valid':>6}")
    print(f"  {'-'*62}")

    valid_count = 0
    for i, pid in enumerate(pids_list):
        ridx   = vor.point_region[i]
        region = vor.regions[ridx]
        has_m1 = -1 in region if region else True
        ok     = bool(region) and not has_m1 and len(region) >= 3
        if ok:
            valid_count += 1
        pos_str = f"({pts_arr[i,0]:.0f}, {pts_arr[i,1]:.0f})"
        print(f"  {i:>4}  {pid:>5}  {pos_str:<22}  {ridx:>5}  "
              f"{len(region):>6}  {str(has_m1):>6}  {str(ok):>6}")

    print(f"\n  Players with valid bounded regions: {valid_count} / {len(pids_list)}")

    # Check guard points don't share region with any player
    player_ridxs = {vor.point_region[i] for i in range(len(pids_list))}
    collisions = []
    for j in range(len(pids_list), len(pts_arr)):
        gidx = vor.point_region[j]
        if gidx in player_ridxs:
            collisions.append(j - len(pids_list))
    if collisions:
        print(f"\n  WARNING: guard[{collisions}] share a region index with a player!")
    else:
        print(f"  Guard-to-player region isolation: OK (no shared regions)")

    # Show where Voronoi vertices fall — should be near player positions
    vx = vor.vertices[:, 0]
    vy = vor.vertices[:, 1]
    print(f"\n  Voronoi vertex x range: {vx.min():.1f} – {vx.max():.1f}")
    print(f"  Voronoi vertex y range: {vy.min():.1f} – {vy.max():.1f}")
    in_frame_v = np.sum((vx >= 0) & (vx <= orig_w) & (vy >= 0) & (vy <= orig_h))
    print(f"  Vertices inside frame: {in_frame_v} / {len(vor.vertices)}")
