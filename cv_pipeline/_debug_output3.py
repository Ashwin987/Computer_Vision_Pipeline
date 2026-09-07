"""
Fast debug render: output3 for first 80 frames only.
Saves debug JPEGs at frames 50, 55, 60, 65, 70.
Does NOT re-track or re-calibrate — stubs only.
"""
import os, sys, pickle, cv2, numpy as np
from tqdm import tqdm
sys.path.insert(0, r'c:\UCLA\yolo model 2')

from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from render_output3 import render_output3

VIDEO_PATH = 'Match_videos/121364_0.mp4'
TRACK_STUB = 'stubs/track_stubs_121364.pkl'
CAM_STUB   = 'stubs/camera_movement_stub_121364.pkl'
CAL_STUB   = 'stubs/homography_stub_121364.pkl'
MAX_FRAMES = 80
FPS        = 24

# 1. Stubs
print("Loading stubs...")
with open(TRACK_STUB, 'rb') as f: tracks = pickle.load(f)
with open(CAM_STUB,   'rb') as f: cam_mv = pickle.load(f)
with open(CAL_STUB,   'rb') as f: hom    = pickle.load(f)
print(f"  track frames : {len(tracks.get('players', []))}")

# 2. Positions (interpolate ball first so all frames get 'position')
print("Adding positions to tracks...")
tracker = Tracker('models/best.pt')
tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
tracker.add_position_to_tracks(tracks)

# 3. Load only first MAX_FRAMES raw video frames (needed for team colours)
print(f"Loading first {MAX_FRAMES} raw video frames...")
cap = cv2.VideoCapture(VIDEO_PATH)
video_frames = []
while len(video_frames) < MAX_FRAMES:
    ret, frm = cap.read()
    if not ret:
        break
    video_frames.append(frm)
cap.release()
print(f"  loaded {len(video_frames)} frames "
      f"({video_frames[0].shape[1]}x{video_frames[0].shape[0]})")

# 4. Camera movement adjustment (full tracks, safe — we truncate later)
print("Adjusting positions for camera movement...")
cam_est = CameraMovementEstimator(video_frames[0])
cam_est.add_adjust_positions_to_tracks(tracks, cam_mv)

# 5. Team assignment — cluster frames are [0,10,20,30,40], all within MAX_FRAMES
print("Assigning teams...")
team_assigner = TeamAssigner()

class PartialFL:
    """List proxy that returns the last loaded frame for out-of-range indices."""
    def __init__(self, frames, total):
        self._f = frames; self._n = total
    def __len__(self):  return self._n
    def __getitem__(self, i):
        return self._f[min(i, len(self._f) - 1)]

pfl = PartialFL(video_frames, len(tracks['players']))
team_assigner.assign_team_color(pfl, tracks['players'])

for fn in tqdm(range(min(MAX_FRAMES, len(tracks['players']))), desc="  teams"):
    frm = video_frames[min(fn, len(video_frames) - 1)]
    for pid, track in tracks['players'][fn].items():
        team = team_assigner.get_player_team(frm, track['bbox'], pid)
        tracks['players'][fn][pid]['team']       = team
        tracks['players'][fn][pid]['team_color'] = team_assigner.team_colors[team]

# 6. Ball possession
print("Ball possession...")
player_assigner   = PlayerBallAssigner()
team_ball_control = []
for fn in range(min(MAX_FRAMES, len(tracks['players']))):
    ball_info = tracks['ball'][fn].get(1, {})
    ball_bbox = ball_info.get('bbox', [0, 0, 1, 1])
    assigned  = player_assigner.assign_ball_to_player(tracks['players'][fn], ball_bbox)
    if assigned != -1:
        tracks['players'][fn][assigned]['has_ball'] = True
        team_ball_control.append(tracks['players'][fn][assigned]['team'])
    else:
        team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
team_ball_control = np.array(team_ball_control)

# 7. Truncate tracks to MAX_FRAMES
print(f"Truncating to first {MAX_FRAMES} frames...")
tracks_short = {k: v[:MAX_FRAMES] for k, v in tracks.items()}

# 8. Render (only MAX_FRAMES frames are processed; debug JPEGs saved inline)
print(f"\n=== render_output3 (first {MAX_FRAMES} frames) ===")
render_output3(video_frames, tracks_short, team_ball_control, [], fps=FPS,
               homography_per_frame=hom)

# 9. Report debug frames
print("\n--- debug images ---")
for fn in [50, 55, 60, 65, 70]:
    path = f"output_videos/output3_dbg_f{fn}.jpg"
    if os.path.exists(path):
        sz = os.path.getsize(path) / 1024
        print(f"  OK  {path}  ({sz:.0f} KB)")
    else:
        print(f"  MISSING  {path}")
