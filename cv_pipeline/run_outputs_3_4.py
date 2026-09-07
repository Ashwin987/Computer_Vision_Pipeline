"""
run_outputs_3_4.py
Re-renders output3.avi (Voronoi control) and output4.avi (pitch heatmap)
from existing stubs — no re-tracking, no re-calibration, no YOLO inference.

Pipeline:
  1. Load stubs  (tracks, camera movement, homography)
  2. Add foot positions + camera-movement adjustment to tracks
  3. Load video frames (needed for team jersey-colour extraction only)
  4. Assign teams and ball possession
  5. Detect transitions (render_output3 needs this)
  6. Load annotated frames from the saved output_videos/output_video.avi
  7. Call render_output3 then render_output4
"""

import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm

from utils import read_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from transition_detector import TransitionDetector

from render_output3 import render_output3
from render_output4 import render_output4

VIDEO_PATH    = 'Match_videos/121364_0.mp4'
TRACK_STUB    = 'stubs/track_stubs_121364.pkl'
CAM_STUB      = 'stubs/camera_movement_stub_121364.pkl'
CAL_STUB      = 'stubs/homography_stub_121364.pkl'
ANNOTATED_SRC = 'output_videos/output_video.avi'
FPS           = 24


def _load_annotated_frames(path, original_w):
    """Read frames from a saved .avi, crop to original_w."""
    if not os.path.exists(path):
        return None
    cap    = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        frames.append(frm[:, :original_w])
    cap.release()
    return frames


def main():
    # ── 1. Load stubs ─────────────────────────────────────────────────────────
    for stub in (TRACK_STUB, CAM_STUB, CAL_STUB):
        if not os.path.exists(stub):
            raise FileNotFoundError(f"Stub missing: {stub}  — run main.py first.")

    print("Loading stubs...")
    with open(TRACK_STUB, 'rb') as f:
        tracks = pickle.load(f)
    with open(CAM_STUB, 'rb') as f:
        camera_movement_per_frame = pickle.load(f)
    with open(CAL_STUB, 'rb') as f:
        homography_per_frame = pickle.load(f)
    print(f"  track frames: {len(tracks.get('players', []))}")

    # ── 2. Add foot positions and camera-movement adjustment ──────────────────
    # Tracker is loaded only to call add_position_to_tracks and
    # interpolate_ball_positions — no YOLO inference happens here.
    print("Adding positions to tracks...")
    tracker = Tracker('models/best.pt')
    tracker.add_position_to_tracks(tracks)
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # CameraMovementEstimator constructor needs a real frame for its feature
    # mask setup, but add_adjust_positions_to_tracks is pure arithmetic.
    print("Loading video frames (required for team colour extraction)...")
    video_frames = read_video(VIDEO_PATH)
    print(f"  {len(video_frames)} frames loaded ({video_frames[0].shape[1]}x{video_frames[0].shape[0]})")

    cam_est = CameraMovementEstimator(video_frames[0])
    cam_est.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # ── 3. Team assignment ────────────────────────────────────────────────────
    print("Assigning teams...")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames, tracks['players'])
    for frame_num, player_track in enumerate(
            tqdm(tracks['players'], desc="  teams")):
        for pid, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track['bbox'], pid)
            tracks['players'][frame_num][pid]['team']       = team
            tracks['players'][frame_num][pid]['team_color'] = \
                team_assigner.team_colors[team]

    # ── 4. Ball possession → team_ball_control ────────────────────────────────
    print("Assigning ball possession...")
    player_assigner   = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(
            tqdm(tracks['players'], desc="  ball")):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned  = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned != -1:
            tracks['players'][frame_num][assigned]['has_ball'] = True
            team_ball_control.append(
                tracks['players'][frame_num][assigned]['team'])
        else:
            team_ball_control.append(
                team_ball_control[-1] if team_ball_control else 0)
    team_ball_control = np.array(team_ball_control)

    # ── 5. Transitions (needed by render_output3) ─────────────────────────────
    print("Detecting transitions...")
    td          = TransitionDetector(fps=FPS)
    transitions = td.detect_transitions(team_ball_control, tracks)
    td.print_summary()

    # ── 6. Annotated frames for render_output4 ────────────────────────────────
    # Reuse the already-rendered output_video.avi as the base overlay frame.
    # If it doesn't exist yet, fall back to raw video frames.
    original_w = video_frames[0].shape[1]
    annotated_frames = _load_annotated_frames(ANNOTATED_SRC, original_w)
    if annotated_frames:
        print(f"Loaded {len(annotated_frames)} annotated frames from {ANNOTATED_SRC}")
    else:
        print(f"WARNING: {ANNOTATED_SRC} not found — using raw video frames as base.")
        annotated_frames = [f.copy() for f in video_frames]

    view_transformer = ViewTransformer()

    # ── 7. Render ─────────────────────────────────────────────────────────────
    print("\n=== render_output3 (Voronoi territorial control) ===")
    render_output3(video_frames, tracks, team_ball_control, transitions, fps=FPS,
                   homography_per_frame=homography_per_frame)

    print("\n=== render_output4 (pitch control heatmap) ===")
    render_output4(video_frames, tracks, team_ball_control, view_transformer,
                   fps=FPS, annotated_frames=annotated_frames,
                   homography_per_frame=homography_per_frame)

    print("\nDone.  Check output_videos/output3.avi and output_videos/output4.avi")


if __name__ == '__main__':
    main()
