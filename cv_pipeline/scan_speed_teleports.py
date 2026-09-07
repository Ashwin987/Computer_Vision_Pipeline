"""
Diagnostic-only script (not part of the pipeline). Replicates the position
pipeline main.py uses (position -> camera-compensated -> view-transformed)
and then, instead of SpeedAndDistance_Estimator's display-facing logic
(which silently holds the last good speed on an implausible jump), logs
every raw frame-to-frame jump that exceeds MAX_JUMP_M / MAX_SPEED_KMH for
every track -- a "teleport" is exactly the kind of physically-impossible
displacement a silent ID swap would produce.

Two modes:
  --before : loads the TRACK_STUB pickle directly (no split applied)
  --after  : loads tracks via Tracker.get_object_tracks() (split applied
             automatically per KNOWN_ID_SPLITS)
"""

import argparse
import pickle

import cv2
import numpy as np

from trackers import Tracker
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from utils import read_video, measure_distance

VIDEO_PATH = 'Match_videos/121364_0.mp4'
CAL_STUB   = 'stubs/homography_stub_121364.pkl'
TRACK_STUB = 'stubs/track_stubs_121364_ball_fallback.pkl'
CAM_STUB   = 'stubs/camera_movement_stub_121364.pkl'

MAX_JUMP_M    = 5.0
MAX_SPEED_KMH = 36.0
PITCH_X_MAX, PITCH_Y_MAX = 105.0, 68.0


def _in_bounds(pos):
    try:
        x, y = float(pos[0]), float(pos[1])
        return 0.0 <= x <= PITCH_X_MAX and 0.0 <= y <= PITCH_Y_MAX
    except (TypeError, IndexError, ValueError):
        return False


def build_tracks(apply_split):
    video_frames = read_video(VIDEO_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    view_transformer = ViewTransformer()
    with open(CAL_STUB, 'rb') as f:
        homography_per_frame = pickle.load(f)
    fb_H = view_transformer.persepctive_trasnformer.astype(np.float32)
    fb_H_inv = np.linalg.inv(fb_H).astype(np.float32)
    homography_per_frame = {
        fn: h if h is not None else (fb_H, fb_H_inv)
        for fn, h in homography_per_frame.items()
    }

    tracker = Tracker('models/best.pt')
    if apply_split:
        tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=TRACK_STUB,
                                           video_path=VIDEO_PATH)
    else:
        with open(TRACK_STUB, 'rb') as f:
            tracks = pickle.load(f)
    tracker.add_position_to_tracks(tracks)

    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames, read_from_stub=True, stub_path=CAM_STUB)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    view_transformer.add_transformed_position_to_tracks(tracks, homography_per_frame)

    return tracks, fps, (tracker.applied_splits if apply_split else {})


def scan(tracks, fps, target_pid=None):
    """Returns list of dicts: one per implausible frame-to-frame jump."""
    events = []
    player_tracks = tracks['players']
    n = len(player_tracks)
    for frame_num in range(1, n):
        prev_frame = player_tracks[frame_num - 1]
        curr_frame = player_tracks[frame_num]
        for track_id, info in curr_frame.items():
            if target_pid is not None and track_id != target_pid:
                continue
            if track_id not in prev_frame:
                continue
            pos_prev = prev_frame[track_id].get('position_transformed')
            pos_curr = info.get('position_transformed')
            if pos_prev is None or pos_curr is None:
                continue
            if not (_in_bounds(pos_prev) and _in_bounds(pos_curr)):
                continue
            dist = measure_distance(pos_prev, pos_curr)
            if dist > MAX_JUMP_M:
                events.append({'pid': track_id, 'frame': frame_num, 'dist_m': dist,
                                'kind': 'TELEPORT', 'pos_prev': pos_prev, 'pos_curr': pos_curr})
                continue
            inst_kmh = dist * fps * 3.6
            if inst_kmh > MAX_SPEED_KMH:
                events.append({'pid': track_id, 'frame': frame_num, 'dist_m': dist,
                                'kmh': inst_kmh, 'kind': 'CAP',
                                'pos_prev': pos_prev, 'pos_curr': pos_curr})
    return events


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['before', 'after'], required=True)
    ap.add_argument('--pid', type=int, default=None, help='restrict to one pid')
    args = ap.parse_args()

    tracks, fps, applied_splits = build_tracks(apply_split=(args.mode == 'after'))
    print(f"\nmode={args.mode}  fps={fps}  applied_splits={applied_splits}")

    events = scan(tracks, fps, target_pid=args.pid)
    print(f"\n=== {len(events)} implausible jump(s) found "
          f"(pid filter={args.pid}) ===")
    for e in sorted(events, key=lambda x: (x['pid'], x['frame'])):
        if e['kind'] == 'TELEPORT':
            print(f"  pid {e['pid']:>4} frame {e['frame']:>4}: TELEPORT "
                  f"dist={e['dist_m']:.1f}m  {e['pos_prev']} -> {e['pos_curr']}")
        else:
            print(f"  pid {e['pid']:>4} frame {e['frame']:>4}: CAP "
                  f"speed={e['kmh']:.1f}km/h (dist={e['dist_m']:.2f}m)  "
                  f"{e['pos_prev']} -> {e['pos_curr']}")

    pids_with_events = sorted(set(e['pid'] for e in events))
    print(f"\n{len(pids_with_events)} distinct pid(s) with at least one implausible jump: "
          f"{pids_with_events}")

    from collections import Counter
    counts = Counter(e['pid'] for e in events)
    max_dist = {}
    for e in events:
        if e['pid'] not in max_dist or e['dist_m'] > max_dist[e['pid']][0]:
            max_dist[e['pid']] = (e['dist_m'], e['frame'])
    print(f"\n=== per-pid summary (total events={len(events)}) ===")
    for pid, cnt in counts.most_common():
        d, fn = max_dist[pid]
        print(f"  pid {pid:>4}: {cnt:>3} event(s), worst jump {d:.1f}m at frame {fn}")
