"""
export_repositioning_data.py — one-time, offline export of a CV run's cached
tracks/camera-movement/homography stubs into a single dependency-free JSON
file the dashboard can read with the standard library alone.

Why this exists: same reason reconstruct_positions.py exists (see that
script's own docstring) - the dashboard's own environment does not have
torch/ultralytics installed, so it can never unpickle or import anything
from trackers/camera_movement_estimator/view_transformer directly. Unlike
reconstruct_positions.py, this script does NOT need to re-run team
assignment, position transformation, or re-decode the video at all: the
player-repositioning feature only needs the three already-cached, already-
plain-Python/numpy stubs (raw pixel bboxes, per-frame camera displacement,
per-frame homography matrices) - none of which reference any torch object -
so this is a straight pickle-to-JSON re-serialization, not a reprocessing
run. Team affiliation for a track is cross-referenced from that match's
already-exported stats.json (players[].team) at read time in the dashboard
instead of being re-derived here, which is what keeps this script free of
the team_assigner/tracker/torch dependency chain entirely.

Usage:
    python export_repositioning_data.py --match-name liverpool_psg_verified \
        --output-dir output_videos
"""
import argparse
import json
import os
import pickle
import sys


def export(match_name, output_dir):
    stub_prefix = f'stubs/cv_analysis_{match_name}'
    track_stub = f'{stub_prefix}_tracks.pkl'
    cam_stub = f'{stub_prefix}_camera_movement.pkl'
    homog_stub = f'{stub_prefix}_homography.pkl'

    for p in (track_stub, cam_stub, homog_stub):
        if not os.path.exists(p):
            print(f"ERROR: required stub not found: {p}", file=sys.stderr)
            print("This script only exports from an already-completed "
                  "run_cv_analysis.py run for this exact match-name - run "
                  "that first.", file=sys.stderr)
            sys.exit(1)

    print(f"Loading cached stubs for {match_name}...")
    with open(track_stub, 'rb') as f:
        tracks = pickle.load(f)
    with open(cam_stub, 'rb') as f:
        camera_movement = pickle.load(f)
    with open(homog_stub, 'rb') as f:
        homography = pickle.load(f)

    n_frames = len(camera_movement)

    # tracks[kind][frame_idx] is {track_id: {'bbox': [x1,y1,x2,y2], ...}} -
    # bbox is the only field the repositioning feature needs (occlusion
    # checking, cutout extraction); is_goalkeeper is kept too since it's
    # already there and free. JSON object keys must be strings, so
    # track_id is stringified - callers convert back to int/compare as
    # strings consistently, same as stats.json's own player_id convention.
    def export_tracks(kind):
        out = []
        frames = tracks.get(kind, [{}] * n_frames)
        for frame_idx in range(n_frames):
            frame = frames[frame_idx] if frame_idx < len(frames) else {}
            out.append({
                str(tid): {
                    "bbox": [round(float(v), 2) for v in info["bbox"]],
                    "is_goalkeeper": bool(info.get("is_goalkeeper", False)),
                }
                for tid, info in frame.items()
                if info.get("bbox") is not None
            })
        return out

    players_out = export_tracks("players")
    referees_out = export_tracks("referees")
    ball_out = export_tracks("ball")

    # camera_movement[frame_idx] = [dx, dy] (0,0 for frames with no detected
    # motion) - already a plain list, exported as-is.
    camera_movement_out = [[round(float(v[0]), 4), round(float(v[1]), 4)] for v in camera_movement]

    # homography[frame_idx] = (H_img_to_pitch, H_pitch_to_img) as 3x3 numpy
    # arrays, or None for a frame with no usable calibration that frame -
    # exported as nested lists (JSON has no numpy array type), None stays None.
    homography_out = {}
    for frame_idx in range(n_frames):
        h = homography.get(frame_idx)
        if h is None:
            homography_out[str(frame_idx)] = None
        else:
            h_fwd, h_inv = h
            homography_out[str(frame_idx)] = [
                [[round(float(v), 8) for v in row] for row in h_fwd],
                [[round(float(v), 8) for v in row] for row in h_inv],
            ]

    payload = {
        "match_name": match_name,
        "n_frames": n_frames,
        "players": players_out,
        "referees": referees_out,
        "ball": ball_out,
        "camera_movement": camera_movement_out,
        "homography": homography_out,
    }

    out_path = os.path.join(output_dir, match_name, 'repositioning_data.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    n_with_homog = sum(1 for v in homography_out.values() if v is not None)
    print(f"Wrote {out_path} - {n_frames} frames, {n_with_homog}/{n_frames} with usable homography.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-name", required=True)
    parser.add_argument("--output-dir", default="output_videos")
    args = parser.parse_args()
    export(args.match_name, args.output_dir)
