"""
reconstruct_positions.py — one-time, offline reconstruction of enriched
per-frame player positions for a CV run whose TRACK_STUB predates position
transformation/team assignment.

Why this exists: Tracker.get_object_tracks() writes its stub to disk BEFORE
add_position_to_tracks/add_transformed_position_to_tracks/team_assigner.
resolve_all_teams ever run on the SAME in-memory tracks object - those
enrichment steps happen later in run_pipeline() and are never re-saved. So
the committed stub only has raw bboxes, not real-world position_transformed
coordinates or team labels (confirmed directly against a real stub file).
Detection/tracking (the expensive, GPU-bound stage) is already cached, so
this only needs to re-run the cheap enrichment stages plus one video
re-decode - not a reprocessing run.

Output is deliberately NOT a pickle: the dashboard app's own environment
does not have ultralytics/torch installed (dropped from
dashboard/requirements.txt - see that file's own header comment), so it
can never import trackers/team_assigner/view_transformer itself. This
script runs offline, once, wherever cv_pipeline's real dependencies are
installed, and writes a small, dependency-free JSON file the dashboard can
read with the standard library alone.

Usage:
    python reconstruct_positions.py --match-name corner1_liverpool_psg \
        --video corner_1_liverpool_psg.mp4 --output-dir output_videos
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np


def reconstruct(match_name, video_path, output_dir):
    from utils import read_video
    from trackers import Tracker
    from team_assigner import TeamAssigner
    from camera_movement_estimator import CameraMovementEstimator
    from view_transformer import ViewTransformer
    from run_cv_analysis import _smooth_positions

    stub_prefix = f'stubs/cv_analysis_{match_name}'
    cal_stub = f'{stub_prefix}_homography.pkl'
    cam_stub = f'{stub_prefix}_camera_movement.pkl'
    track_stub = f'{stub_prefix}_tracks.pkl'

    for p in (cal_stub, track_stub):
        if not os.path.exists(p):
            print(f"ERROR: required stub not found: {p}", file=sys.stderr)
            print("This script only reconstructs from an already-completed "
                  "run_cv_analysis.py run for this exact match-name - run "
                  "that first.", file=sys.stderr)
            sys.exit(1)

    print(f"Loading cached stubs for {match_name}...")
    with open(track_stub, 'rb') as f:
        tracks = pickle.load(f)
    with open(cal_stub, 'rb') as f:
        homography_per_frame = pickle.load(f)

    print("Re-decoding video (cheap - detection/tracking is NOT re-run)...")
    video_frames = read_video(video_path)
    n_frames = len(video_frames)

    view_transformer = ViewTransformer()
    _fb_H = view_transformer.persepctive_trasnformer.astype(np.float32)
    _fb_H_inv = np.linalg.inv(_fb_H).astype(np.float32)
    homography_per_frame = {
        fn: h if h is not None else (_fb_H, _fb_H_inv)
        for fn, h in homography_per_frame.items()
    }

    tracker = Tracker.__new__(Tracker)  # only need add_position_to_tracks - no model load needed
    tracker.applied_splits = {}  # normally set in __init__; only ever populated for one hardcoded
                                  # legacy clip (KNOWN_ID_SPLITS_VIDEO), never these corner segments
    tracker.add_position_to_tracks(tracks)

    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    if os.path.exists(cam_stub):
        with open(cam_stub, 'rb') as f:
            camera_movement_per_frame = pickle.load(f)
    else:
        print("No camera-movement stub found - estimating fresh (still no detection/tracking rerun).")
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    view_transformer.add_transformed_position_to_tracks(tracks, homography_per_frame)
    _smooth_positions(tracks, window=1)

    print("Resolving team colors (two-pass, same as the real pipeline)...")
    team_assigner = TeamAssigner()
    team_assigner.resolve_all_teams(video_frames, tracks['players'], tracks.get('referees', [{}] * n_frames))
    # Same 3-step sequence run_cv_analysis.py uses - resolve_all_teams alone
    # only builds internal vote/pending state, it does NOT write the final
    # 'team' field onto each track (confirmed: skipping these left every
    # player's team at its 0/unresolved default).
    team_assigner.finalize_fallback_assignments()
    team_assigner.merge_fragmented_tracks(video_frames, tracks['players'], exclude_pairs=list(tracker.applied_splits.items()))
    team_assigner.apply_final_teams(tracks['players'])

    print(f"Team colors (BGR): {team_assigner.locked_colors}")

    frames_out = []
    for frame_num in range(n_frames):
        frame_players = tracks['players'][frame_num] if frame_num < len(tracks['players']) else {}
        out_frame = []
        for pid, info in frame_players.items():
            team = info.get('team', 0)
            if team not in (1, 2):
                continue  # goalkeepers(3)/unresolved(0) excluded - see module docstring's scope note below
            pos = info.get('position_transformed')
            if pos is None:
                continue
            out_frame.append({
                "player_id": int(pid),
                "team": int(team),
                "is_goalkeeper": bool(info.get('is_goalkeeper', False)),
                "x": round(float(pos[0]), 3),
                "y": round(float(pos[1]), 3),
            })
        frames_out.append(out_frame)

    out_path = os.path.join(output_dir, match_name, 'player_positions.json')
    payload = {
        "match_name": match_name,
        "n_frames": n_frames,
        "team_colors_bgr": {str(k): [round(float(c), 1) for c in v] for k, v in team_assigner.locked_colors.items()},
        "frames": frames_out,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    n_with_data = sum(1 for fr in frames_out if fr)
    print(f"Wrote {out_path} - {n_with_data}/{n_frames} frames have resolved (team 1/2) player positions.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match-name', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--output-dir', default='output_videos')
    args = parser.parse_args()
    reconstruct(args.match_name, args.video, args.output_dir)


if __name__ == '__main__':
    main()
