#!/usr/bin/env python3
"""
apply_calibration_overrides_to_stats.py — propagate manual per-frame
calibration overrides (manual_calibration_tool.py, saved into this match's
calibration_status.json) into its speed/distance numbers in stats.json.

Why this is a separate script rather than "automatic": Game Board reads its
per-frame homography live, at display time (dashboard/player_repositioning.py
now merges calibration_status.json's frame_overrides in automatically - see
corner_kicks.apply_frame_overrides). Speed/distance stats are NOT read live
the same way - they're aggregated once, at CV-processing time, into
stats.json's players[] list. There is no live per-frame homography read left
by the time the dashboard displays a player's top speed / distance, so a
frame_overrides entry can only reach those numbers by re-running the
aggregation - this script is that re-run, scoped to just the cheap
enrichment stages (position transform + speed/distance), never the
expensive detection/tracking stage, same "re-run only what's cheap" pattern
reconstruct_positions.py already established for corner-kick data.

Usage:
    python apply_calibration_overrides_to_stats.py --match-name liverpool_psg_verified
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from view_transformer.view_transformer import ViewTransformer
from speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistance_Estimator
from run_cv_analysis import _build_player_stats, _build_calibration_stats

STUBS_DIR = Path(__file__).parent / "stubs"
OUTPUT_DIR = Path(__file__).parent / "output_videos"


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def reprocess(match_name):
    tracks_path = STUBS_DIR / f"cv_analysis_{match_name}_tracks.pkl"
    camera_path = STUBS_DIR / f"cv_analysis_{match_name}_camera_movement.pkl"
    homography_path = STUBS_DIR / f"cv_analysis_{match_name}_homography.pkl"
    stats_path = OUTPUT_DIR / match_name / "stats.json"
    calib_status_path = OUTPUT_DIR / match_name / "calibration_status.json"

    for p in (tracks_path, camera_path, homography_path, stats_path):
        if not p.exists():
            raise FileNotFoundError(f"required file missing: {p}")

    tracks = _load_pickle(tracks_path)
    camera_movement_per_frame = _load_pickle(camera_path)
    homography_per_frame = _load_pickle(homography_path)
    # ViewTransformer.add_transformed_position_to_tracks only checks
    # "frame_num in homography_per_frame", not whether the value itself is
    # None - a frame the calibrator explicitly failed on is stored as an
    # actual None value (not just a missing key), which crashes that
    # unconditional [0].astype() call. Pre-existing fragility in shared
    # production code, not something to change here - just avoid it.
    homography_per_frame = {f: h for f, h in homography_per_frame.items() if h is not None}

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    team_by_pid = {p["player_id"]: p["team"] for p in stats["players"]}

    n_overrides = 0
    if calib_status_path.exists():
        with open(calib_status_path, "r", encoding="utf-8") as f:
            calib_status = json.load(f)
        overrides = calib_status.get("frame_overrides", {}) or {}
        for frame_str, entry in overrides.items():
            # ViewTransformer.add_transformed_position_to_tracks calls
            # .astype() directly on these - must be numpy arrays, same as
            # what's already in the unpickled homography_per_frame, not the
            # plain JSON-serializable nested lists calibration_status.json
            # stores them as.
            homography_per_frame[int(frame_str)] = (np.array(entry["H"]), np.array(entry["H_inv"]))
        n_overrides = len(overrides)
    print(f"Applying {n_overrides} frame override(s) on top of {len(homography_per_frame)} "
          f"originally-solved frames.")

    # Re-attach existing team labels (never re-run team assignment itself -
    # that's the expensive, non-deterministic-without-a-fixed-seed stage,
    # and unaffected by a homography correction).
    for frame in tracks["players"]:
        for pid, info in frame.items():
            info["team"] = team_by_pid.get(pid, 0)

    # Re-run only the cheap enrichment stages, in the same order main.py /
    # run_cv_analysis.py originally run them.
    for obj, obj_tracks in tracks.items():
        for frame in obj_tracks:
            for info in frame.values():
                bbox = info["bbox"]
                if obj == "ball":
                    from utils.bbox_utils import get_center_of_bbox
                    info["position"] = get_center_of_bbox(bbox)
                else:
                    from utils.bbox_utils import get_foot_position
                    info["position"] = get_foot_position(bbox)

    CameraMovementEstimator.add_adjust_positions_to_tracks(None, tracks, camera_movement_per_frame)

    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks, homography_per_frame=homography_per_frame)

    calibration_confidence_per_frame = view_transformer.compute_frame_confidence(tracks)

    fps = stats["video"]["fps"]
    speed_estimator = SpeedAndDistance_Estimator(fps)
    speed_estimator.add_speed_and_distance_to_tracks(tracks, calibration_confidence_per_frame)

    new_players = _build_player_stats(tracks)
    new_calibration = _build_calibration_stats(calibration_confidence_per_frame)

    old_players_by_id = {p["player_id"]: p for p in stats["players"]}
    changed = []
    for p in new_players:
        old = old_players_by_id.get(p["player_id"])
        if old and (abs(old["top_speed_kmh"] - p["top_speed_kmh"]) > 0.01
                    or abs(old["total_distance_m"] - p["total_distance_m"]) > 0.01):
            changed.append((p["player_id"], old, p))

    stats["players"] = new_players
    stats["calibration"] = new_calibration
    tmp = str(stats_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    import os
    os.replace(tmp, stats_path)

    print(f"Rewrote {stats_path}")
    print(f"Calibration confidence: mean={new_calibration['mean_confidence']} "
          f"high={new_calibration['pct_high']}% medium={new_calibration['pct_medium']}% low={new_calibration['pct_low']}%")
    print(f"{len(changed)} player(s) had a materially changed top_speed_kmh or total_distance_m:")
    for pid, old, new in changed[:20]:
        print(f"  player {pid}: top_speed {old['top_speed_kmh']}->{new['top_speed_kmh']} km/h, "
              f"distance {old['total_distance_m']}->{new['total_distance_m']} m")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match-name", required=True, help="e.g. liverpool_psg_verified (matches the stubs/output_videos naming)")
    args = ap.parse_args()
    reprocess(args.match_name)
