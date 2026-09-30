#!/usr/bin/env python3
"""
backfill_tactical_event_frames_to_stats.py — recompute tactical_events
(BREAK/PRESS/SPACE/SPRINT/...) from the cached tracks stub and rewrite
stats.json's tactical_events.highlights to include each event's real frame
number, instead of only the coarse 20-second window bucket
_build_tactical_events_stats previously kept.

Why a separate script, not automatic: same reasoning
apply_calibration_overrides_to_stats.py already established - tactical
event detection (space-control grid + event detection + window ranking) is
a real, if cheap, recomputation, not a live per-request read. Getting the
frame field into an already-processed match's stats.json means re-running
this one stage from the cached tracks.pkl, never the expensive detection/
tracking stage itself, and never a full run_cv_analysis.py pass.

Why the video file is still needed (unlike apply_calibration_overrides_to_
stats.py, which only touches pickled data): compute_pitch_verts_from_tracks
needs the frame's pixel dimensions (h, w) to build the space-control
sampling grid. Only the dimensions are read (cv2.VideoCapture's own
properties, no frame decode) - the curated match's already-committed
peak_momentum_segment.mp4 covers this with no new dependency.

Real, confirmed dependency this script has that apply_calibration_overrides_
to_stats.py didn't: the cached tracks.pkl is genuinely raw (bbox +
is_goalkeeper only - confirmed directly, not assumed) - team, speed, and
position_transformed are all added by earlier in-place pipeline stages,
never persisted to the stub. tactical_events_detector.detect() reads
exactly position_transformed/speed/team (confirmed by reading its own field
accesses) - so this script re-runs the same enrichment chain
apply_calibration_overrides_to_stats.py already established (team
reattachment, position, camera-movement adjustment, view transform,
speed/distance) before tactical-event detection, not just the detection
stage alone. A sanity check (recomputed event counts must equal the
already-shipped counts exactly) guards against silently shipping a result
from a chain that diverged somewhere.

Usage:
    python backfill_tactical_event_frames_to_stats.py --match-name liverpool_psg_verified \\
        --video-path ../dashboard/curated_matches/liverpool_psg/peak_momentum_segment.mp4
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from view_transformer.view_transformer import ViewTransformer
from speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistance_Estimator
from tactical_events.space_control import (
    build_pitch_mask, build_sampling_grid, compute_space_control_per_frame,
    compute_pitch_verts_from_tracks,
)
from tactical_events.tactical_events_detector import TacticalEventsDetector
from tactical_events.event_ranking import rank_events_by_window
from render_output3 import STEP as _TAC_STEP, _FALLBACK_PITCH_VERTS as _TAC_FALLBACK_VERTS
from run_cv_analysis import _build_tactical_events_stats

STUBS_DIR = Path(__file__).parent / "stubs"
OUTPUT_DIR = Path(__file__).parent / "output_videos"


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _video_dims(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"couldn't open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not w or not h:
        raise RuntimeError(f"couldn't read frame dimensions from {video_path}")
    return h, w


def reprocess(match_name, video_path):
    tracks_path = STUBS_DIR / f"cv_analysis_{match_name}_tracks.pkl"
    camera_path = STUBS_DIR / f"cv_analysis_{match_name}_camera_movement.pkl"
    homography_path = STUBS_DIR / f"cv_analysis_{match_name}_homography.pkl"
    stats_path = OUTPUT_DIR / match_name / "stats.json"
    for p in (tracks_path, camera_path, homography_path, stats_path):
        if not p.exists():
            raise FileNotFoundError(f"required file missing: {p}")
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video file missing: {video_path}")

    tracks = _load_pickle(tracks_path)
    camera_movement_per_frame = _load_pickle(camera_path)
    homography_per_frame = _load_pickle(homography_path)
    # Same reasoning as apply_calibration_overrides_to_stats.py: a frame the
    # calibrator explicitly failed on is stored as an actual None value, not
    # a missing key - ViewTransformer's unconditional [0].astype() crashes
    # on that if left in.
    homography_per_frame = {f: h for f, h in homography_per_frame.items() if h is not None}

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    fps = stats["video"]["fps"]
    team_by_pid = {p["player_id"]: p["team"] for p in stats["players"]}

    # ── Re-run the same enrichment chain apply_calibration_overrides_to_
    # stats.py already established (team reattachment, position, camera-
    # movement adjustment, view transform, speed/distance) - confirmed
    # directly that the cached tracks.pkl is raw (bbox + is_goalkeeper only)
    # and tactical_events_detector.detect() needs position_transformed/
    # speed/team, none of which survive in the stub on their own.
    for frame in tracks["players"]:
        for pid, info in frame.items():
            info["team"] = team_by_pid.get(pid, 0)

    from utils.bbox_utils import get_center_of_bbox, get_foot_position
    for obj, obj_tracks in tracks.items():
        for frame in obj_tracks:
            for info in frame.values():
                bbox = info["bbox"]
                info["position"] = get_center_of_bbox(bbox) if obj == "ball" else get_foot_position(bbox)

    CameraMovementEstimator.add_adjust_positions_to_tracks(None, tracks, camera_movement_per_frame)

    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks, homography_per_frame=homography_per_frame)

    calibration_confidence_per_frame = view_transformer.compute_frame_confidence(tracks)
    speed_estimator = SpeedAndDistance_Estimator(fps)
    speed_estimator.add_speed_and_distance_to_tracks(tracks, calibration_confidence_per_frame)

    h, w = _video_dims(video_path)
    pitch_verts = compute_pitch_verts_from_tracks(tracks, h, w, fallback_verts=_TAC_FALLBACK_VERTS)
    pitch_mask = build_pitch_mask(pitch_verts, h, w)
    grid, grid_rows, grid_cols, row_idx, col_idx = build_sampling_grid(pitch_mask, _TAC_STEP)
    space_control_per_frame = compute_space_control_per_frame(tracks, grid, top_frac=0.10)

    detector = TacticalEventsDetector(fps=fps)
    events_by_frame = detector.detect(tracks, space_control_per_frame)
    ranked_windows, window_frames = rank_events_by_window(
        events_by_frame, tracks, space_control_per_frame, fps=fps)

    old_highlights = stats.get("tactical_events", {}).get("highlights", [])
    new_tactical_events = _build_tactical_events_stats(detector, ranked_windows)

    # Sanity check before trusting this over the shipped data: counts/total
    # should match exactly (same detector, same tracks, same fps) - if they
    # don't, something about this recomputation diverged from the original
    # run and the result shouldn't silently replace what's there.
    old_counts = stats.get("tactical_events", {}).get("counts", {})
    if old_counts != new_tactical_events["counts"]:
        raise RuntimeError(
            f"recomputed event counts don't match the original: "
            f"old={old_counts} new={new_tactical_events['counts']} - not overwriting stats.json"
        )

    stats["tactical_events"] = new_tactical_events
    tmp = str(stats_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    os.replace(tmp, stats_path)

    new_highlights = new_tactical_events["highlights"]
    n_with_frame = sum(1 for e in new_highlights if e.get("frame") is not None)
    print(f"Rewrote {stats_path}")
    print(f"counts verified unchanged: {new_tactical_events['counts']}")
    print(f"highlights: {len(old_highlights)} -> {len(new_highlights)} "
          f"({n_with_frame}/{len(new_highlights)} now have a real frame number)")
    for e in new_highlights[:5]:
        print(" ", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match-name", required=True, help="e.g. liverpool_psg_verified (matches the stubs/output_videos naming)")
    ap.add_argument("--video-path", required=True, help="the analyzed clip, e.g. ../dashboard/curated_matches/liverpool_psg/peak_momentum_segment.mp4")
    args = ap.parse_args()
    reprocess(args.match_name, args.video_path)
