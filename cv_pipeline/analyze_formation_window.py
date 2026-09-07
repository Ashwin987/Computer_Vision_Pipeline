"""
analyze_formation_window.py — formation shape + defensive-line-height
analysis for a single reliable window, as the foundation for a future
multi-window tactical recommendation system.

Standalone, additive analysis script. Does NOT import, modify, or touch
main.py, fast_setup.py, fast_render_*.py, or any render_output*.py file
(it only *imports* constants and the already-proven `_link_identities`
function from render_output3.py, exactly as-is — the pure proximity+team
matcher, no jersey-colour gate, since at this short a window it already
works well, per render_output3.py's own use of it).

Deliberately does NOT reuse fast_common.load_fast_pipeline(): that
function also draws full annotated/output video frames (multiple extra
1920x1080 frame lists) which this script has no use for and which would
roughly double-to-triple the memory footprint for no benefit. Instead
this script recomputes just the two purely-numeric enrichment steps it
needs (position -> position_adjusted -> position_transformed) directly
from the cached fastpipe_*.pkl stubs, and streams the video only for the
window's frames (never the whole match) to get real jersey-colour team
labels — team can't be inferred from bbox coordinates alone.

Pipeline for a window [start_frame, end_frame):
  1. Load cached tracks/camera-movement/homography, sliced to the window.
  2. position (bbox math) -> position_adjusted (camera-compensated) ->
     position_transformed (real-world metres, via ViewTransformer).
  3. Stream just the window's video frames once to resolve a real
     jersey-colour team (1/2) per player per frame.
  4. render_output3._link_identities() to collapse fragmented raw IDs
     into stable per-player identities for the window.
  5. Per team: exclude the goalkeeper (heuristic: the identity whose
     mean position is most extreme toward either goal line — keepers
     sit deepest, essentially always more so than any outfield player),
     then cluster the remaining outfield players into rows by gaps along
     the defensive-to-attacking axis, per frame, to get a formation
     string and defensive-line depth.
  6. Report the modal formation and defensive-line-height stats across
     the window, plus how many frames were confidently classified.

Usage:
    python analyze_formation_window.py [start_frame] [end_frame] [--source {fastpipe,main}]
    (defaults to --source fastpipe, window 7200:8640, the initial test window)

    python analyze_formation_window.py --source main
    (analyses the WHOLE short 750-frame 121364_0.mp4 clip using main.py's
    cached full-resolution/every-frame tracking stubs instead of
    fast_setup.py's half-resolution/frame-skip ones — same downstream
    analysis, different upstream tracking fidelity, to isolate whether
    fastpipe's noisy formation reads are caused by its detection
    fragmentation specifically or are more fundamental to this approach.)
"""

import os
import pickle
import sys
import time
from collections import Counter

import cv2
import numpy as np

from utils import get_center_of_bbox, get_foot_position
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
# pitch_calibrator (ultralytics -> torch) must be imported before team_assigner
# (sklearn): on Windows, importing sklearn first corrupts torch's OpenMP DLL
# init ("DLL initialization routine failed" loading torch's c10.dll) — a
# known sklearn/torch import-order conflict, not specific to this script.
# fast_common.py avoids it the same way (imports its YOLO-based Tracker
# before TeamAssigner).
from pitch_calibrator import _PITCH_X_MAX, _PITCH_Y_MAX
from team_assigner import TeamAssigner
from render_output3 import _link_identities, PROXIMITY_PX, LOST_BUF_FRAMES, MIN_APPEARANCE_FRAMES
import fast_common

PITCH_LENGTH = _PITCH_X_MAX   # 105.0 m, goal-line-to-goal-line (the "depth" axis)
PITCH_WIDTH  = _PITCH_Y_MAX   #  68.0 m, touchline-to-touchline

# Per-frame homography in the cached calibration stub is occasionally
# degenerate (a bad keypoint detection RANSAC-fits a poorly-conditioned H),
# which extrapolates a player's pixel position to a wildly impossible world
# coordinate (observed: x as extreme as -70,000m, y as extreme as +597,000m
# on this match). ViewTransformer only checks that the source PIXEL point is
# inside the visible-pitch polygon — it has no way to know its own homography
# solve was bad — so those outliers pass through as valid-looking floats.
# Reject anything outside the pitch plus a generous margin (touchline/
# goal-line calibration noise is normal; being thousands of metres away is
# not) before using position_transformed for anything.
PITCH_MARGIN = 15.0

CLUSTER_FRAMES = [0, 10, 20, 30, 40]   # window-local offsets for the jersey KMeans fit

# A new tactical "row" starts when the gap between two consecutive
# players' depth (sorted defensive->attacking) exceeds this many metres.
# Within-row spread (e.g. a back four, or a front three) is typically a
# few metres; the gap between distinct lines (defence/midfield/attack) in
# a settled shape is usually noticeably larger. Heuristic, not tuned —
# flagged in the summary as such.
ROW_GAP_METRES = 8.0

# Below this many valid outfield players (of up to 10) detected in a
# frame, the read is treated as too noisy/incomplete to classify.
MIN_OUTFIELD_FOR_CONFIDENT_READ = 8

# Rolling-window smoothing for formation classification: treating every
# raw frame as an independent snapshot is noisy (observed modal-formation
# shares of only 4.9%-22.8% across every test so far) — a real settled
# shape is a multi-second pattern, not a single-instant one, and a lot of
# the frame-to-frame flicker is just detection/tracking jitter rather than
# genuine shape change. Instead: average each player's position over a
# rolling window, classify ONCE per window on the smoothed positions, and
# slide the window across the clip.
WINDOW_SECONDS = 2.5     # ~48-72 frames at 24fps, per spec; 60 frames at 24fps
STEP_SECONDS   = 1.0     # slide the window forward this often
# A player only counts toward a window's smoothed average if they have a
# valid position in at least this fraction of the window's frames — guards
# against averaging in a player who was mostly absent (long tracking gap/
# interpolation artifact) for just a couple of real detections.
MIN_PRESENCE_FRAC = 0.6

# A real goalkeeper's mean position essentially never sits farther than
# this from their own goal line, even a very aggressive sweeper-keeper.
# Goalkeepers usually wear a kit colour distinct from BOTH outfield teams,
# so the two-cluster jersey KMeans often fails to classify them into team
# 1/2 at all — the keeper then simply never appears in valid_sids. When
# that happens, the "most extreme player" heuristic below would otherwise
# grab a genuine deep defender and mislabel them as the keeper (observed
# on this window's Team 1: the extremum candidate averaged 43.9m from
# goal — essentially the halfway line, not a keeper). Above this
# threshold we conclude no identity in the pool looks like a real keeper
# and skip GK exclusion entirely rather than silently corrupting the
# defensive line by dropping a real defender.
GK_MAX_DEPTH_M = 25.0

DEFAULT_START = 7200
DEFAULT_END   = 8640

# Two data sources for the SAME downstream analysis (position math, team
# assignment, identity linking, row clustering, GK/defensive-line stats):
#   'fastpipe' - fast_setup.py's half-resolution/every-3rd-frame pipeline,
#                the full 42,475-frame Liverpool/PSG match (windowed).
#   'main'     - main.py's proven full-resolution/every-frame tracking
#                stubs for the short 750-frame Union Berlin clip
#                (121364_0.mp4), analysed whole — no windowing needed at
#                this length. Used to test whether fastpipe's noisy
#                formation reads are caused by its half-res/frame-skip
#                fragmentation specifically, or are a more fundamental
#                limit of this approach.
# main.py's own stub files have no separate meta.pkl (fps/n_frames) — fps
# is read directly from the video file via cv2 below (both sample videos
# are actually 25fps; a previously hardcoded 24 here silently skewed every
# frame-index -> seconds conversion in this script and analyze_danger_
# score.py by a factor of 25/24 without anything surfacing the mismatch).
SOURCES = {
    'fastpipe': {
        'meta_path':  fast_common.CACHE_META,
        'track_path': fast_common.CACHE_TRACK,
        'cam_path':   fast_common.CACHE_CAM,
        'cal_path':   fast_common.CACHE_CAL,
        'video_path': None,   # resolved from meta
        # fast_setup.py's DETECT_EVERY=3 (half-res, every-3rd-frame player/
        # ball/referee detection) — a bbox present at any OTHER frame index
        # is guaranteed to be bbox-level linear interpolation, never a real
        # YOLO detection. Used by analyze_danger_score.py's ball-detection-
        # gap diagnosis to tell "real" from "interpolated/fabricated".
        'detect_every': 3,
    },
    'main': {
        'meta_path':  None,
        'track_path': 'stubs/track_stubs_121364.pkl',
        'cam_path':   'stubs/camera_movement_stub_121364.pkl',
        'cal_path':   'stubs/homography_stub_121364.pkl',
        'video_path': 'Match_videos/121364_0.mp4',
        'detect_every': 1,   # main.py's Tracker detects every frame — no skip
    },
}


def load_window_tracks(start_frame, end_frame, source='fastpipe'):
    """Load the cached stubs for `source` and compute position_transformed
    for the [start_frame, end_frame) window only — no video frames, no
    Tracker/YOLO instantiation, no annotated-frame rendering."""
    cfg = SOURCES[source]

    if cfg['meta_path'] is not None:
        with open(cfg['meta_path'], 'rb') as f:
            meta = pickle.load(f)
        video_path, fps, n_frames = meta['video_path'], meta['fps'], meta['n_frames']
    else:
        video_path = cfg['video_path']
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))   # real fps, not a hardcoded assumption
        cap.release()

    for path in (cfg['track_path'], cfg['cam_path'], cfg['cal_path']):
        if not os.path.exists(path):
            raise RuntimeError(f"Missing cached stub for source='{source}': {path}")

    end_frame = min(end_frame, n_frames)
    start_frame = max(0, start_frame)
    if start_frame >= end_frame:
        raise ValueError(f"start_frame ({start_frame}) must be < end_frame ({end_frame})")

    with open(cfg['track_path'], 'rb') as f:
        full_tracks = pickle.load(f)
    with open(cfg['cam_path'], 'rb') as f:
        full_camera_movement = pickle.load(f)
    with open(cfg['cal_path'], 'rb') as f:
        full_homography = pickle.load(f)

    tracks = {obj: full_tracks[obj][start_frame:end_frame]
              for obj in ('players', 'referees', 'ball')}
    camera_movement_per_frame = full_camera_movement[start_frame:end_frame]
    homography_per_frame = {i: full_homography.get(start_frame + i)
                            for i in range(end_frame - start_frame)}

    # position: pure bbox math (no video needed)
    for object_name, object_tracks in tracks.items():
        for frame_tracks in object_tracks:
            for track_id, track_info in frame_tracks.items():
                bbox = track_info['bbox']
                if object_name == 'ball':
                    track_info['position'] = get_center_of_bbox(bbox)
                else:
                    track_info['position'] = get_foot_position(bbox)

    # position_adjusted: camera-compensated (pure math). CameraMovementEstimator's
    # constructor only uses its `frame` arg for frame.shape (building a feature
    # mask that add_adjust_positions_to_tracks never touches) — a tiny dummy
    # array avoids needing any real pixel data for this step.
    dummy_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    camera_movement_estimator = CameraMovementEstimator(dummy_frame)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # position_transformed: real-world metres (pure math), same fallback-to-fixed-
    # homography behaviour as fast_common.load_fast_pipeline uses for consistency.
    view_transformer = ViewTransformer()
    fb_H = view_transformer.persepctive_trasnformer.astype(np.float32)
    fb_H_inv = np.linalg.inv(fb_H).astype(np.float32)
    homography_per_frame = {
        fn: h if h is not None else (fb_H, fb_H_inv)
        for fn, h in homography_per_frame.items()
    }
    view_transformer.add_transformed_position_to_tracks(tracks, homography_per_frame)

    return tracks, video_path, fps, start_frame, end_frame


def assign_teams_for_window(tracks, video_path, start_frame, end_frame):
    """Stream ONLY this window's video frames once, resolving
    tracks['players'][f][pid]['team'] for every player in every frame."""
    n = end_frame - start_frame
    team_assigner = TeamAssigner()
    max_cluster_fn = min(max(CLUSTER_FRAMES), n - 1)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    early_frames = []
    for fn in range(max_cluster_fn + 1):
        ret, frame = cap.read()
        if not ret:
            raise IOError(f"video ended at window-local frame {fn}")
        early_frames.append(frame)

    team_assigner.assign_team_color(early_frames, tracks['players'])

    for fn, frame in enumerate(early_frames):
        for pid, info in tracks['players'][fn].items():
            bbox = info.get('bbox')
            if bbox is not None:
                info['team'] = team_assigner.get_player_team(frame, bbox, pid)
    early_frames = None

    fn = max_cluster_fn + 1
    while fn < n:
        ret, frame = cap.read()
        if not ret:
            break
        for pid, info in tracks['players'][fn].items():
            bbox = info.get('bbox')
            if bbox is not None:
                info['team'] = team_assigner.get_player_team(frame, bbox, pid)
        fn += 1
    cap.release()

    if fn != n:
        print(f"  WARNING: video yielded {fn} of {n} window frames — "
              f"remaining frames have no team and will read as noisy/unclassified.")


def build_team_positions(tracks, n, pid_to_sid, sid_team, valid_sids, team):
    """Per frame, dict {sid: (x_m, y_m)} restricted to trusted (linked,
    non-noise) identities of `team` with a valid position_transformed."""
    team_sids = {sid for sid in valid_sids if sid_team.get(sid) == team}
    frames = []
    for frame_num in range(n):
        player_data = tracks['players'][frame_num] if frame_num < len(tracks['players']) else {}
        frame_positions = {}
        for pid, info in player_data.items():
            sid = pid_to_sid.get(pid)
            if sid not in team_sids:
                continue
            pos = info.get('position_transformed')
            if pos is None:
                continue
            x, y = float(pos[0]), float(pos[1])
            if not (-PITCH_MARGIN <= x <= PITCH_LENGTH + PITCH_MARGIN
                    and -PITCH_MARGIN <= y <= PITCH_WIDTH + PITCH_MARGIN):
                continue   # degenerate-homography outlier — see PITCH_MARGIN
            # Clip to the true pitch rectangle: the margin above only guards
            # against catastrophic homography blowups, not ordinary
            # touchline/goal-line calibration noise — without clipping, a
            # point a few metres over the line would still produce a
            # nonsensical negative "distance from own goal line" downstream.
            x = min(max(x, 0.0), PITCH_LENGTH)
            y = min(max(y, 0.0), PITCH_WIDTH)
            frame_positions[sid] = (x, y)
        frames.append(frame_positions)
    return frames, team_sids


def _pooled_mean_x(frames):
    xs = [x for frame_positions in frames for (x, _y) in frame_positions.values()]
    return float(np.mean(xs)) if xs else None


def determine_goal_sides(frames_by_team):
    """Decide which goal (x≈0 or x≈105) each team defends, RELATIVE to the
    other team, rather than against a fixed pitch-halfway line.

    An absolute halfway threshold can put both teams on the same side
    during a lopsided spell of play within a short window (e.g. one team
    camped deep defending a sustained attack) — every player pushed
    toward one end drags BOTH teams' pooled mean below halfway, which was
    observed on the first pass of this window and is impossible in
    reality: the two teams can never defend the same goal. Comparing the
    two teams' pooled mean x against each other instead guarantees one
    is assigned each side.
    """
    means = {team: _pooled_mean_x(frames) for team, frames in frames_by_team.items()}
    teams_by_mean_x = sorted(means, key=lambda t: (means[t] if means[t] is not None
                                                     else float('inf')))
    return {team: (rank == 0) for rank, team in enumerate(teams_by_mean_x)}


def identify_goalkeeper(frames, defends_low):
    """Heuristic: the identity with the most extreme mean depth (x)
    toward the team's own goal line is the keeper — even the deepest
    outfield defender sits meaningfully higher, on average, than the
    keeper. Returns (gk_sid_or_None, candidate_depth_m_or_None); the
    caller decides whether candidate_depth is plausible (see
    GK_MAX_DEPTH_M) — this function doesn't reject anything itself so the
    caller can report the rejected candidate's depth for diagnosis."""
    sums, counts = {}, {}
    for frame_positions in frames:
        for sid, (x, _y) in frame_positions.items():
            sums[sid] = sums.get(sid, 0.0) + x
            counts[sid] = counts.get(sid, 0) + 1
    means = {sid: sums[sid] / counts[sid] for sid in sums if counts.get(sid, 0) > 0}
    if not means:
        return None, None
    candidate = (min(means, key=lambda s: means[s]) if defends_low
                 else max(means, key=lambda s: means[s]))
    candidate_depth = means[candidate] if defends_low else (PITCH_LENGTH - means[candidate])
    return candidate, candidate_depth


def cluster_rows(depths_sorted, gap_threshold):
    rows = [[depths_sorted[0]]]
    for d in depths_sorted[1:]:
        if d - rows[-1][-1] > gap_threshold:
            rows.append([d])
        else:
            rows[-1].append(d)
    return rows


def _formation_from_depths(depths):
    depths.sort()
    rows = cluster_rows(depths, ROW_GAP_METRES)
    formation = "-".join(str(len(row)) for row in rows)
    def_line_depth = float(np.mean(rows[0]))
    return formation, def_line_depth


def _summarize_reads(formations, def_line_depths, noisy_count, unit_label):
    formation_counts = Counter(formations)
    modal_formation, modal_count = (formation_counts.most_common(1)[0]
                                     if formation_counts else (None, 0))
    classified = len(formations)
    return {
        f'classified_{unit_label}': classified,
        f'noisy_{unit_label}': noisy_count,
        'formation_counts': formation_counts,
        'modal_formation': modal_formation,
        'modal_formation_share': (modal_count / classified) if classified else 0.0,
        'def_line_avg_m': float(np.mean(def_line_depths)) if def_line_depths else None,
        'def_line_min_m': float(np.min(def_line_depths)) if def_line_depths else None,
        'def_line_max_m': float(np.max(def_line_depths)) if def_line_depths else None,
    }


def _analyze_team_perframe(frames, gk_sid, defends_low):
    """Original approach: classify every single frame independently."""
    formations, def_line_depths = [], []
    noisy_frames = 0

    for frame_positions in frames:
        depths = [x if defends_low else (PITCH_LENGTH - x)
                  for sid, (x, _y) in frame_positions.items() if sid != gk_sid]

        if len(depths) < MIN_OUTFIELD_FOR_CONFIDENT_READ:
            noisy_frames += 1
            continue

        formation, def_line_depth = _formation_from_depths(depths)
        formations.append(formation)
        def_line_depths.append(def_line_depth)

    return _summarize_reads(formations, def_line_depths, noisy_frames, 'frames')


def _analyze_team_windowed(frames, gk_sid, defends_low, fps):
    """Average each player's position over a rolling window before
    classifying, instead of classifying every raw frame — a settled shape
    is a multi-second pattern, not a single-instant one."""
    n = len(frames)
    window_frames = max(1, round(WINDOW_SECONDS * fps))
    step_frames = max(1, round(STEP_SECONDS * fps))
    min_presence = MIN_PRESENCE_FRAC * window_frames

    formations, def_line_depths = [], []
    noisy_windows = 0

    w_start = 0
    while w_start + window_frames <= n:
        w_end = w_start + window_frames
        sums, counts = {}, {}
        for frame_positions in frames[w_start:w_end]:
            for sid, (x, y) in frame_positions.items():
                if sid == gk_sid:
                    continue
                s = sums.get(sid)
                if s is None:
                    sums[sid] = [x, y]
                else:
                    s[0] += x; s[1] += y
                counts[sid] = counts.get(sid, 0) + 1

        depths = []
        for sid, cnt in counts.items():
            if cnt < min_presence:
                continue   # present in too little of this window — see MIN_PRESENCE_FRAC
            avg_x = sums[sid][0] / cnt
            depths.append(avg_x if defends_low else (PITCH_LENGTH - avg_x))

        if len(depths) < MIN_OUTFIELD_FOR_CONFIDENT_READ:
            noisy_windows += 1
        else:
            formation, def_line_depth = _formation_from_depths(depths)
            formations.append(formation)
            def_line_depths.append(def_line_depth)

        w_start += step_frames

    result = _summarize_reads(formations, def_line_depths, noisy_windows, 'windows')
    result['window_frames'] = window_frames
    result['step_frames'] = step_frames
    return result


def analyze_team(frames, team_sids, defends_low, team, fps):
    gk_candidate, gk_candidate_depth = identify_goalkeeper(frames, defends_low)
    gk_confident = gk_candidate is not None and gk_candidate_depth <= GK_MAX_DEPTH_M
    gk_sid = gk_candidate if gk_confident else None
    outfield_sids = team_sids - ({gk_sid} if gk_sid is not None else set())

    return {
        'team': team,
        'defends_low_x': defends_low,
        'goalkeeper_sid': gk_sid,
        'goalkeeper_confident': gk_confident,
        'goalkeeper_candidate_rejected': (gk_candidate if not gk_confident else None),
        'goalkeeper_candidate_rejected_depth_m': (gk_candidate_depth if not gk_confident else None),
        'n_outfield_identities': len(outfield_sids),
        'perframe': _analyze_team_perframe(frames, gk_sid, defends_low),
        'windowed': _analyze_team_windowed(frames, gk_sid, defends_low, fps),
    }


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Formation shape + defensive-line-height analysis "
                     "for a window (or, with --source main, a whole short clip).")
    p.add_argument('start_frame', type=int, nargs='?', default=None)
    p.add_argument('end_frame', type=int, nargs='?', default=None)
    p.add_argument('--source', choices=sorted(SOURCES), default='fastpipe',
                    help="'fastpipe' (default): fast_setup.py's cached "
                         "half-res/frame-skip pipeline, windowed. "
                         "'main': main.py's cached full-res/every-frame "
                         "stubs for the 750-frame 121364_0.mp4 clip.")
    args = p.parse_args()

    if args.source == 'fastpipe':
        start = args.start_frame if args.start_frame is not None else DEFAULT_START
        end   = args.end_frame if args.end_frame is not None else DEFAULT_END
    else:
        # 'main': analyse the whole short clip by default — no windowing needed.
        start = args.start_frame if args.start_frame is not None else 0
        end   = args.end_frame if args.end_frame is not None else 10**9  # clamped in load_window_tracks
    return start, end, args.source


def main():
    start_frame, end_frame, source = _parse_args()

    print(f"analyze_formation_window: source='{source}'  "
          f"requested window [{start_frame}:{end_frame})")

    t0 = time.time()
    tracks, video_path, fps, start_frame, end_frame = load_window_tracks(
        start_frame, end_frame, source=source)
    n = end_frame - start_frame
    print(f"Loaded + position-transformed window [{start_frame}:{end_frame}) "
          f"({n} frames) in {time.time() - t0:.2f}s")

    t1 = time.time()
    print(f"Streaming {video_path} for window frames only "
          f"(jersey-colour team assignment)...")
    assign_teams_for_window(tracks, video_path, start_frame, end_frame)
    print(f"Team assignment took {time.time() - t1:.1f}s")

    t2 = time.time()
    pid_to_sid, sid_team, sid_appearance_count = _link_identities(tracks, n)
    valid_sids = {sid for sid, cnt in sid_appearance_count.items()
                  if cnt >= MIN_APPEARANCE_FRAMES}
    print(f"Identity linking took {time.time() - t2:.2f}s  "
          f"({len(pid_to_sid)} raw IDs -> {len(sid_team)} merged identities, "
          f"{len(valid_sids)} kept >= {MIN_APPEARANCE_FRAMES} frames)")

    frames_by_team = {}
    sids_by_team = {}
    for team in (1, 2):
        frames_by_team[team], sids_by_team[team] = build_team_positions(
            tracks, n, pid_to_sid, sid_team, valid_sids, team)
    defends_low_by_team = determine_goal_sides(frames_by_team)

    results = [analyze_team(frames_by_team[team], sids_by_team[team],
                             defends_low_by_team[team], team, fps)
               for team in (1, 2)]

    print("\n" + "=" * 70)
    print(f"FORMATION WINDOW SUMMARY  [{start_frame}:{end_frame})  "
          f"({n} frames, {n / fps:.1f}s @ {fps}fps)")
    windowed_sample = results[0]['windowed']
    print(f"(identity linking: proximity<={PROXIMITY_PX}px, "
          f"lost-buffer<={LOST_BUF_FRAMES} frames; "
          f"row-gap threshold: {ROW_GAP_METRES}m; "
          f"min outfield for a confident read: {MIN_OUTFIELD_FOR_CONFIDENT_READ}; "
          f"rolling window: {WINDOW_SECONDS}s ({windowed_sample['window_frames']} frames), "
          f"step: {STEP_SECONDS}s ({windowed_sample['step_frames']} frames), "
          f"min presence: {MIN_PRESENCE_FRAC*100:.0f}%)")
    print("=" * 70)

    def _print_reads(label, r, unit_label, n_units):
        classified = r[f'classified_{unit_label}']
        noisy = r[f'noisy_{unit_label}']
        print(f"  [{label}] {unit_label} classified (>= {MIN_OUTFIELD_FOR_CONFIDENT_READ} "
              f"outfield players): {classified} / {n_units}   "
              f"(noisy/incomplete: {noisy})")
        if r['modal_formation'] is not None:
            print(f"  [{label}] Modal formation: {r['modal_formation']}  "
                  f"({r['modal_formation_share']*100:.1f}% of classified {unit_label})")
            top5 = r['formation_counts'].most_common(5)
            print(f"  [{label}] Top formation readings: "
                  + ", ".join(f"{f} ({c})" for f, c in top5))
            print(f"  [{label}] Defensive line height (avg distance from own goal line): "
                  f"{r['def_line_avg_m']:.1f}m  "
                  f"(range {r['def_line_min_m']:.1f}m - {r['def_line_max_m']:.1f}m)")
        else:
            print(f"  [{label}] No {unit_label} had a confident read — no formation determined.")

    for r in results:
        goal_side = "x≈0" if r['defends_low_x'] else "x≈105"
        print(f"\nTeam {r['team']}  (defends goal near {goal_side}; "
              f"{r['n_outfield_identities']} outfield identities tracked)")
        if r['goalkeeper_confident']:
            print(f"  Goalkeeper: sid={r['goalkeeper_sid']} (excluded from formation rows)")
        else:
            rejected = r['goalkeeper_candidate_rejected']
            rejected_depth = r['goalkeeper_candidate_rejected_depth_m']
            if rejected is not None:
                print(f"  Goalkeeper: NOT confidently identified — closest candidate (sid={rejected}) "
                      f"averaged {rejected_depth:.1f}m from goal (> {GK_MAX_DEPTH_M}m cutoff), "
                      f"too far to be a real keeper. Likely cause: keeper kit colour doesn't "
                      f"cluster with either team's jersey colour, so the real keeper was probably "
                      f"never linked into this team's identity pool at all. No player excluded — "
                      f"formation/defensive-line stats below may include 0 or (rarely) 1 extra "
                      f"non-keeper identity vs. a true 10-outfield read.")
            else:
                print("  Goalkeeper: NOT identified (no tracked identities for this team).")

        print()
        _print_reads("per-frame (old)", r['perframe'], 'frames', n)
        print()
        n_windows = r['windowed']['classified_windows'] + r['windowed']['noisy_windows']
        _print_reads("windowed (new)", r['windowed'], 'windows', n_windows)

        old_share = r['perframe']['modal_formation_share']
        new_share = r['windowed']['modal_formation_share']
        delta = (new_share - old_share) * 100
        print(f"  -> modal-share change: {old_share*100:.1f}% -> {new_share*100:.1f}%  "
              f"({'+' if delta >= 0 else ''}{delta:.1f} points)")
    print("=" * 70)


if __name__ == '__main__':
    main()
