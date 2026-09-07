"""
sample_segments.py — random validated test-window sampler for batch_validate.py.

Cuts 30-60s windows of LIVE GAMEPLAY out of full-length match videos, screens
each candidate against several cheap reject-and-reroll gates, and extracts
the survivors with ffmpeg into dev_clips/ and holdout_clips/ for the batch
harness. Source videos are split BY VIDEO into DEV / HOLDOUT sets (never by
window) so results stay comparable across runs of the pipeline.

Reuses existing model/logic instead of rebuilding it:
  - ultralytics YOLO('models/best.pt')            — same player/gk/ref/ball
    detector fast_setup.py and trackers/tracker.py use, for the main-camera
    and live-play checks (rules 0b/2).
  - team_assigner.TeamAssigner.get_player_color    — same jersey-colour
    sampler used everywhere else in this repo, for the "two team colours"
    part of the live-play heuristic.
  - camera_movement_estimator.CameraMovementEstimator.estimate_frame_movement
    — same optical-flow math fast_setup.py's own camera_movement() calls,
    for the motion-magnitude strata feature.

VALIDATION ORDER (cheap first): the time-range guard is enforced by
CONSTRUCTION (windows are only ever drawn from already-live sub-ranges, see
valid_subranges/sample_window_start) rather than drawn-then-rejected, so it
never wastes an attempt. Remaining gates run cheapest-first: content check
(0b + 2 combined, 3-5 YOLO frames) before the scene-cut scan (1, which has
to decode most of the window's frames to compute frame-to-frame histogram
diffs) — content failures are both more common and far cheaper to detect.

Usage:
    python sample_segments.py [--seed 42] [--resample]
                               [--source-dir "Match_videos"]
                               [--dev-out dev_clips] [--holdout-out holdout_clips]
                               [--manifest sample_segments_manifest.json]
                               [--probe-only]
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time as time_mod

import cv2
import numpy as np

from team_assigner import TeamAssigner
from camera_movement_estimator import CameraMovementEstimator

# Force line-buffered stdout regardless of invocation (python -u or not) --
# each source's probe/window-sampling result is one YOLO-heavy print that
# can be minutes apart on this CPU-only machine; without this, block
# buffering under a piped/backgrounded process can hide all progress until
# exit, which looks identical to a hang and wastes time chasing a
# non-existent bug (confirmed while building this script).
sys.stdout.reconfigure(line_buffering=True)

# ─────────────────────────────────────────────────────────────────────────
# Source / split configuration
# ─────────────────────────────────────────────────────────────────────────
SOURCE_DIR = os.path.join('Match_videos')
PLAYER_MODEL = 'models/best.pt'

# 121364_0.mp4 / LiverpoolPSG_short.mp4 are already-tuned-on clips (used
# throughout prior validation sessions) — excluded entirely so this batch
# measures generalization, not re-tests clips the pipeline was checked
# against before.
EXCLUDE_VIDEOS = {'121364_0.mp4', 'LiverpoolPSG_short.mp4'}

# NOTE on filenames: the user-specified "AlgeriaArgentina.mp4" does not
# exist on disk — the actual file is "AlgeriaArgentia.mp4" (missing the
# second 'n'). Used as-is below since it's an unambiguous near-match and
# the file's mtime shows it was added the same day as this request, but
# flagged here (and in the run report) in case that was a typo for a
# differently-named file rather than this one.
DEV_VIDEOS = {
    'BarcaMadridPT1.mp4': 13,
    'ChelseaCity.mp4': 13,
    'LiverpoolMadrid.mp4': 13,
}
HOLDOUT_VIDEOS = {
    'MarseilevsLyonFull__trim_no_halftime.mp4': 10,
    'AlgeriaArgentia.mp4': 10,
}

WINDOW_MIN_SEC = 30.0
WINDOW_MAX_SEC = 60.0
MAX_ATTEMPTS_PER_WINDOW = 15

DEFAULT_SEED = 42

# ── Time-range guard (rule 0a) ──────────────────────────────────────────
EDGE_EXCLUDE_FRAC = 0.05          # skip first/last 5% of each source
HALFTIME_PROBE_LO_FRAC = 0.15     # only scan for a halftime gap in the
HALFTIME_PROBE_HI_FRAC = 0.85     # middle 70% of the timeline
HALFTIME_PROBE_POINTS = 16        # coarse probe points across that span --
                                   # this machine's YOLO inference is CPU-only
                                   # (no CUDA, confirmed), ~1-6s/call, so probe
                                   # density is deliberately kept just dense
                                   # enough to localize a >=15min real gap out
                                   # of a ~70%-of-duration span, not exhaustive
HALFTIME_MIN_GAP_FRAC = 0.02      # a "real" halftime run must span at
                                   # least 2% of total duration to count
                                   # (filters out one flaky probe frame)
HALFTIME_PAD_SEC = 30.0           # pad the detected gap a bit on both ends

# ── Rule 2: not-main-camera (tactical wide shot vs close-up/replay) ────
MAIN_CAMERA_MIN_PLAYERS = 10
MAIN_CAMERA_MAX_MEAN_AREA_FRAC = 0.02   # close-ups have much bigger boxes

# ── Rule 0b: live-play heuristic (two teams + spread out) ──────────────
LIVE_PLAY_MIN_COLOR_SEPARATION = 40.0   # BGR distance between the 2 rough
                                         # per-frame colour clusters
LIVE_PLAY_MIN_CLUSTER_BALANCE = 0.25    # each cluster must hold >=25% of
                                         # samples (rules out 19-1 splits)
LIVE_PLAY_MIN_X_SPREAD_FRAC = 0.15      # std(box-center x) / frame width;
                                         # huddles/warmup clusters are tight

TWO_CLUSTER_MIN_VARIANCE_EXPLAINED = 0.65
# Gates whether a forced k=2 KMeans split on player colours is trustworthy
# at all before applying LIVE_PLAY_MIN_COLOR_SEPARATION/CLUSTER_BALANCE to
# it. "Variance explained" = 1 - (k=2 inertia / k=1 inertia), the same
# elbow-method quantity used to pick k in unsupervised clustering: how much
# of the total colour spread the 2-way split actually accounts for versus
# just using one shared mean. When both kits read close to neutral/
# grass-adjacent (seen on AlgeriaArgentia.mp4), KMeans still returns *a*
# split, but it's fitting noise, not two teams — and cluster_balance on that
# noisy split is not a meaningful "two teams present" signal.
#
# Threshold derived from real data, not guessed: sampled 20 AlgeriaArgentia
# frames that failed the live-play colour check where color_sep alone was
# also weak (color_sep-only or color_sep+cluster_balance failures) scored
# variance_explained 0.41-0.63 (median 0.60). The SAME check on 30 accepted
# BarcaMadridPT1/ChelseaCity frames (real high-contrast kits, genuine team
# structure) scored 0.79-0.94 (min 0.788) -- zero overlap. 0.65 sits in the
# gap with a >=0.13 margin below the observed genuine-bimodal floor.
#
# Important scope note: this does NOT cover every AlgeriaArgentia rejection.
# A separate subset of failures had HIGH variance_explained (0.6-0.89,
# overlapping the genuine-team range) but still failed cluster_balance
# alone (e.g. 0.05-0.24) -- KMeans was isolating a small, genuinely
# distinct-coloured minority (most likely the goalkeeper's kit, the one
# player class included in `boxes` besides 'player' that's expected to
# differ from both teams) rather than fitting noise. That is a real,
# well-separated split; it is correctly still rejected by cluster_balance,
# not a false positive of this gate.

# ── Rule 1: scene cut ───────────────────────────────────────────────────
CUT_SAMPLE_STRIDE_FRAMES = 3      # histogram-compare every 3rd decoded frame
CUT_HIST_SCALE = 0.25             # downscale before histogram (speed)
CUT_SPIKE_FACTOR = 6.0            # per spec: a real cut spikes 5-10x baseline
CUT_MIN_ABS_DIFF = 0.12           # floor so a near-zero baseline (very
                                   # static shot) doesn't false-positive

# ── Stratified sampling (density x camera-motion) ───────────────────────
CROWDED_MIN_PLAYERS = 16.0
HIGH_MOTION_MIN_PX = 5.0
STRATA = ['crowded_high_motion', 'crowded_low_motion',
          'sparse_high_motion', 'sparse_low_motion']

MOTION_PROBE_GAP_SEC = 0.4        # spacing of the close frame pair used
                                   # for the camera-motion / player-speed
                                   # proxy features
MOTION_MATCH_MAX_PX = 250.0       # nearest-neighbour cap for the cheap
                                   # frame-to-frame player-speed proxy


# ─────────────────────────────────────────────────────────────────────────
# ffprobe helpers
# ─────────────────────────────────────────────────────────────────────────
def _ffprobe_field(video_path, stream_spec, entry):
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', stream_spec,
         '-show_entries', f'stream={entry}', '-of',
         'default=noprint_wrappers=1:nokey=1', video_path],
        capture_output=True, text=True, check=True).stdout.strip()
    return out


def probe_video_meta(video_path):
    """Returns (duration_sec, fps, width, height). Uses ffprobe (not
    cv2.CAP_PROP_FRAME_COUNT/FPS) for duration/fps — more reliable across
    the variable-frame-rate broadcast sources in this batch than trusting
    a container's frame-count metadata."""
    dur_out = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
        capture_output=True, text=True, check=True).stdout.strip()
    duration = float(dur_out)

    rate_str = _ffprobe_field(video_path, 'v:0', 'r_frame_rate')
    num, den = rate_str.split('/')
    fps = float(num) / float(den)

    w = int(_ffprobe_field(video_path, 'v:0', 'width'))
    h = int(_ffprobe_field(video_path, 'v:0', 'height'))
    return duration, fps, w, h


def has_audio_stream(video_path):
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'a',
         '-show_entries', 'stream=index', '-of',
         'default=noprint_wrappers=1:nokey=1', video_path],
        capture_output=True, text=True, check=True).stdout.strip()
    return bool(out)


# ─────────────────────────────────────────────────────────────────────────
# Frame access
# ─────────────────────────────────────────────────────────────────────────
def grab_frame_at(video_path, t_sec):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def grab_frames_at(video_path, times_sec):
    """Grabs several timestamps from ONE VideoCapture. Sorted internally so
    each seek only ever needs to move forward (cheaper than re-opening per
    timestamp), results returned in the original input order."""
    order = sorted(range(len(times_sec)), key=lambda i: times_sec[i])
    frames = [None] * len(times_sec)
    cap = cv2.VideoCapture(video_path)
    for i in order:
        cap.set(cv2.CAP_PROP_POS_MSEC, times_sec[i] * 1000.0)
        ret, frame = cap.read()
        frames[i] = frame if ret else None
    cap.release()
    return frames


# ─────────────────────────────────────────────────────────────────────────
# YOLO-based per-frame analysis (shared by rules 0b, 2, and the halftime
# probe)
# ─────────────────────────────────────────────────────────────────────────
_ta_singleton = None


def _get_team_assigner():
    global _ta_singleton
    if _ta_singleton is None:
        _ta_singleton = TeamAssigner()
    return _ta_singleton


def analyze_frame(frame, model, scale=0.5):
    """One YOLO pass + cheap colour/geometry checks. Returns a dict with
    everything rules 0b/2 and the halftime probe need, computed once."""
    empty = {
        'n_players': 0, 'mean_area_frac': 0.0, 'x_spread_frac': 0.0,
        'color_separation': 0.0, 'cluster_balance': 0.0,
        'main_camera': False, 'live_play': False, 'boxes': [],
    }
    if frame is None:
        return empty

    h, w = frame.shape[:2]
    small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    result = model.predict(small, conf=0.25, verbose=False)[0]
    names = result.names

    boxes = []
    for box, cls_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
        if names[int(cls_id)] in ('player', 'goalkeeper'):
            boxes.append((box / scale).tolist())

    n_players = len(boxes)
    if n_players == 0:
        return empty

    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    mean_area_frac = float(np.mean(areas)) / (w * h)
    xs = [(b[0] + b[2]) / 2.0 for b in boxes]
    x_spread_frac = float(np.std(xs)) / w

    main_camera = (n_players >= MAIN_CAMERA_MIN_PLAYERS
                   and mean_area_frac <= MAIN_CAMERA_MAX_MEAN_AREA_FRAC)

    color_sep, cluster_balance = 0.0, 0.0
    two_cluster_justified = False
    if n_players >= 4:
        ta = _get_team_assigner()
        colors = np.array([ta.get_player_color(frame, b) for b in boxes], dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        inertia_k2, labels, centers = cv2.kmeans(colors, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.ravel(), minlength=2)
        color_sep = float(np.linalg.norm(centers[0] - centers[1]))
        cluster_balance = float(counts.min()) / float(counts.sum())

        inertia_k1 = float(np.sum((colors - colors.mean(axis=0)) ** 2))
        variance_explained = 1.0 - (float(inertia_k2) / inertia_k1) if inertia_k1 > 1e-6 else 0.0
        two_cluster_justified = variance_explained >= TWO_CLUSTER_MIN_VARIANCE_EXPLAINED

    if two_cluster_justified:
        color_ok = (color_sep >= LIVE_PLAY_MIN_COLOR_SEPARATION
                    and cluster_balance >= LIVE_PLAY_MIN_CLUSTER_BALANCE)
    else:
        # The forced k=2 split isn't statistically justified (see
        # TWO_CLUSTER_MIN_VARIANCE_EXPLAINED above) -- color_sep/
        # cluster_balance from it aren't a meaningful "two teams present"
        # signal either way, so don't penalize live play on them. Fall back
        # to the player-distribution check alone.
        color_ok = True

    live_play = (main_camera
                 and color_ok
                 and x_spread_frac >= LIVE_PLAY_MIN_X_SPREAD_FRAC)

    return {
        'n_players': n_players, 'mean_area_frac': mean_area_frac,
        'x_spread_frac': x_spread_frac, 'color_separation': color_sep,
        'cluster_balance': cluster_balance, 'main_camera': main_camera,
        'live_play': live_play, 'boxes': boxes,
    }


# ─────────────────────────────────────────────────────────────────────────
# Rule 0a: time-range guard + halftime probe
# ─────────────────────────────────────────────────────────────────────────
def merge_intervals(intervals):
    if not intervals:
        return []
    ivs = sorted(intervals)
    merged = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def valid_subranges(duration, excluded):
    merged = merge_intervals(excluded)
    subranges = []
    prev_end = 0.0
    for s, e in merged:
        if s > prev_end:
            subranges.append((prev_end, s))
        prev_end = max(prev_end, e)
    if prev_end < duration:
        subranges.append((prev_end, duration))
    return subranges


def probe_source(video_path, model, verbose_prefix=''):
    """Probes duration/fps/resolution and scans the middle of the timeline
    for a halftime-shaped gap (a sustained run of NOT-live-play probe
    points). Returns (duration, fps, (w,h), excluded_ranges, probe_log)."""
    duration, fps, w, h = probe_video_meta(video_path)
    edge = duration * EDGE_EXCLUDE_FRAC
    excluded = [(0.0, edge), (duration - edge, duration)]

    lo = duration * HALFTIME_PROBE_LO_FRAC
    hi = duration * HALFTIME_PROBE_HI_FRAC
    probe_times = np.linspace(lo, hi, HALFTIME_PROBE_POINTS)
    frames = grab_frames_at(video_path, probe_times.tolist())

    probe_log = []
    live_flags = []
    for t, frame in zip(probe_times, frames):
        feats = analyze_frame(frame, model)
        live_flags.append(feats['live_play'])
        probe_log.append({'t_sec': round(float(t), 1), 'live_play': feats['live_play'],
                          'n_players': feats['n_players']})

    # Find the longest contiguous non-live run among the probe points.
    best_run = None
    run_start = None
    for i, live in enumerate(live_flags):
        if not live:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run = (run_start, i - 1)
                if best_run is None or (run[1] - run[0]) > (best_run[1] - best_run[0]):
                    best_run = run
                run_start = None
    if run_start is not None:
        run = (run_start, len(live_flags) - 1)
        if best_run is None or (run[1] - run[0]) > (best_run[1] - best_run[0]):
            best_run = run

    halftime_range = None
    if best_run is not None:
        t_start = float(probe_times[best_run[0]])
        t_end = float(probe_times[best_run[1]])
        if (t_end - t_start) >= duration * HALFTIME_MIN_GAP_FRAC:
            halftime_range = (max(0.0, t_start - HALFTIME_PAD_SEC),
                              min(duration, t_end + HALFTIME_PAD_SEC))
            excluded.append(halftime_range)

    print(f"{verbose_prefix}duration={duration:.1f}s fps={fps:.2f} res={w}x{h}  "
          f"halftime_gap={'None (verified clean)' if halftime_range is None else f'{halftime_range[0]:.0f}s-{halftime_range[1]:.0f}s'}")

    return duration, fps, (w, h), excluded, probe_log


def sample_window_start(subranges, win_dur, rng):
    usable = [(s, e) for s, e in subranges if e - s >= win_dur]
    if not usable:
        return None
    weights = [e - s for s, e in usable]
    total = sum(weights)
    r = rng.uniform(0, total)
    acc = 0.0
    for (s, e), w in zip(usable, weights):
        acc += w
        if r <= acc:
            return rng.uniform(s, e - win_dur)
    return usable[-1][0]


# ─────────────────────────────────────────────────────────────────────────
# Rule 1: scene-cut detection
# ─────────────────────────────────────────────────────────────────────────
def detect_scene_cut(video_path, start_sec, end_sec, fps):
    """Streams the window once (grab() to skip, retrieve()/read() only on
    sampled frames — cheaper than fully decoding every frame), computes a
    downscaled grayscale-histogram Bhattacharyya distance between
    consecutive SAMPLED frames, and flags a cut if any diff spikes
    CUT_SPIKE_FACTOR above the window's own median diff."""
    cap = cv2.VideoCapture(video_path)
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    diffs = []
    prev_hist = None
    fn = start_frame
    while fn < end_frame:
        take = (fn - start_frame) % CUT_SAMPLE_STRIDE_FRAMES == 0
        if take:
            ret, frame = cap.read()
        else:
            ret = cap.grab()
            frame = None
        if not ret:
            break
        if take and frame is not None:
            small = cv2.resize(frame, None, fx=CUT_HIST_SCALE, fy=CUT_HIST_SCALE)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            cv2.normalize(hist, hist)
            if prev_hist is not None:
                diffs.append(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            prev_hist = hist
        fn += 1
    cap.release()

    if len(diffs) < 5:
        return False, 0.0, 0.0   # window too short to sample meaningfully; don't block on it
    baseline = float(np.median(diffs))
    max_diff = float(np.max(diffs))
    spike_ratio = max_diff / max(baseline, 1e-4)
    has_cut = spike_ratio >= CUT_SPIKE_FACTOR and max_diff > CUT_MIN_ABS_DIFF
    return has_cut, baseline, max_diff


# ─────────────────────────────────────────────────────────────────────────
# Strata features: camera motion + a cheap player-speed proxy
# ─────────────────────────────────────────────────────────────────────────
def window_motion_features(video_path, mid_sec, model):
    """Two close frames (MOTION_PROBE_GAP_SEC apart) near the window's
    midpoint: camera-motion magnitude via the SAME CameraMovementEstimator
    math fast_setup.py uses, plus a cheap nearest-neighbour frame-to-frame
    player-displacement proxy (NOT real tracking/identity — only used for
    coarse strata bucketing and manifest bookkeeping, not the harness's own
    speed metric)."""
    t0, t1 = mid_sec, mid_sec + MOTION_PROBE_GAP_SEC
    f0, f1 = grab_frames_at(video_path, [t0, t1])
    if f0 is None or f1 is None:
        return 0.0, 0.0

    estimator = CameraMovementEstimator(f0)
    gray0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    features0 = cv2.goodFeaturesToTrack(gray0, **estimator.features)
    if features0 is None:
        cam_mag = 0.0
    else:
        _, _, cam_mag, _ = estimator.estimate_frame_movement(gray0, features0, gray1)

    feats0 = analyze_frame(f0, model)
    feats1 = analyze_frame(f1, model)
    centers0 = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in feats0['boxes']]
    centers1 = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in feats1['boxes']]
    disps = []
    remaining = list(centers1)
    for pa in centers0:
        if not remaining:
            break
        dists = [math.hypot(pa[0] - pb[0], pa[1] - pb[1]) for pb in remaining]
        j = int(np.argmin(dists))
        if dists[j] < MOTION_MATCH_MAX_PX:
            disps.append(dists[j])
        remaining.pop(j)
    player_speed_px_s = (float(np.mean(disps)) / MOTION_PROBE_GAP_SEC) if disps else 0.0

    return float(cam_mag), player_speed_px_s


# ─────────────────────────────────────────────────────────────────────────
# Window validation (fail-fast, cheapest gate first)
# ─────────────────────────────────────────────────────────────────────────
def validate_window(video_path, start_sec, dur_sec, fps, model):
    """Returns (ok, reason, stats). reason is one of:
    'not_live_play', 'scene_cut', None (== passed)."""
    end_sec = start_sec + dur_sec
    sample_times = np.linspace(start_sec + 1.0, end_sec - 1.0, 5).tolist()
    frames = grab_frames_at(video_path, sample_times)
    frame_feats = [analyze_frame(f, model) for f in frames]

    n_live = sum(1 for feats in frame_feats if feats['live_play'])
    n_main_camera = sum(1 for feats in frame_feats if feats['main_camera'])
    mean_n_players = float(np.mean([f['n_players'] for f in frame_feats]))

    content_stats = {
        'n_sampled_frames': len(frame_feats),
        'n_live_play_frames': n_live,
        'n_main_camera_frames': n_main_camera,
        'mean_n_players': round(mean_n_players, 2),
    }

    # Require a clear majority of sampled frames to read as live tactical
    # play -- a single flaky frame (occlusion, graphic overlay) shouldn't
    # sink an otherwise-good window, but most of them must agree.
    if n_live < 3:
        return False, 'not_live_play', content_stats

    has_cut, baseline, max_diff = detect_scene_cut(video_path, start_sec, end_sec, fps)
    content_stats['cut_baseline'] = round(baseline, 4)
    content_stats['cut_max_diff'] = round(max_diff, 4)
    if has_cut:
        return False, 'scene_cut', content_stats

    cam_mag, player_speed = window_motion_features(video_path, start_sec + dur_sec / 2.0, model)
    content_stats['mean_camera_motion_px'] = round(cam_mag, 2)
    content_stats['mean_player_speed_px_s'] = round(player_speed, 1)

    return True, None, content_stats


def classify_stratum(stats):
    density = 'crowded' if stats['mean_n_players'] >= CROWDED_MIN_PLAYERS else 'sparse'
    motion = 'high_motion' if stats['mean_camera_motion_px'] >= HIGH_MOTION_MIN_PX else 'low_motion'
    return f'{density}_{motion}'


# ─────────────────────────────────────────────────────────────────────────
# Per-video window selection
# ─────────────────────────────────────────────────────────────────────────
def sample_windows_for_video(video_path, video_name, n_target, duration, excluded,
                              fps, model, rng, rejections_log):
    subranges = valid_subranges(duration, excluded)
    strata_cycle = [STRATA[i % len(STRATA)] for i in range(n_target)]
    rng.shuffle(strata_cycle)

    selected = []
    for slot_idx in range(n_target):
        desired_stratum = strata_cycle[slot_idx]
        fallback = None
        found = None
        for attempt in range(1, MAX_ATTEMPTS_PER_WINDOW + 1):
            win_dur = rng.uniform(WINDOW_MIN_SEC, WINDOW_MAX_SEC)
            start = sample_window_start(subranges, win_dur, rng)
            if start is None:
                rejections_log.append({'video': video_name, 'slot': slot_idx, 'attempt': attempt,
                                       'start_sec': None, 'reason': 'no_valid_subrange'})
                break

            ok, reason, stats = validate_window(video_path, start, win_dur, fps, model)
            entry = {'video': video_name, 'slot': slot_idx, 'attempt': attempt,
                     'start_sec': round(start, 1), 'duration_sec': round(win_dur, 1),
                     'reason': reason, 'stats': stats}
            rejections_log.append(entry)
            if not ok:
                continue

            stratum = classify_stratum(stats)
            candidate = {'start_sec': start, 'duration_sec': win_dur, 'stats': stats,
                        'stratum': stratum, 'attempt': attempt}
            if stratum == desired_stratum:
                found = candidate
                break
            elif fallback is None:
                fallback = candidate
            # keep trying a few more times for the desired stratum, but
            # don't burn the whole budget chasing it once we have SOME
            # valid candidate in hand
            if fallback is not None and attempt >= MAX_ATTEMPTS_PER_WINDOW - 3:
                found = fallback
                break

        if found is None:
            found = fallback
        if found is None:
            print(f"    no clean window found for {video_name} slot {slot_idx}")
            continue
        selected.append(found)
    return selected


# ─────────────────────────────────────────────────────────────────────────
# ffmpeg extraction
# ─────────────────────────────────────────────────────────────────────────
def extract_clip(video_path, start_sec, dur_sec, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = ['ffmpeg', '-y', '-accurate_seek', '-ss', f'{start_sec:.3f}',
           '-i', video_path, '-t', f'{dur_sec:.3f}',
           '-map', '0:v:0']
    if has_audio_stream(video_path):
        cmd += ['-map', '0:a:0?']
        cmd += ['-c:a', 'aac']
    cmd += ['-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
            '-avoid_negative_ts', 'make_zero', out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}:\n{result.stderr[-2000:]}")


# ─────────────────────────────────────────────────────────────────────────
# Manifest
# ─────────────────────────────────────────────────────────────────────────
def load_manifest(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def save_manifest(path, manifest):
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--resample', action='store_true',
                        help='Ignore an existing manifest and regenerate all windows fresh')
    parser.add_argument('--source-dir', default=SOURCE_DIR)
    parser.add_argument('--dev-out', default='dev_clips')
    parser.add_argument('--holdout-out', default='holdout_clips')
    parser.add_argument('--manifest', default='sample_segments_manifest.json')
    parser.add_argument('--probe-only', action='store_true',
                        help='Only probe sources (duration/fps/excluded ranges) and exit -- no sampling/extraction')
    args = parser.parse_args()

    rng = random.Random(args.seed)

    all_videos = {}
    all_videos.update({k: ('dev', v) for k, v in DEV_VIDEOS.items()})
    all_videos.update({k: ('holdout', v) for k, v in HOLDOUT_VIDEOS.items()})

    missing = [v for v in all_videos if not os.path.exists(os.path.join(args.source_dir, v))]
    if missing:
        print(f"ERROR: missing source video(s): {missing}")
        print(f"Files present in {args.source_dir}: {sorted(os.listdir(args.source_dir))}")
        return

    # Load any existing manifest FIRST (before touching the model/probing) so
    # already-probed sources can be skipped -- probing is itself expensive on
    # this CPU-only machine (observed 63-123s per source for a 16-point
    # halftime scan), and re-probing on every invocation (including
    # --probe-only re-runs) would silently throw that work away.
    manifest = None if args.resample else load_manifest(args.manifest)
    if manifest is not None:
        print(f"Loaded existing manifest ({args.manifest}) — pass --resample to regenerate everything.")
    source_meta = dict(manifest['sources']) if manifest else {}
    windows = manifest['windows'] if manifest else []
    rejections = manifest['rejections'] if manifest else []

    videos_needing_probe = [v for v in all_videos if v not in source_meta]
    # The model is also needed for window sampling (content/live-play checks),
    # which runs for every video not already fully represented in the
    # manifest -- not just ones needing a fresh probe. Only truly skip
    # loading it if this is a --probe-only invocation where every source is
    # already probed (nothing at all left to do).
    windows_by_video_precheck = {}
    for w in (manifest['windows'] if manifest else []):
        windows_by_video_precheck.setdefault(w['source'], []).append(w)
    videos_needing_sampling = [v for v, (_, n) in all_videos.items()
                               if len(windows_by_video_precheck.get(v, [])) < n]

    model = None
    if videos_needing_probe or (not args.probe_only and videos_needing_sampling):
        print(f"Loading YOLO model ({PLAYER_MODEL}) ...")
        from ultralytics import YOLO
        model = YOLO(PLAYER_MODEL)
        print("  model loaded.")

    print(f"\n{'=' * 70}\nPROBING SOURCES\n{'=' * 70}")
    for video_name in all_videos:
        if video_name not in videos_needing_probe:
            meta = source_meta[video_name]
            print(f"  {video_name}: reusing probe from manifest — duration={meta['duration_sec']}s "
                 f"excluded={meta['excluded_ranges_sec']}")
            continue
        video_path = os.path.join(args.source_dir, video_name)
        print(f"  {video_name}: probing ({HALFTIME_PROBE_POINTS} points) ...")
        t0 = time_mod.time()
        duration, fps, (w, h), excluded, probe_log = probe_source(
            video_path, model, verbose_prefix=f"  {video_name}: ")
        print(f"    ({time_mod.time() - t0:.1f}s to probe)")
        source_meta[video_name] = {
            'duration_sec': round(duration, 1), 'fps': round(fps, 3),
            'resolution': f'{w}x{h}',
            'excluded_ranges_sec': [[round(s, 1), round(e, 1)] for s, e in excluded],
            'halftime_probe_log': probe_log,
        }
        # Save after EVERY source, not just at the very end -- so a
        # --probe-only run (or one interrupted partway through) still
        # persists whatever probing has completed so far, rather than
        # only existing in that run's console output.
        save_manifest(args.manifest, {
            'seed': args.seed, 'generated_at': time_mod.strftime('%Y-%m-%dT%H:%M:%S'),
            'source_dir': args.source_dir, 'sources': source_meta,
            'dev_videos': DEV_VIDEOS, 'holdout_videos': HOLDOUT_VIDEOS,
            'windows': windows, 'rejections': rejections,
        })

    if args.probe_only:
        print(f"\n--probe-only: stopping before sampling/extraction. Saved probe results to {args.manifest}.")
        for name, meta in source_meta.items():
            print(f"  {name}: duration={meta['duration_sec']}s excluded={meta['excluded_ranges_sec']}")
        return
    windows_by_video = {}
    for w in windows:
        windows_by_video.setdefault(w['source'], []).append(w)

    def _checkpoint():
        windows_now = [w for entries in windows_by_video.values() for w in entries]
        save_manifest(args.manifest, {
            'seed': args.seed, 'generated_at': time_mod.strftime('%Y-%m-%dT%H:%M:%S'),
            'source_dir': args.source_dir, 'sources': source_meta,
            'dev_videos': DEV_VIDEOS, 'holdout_videos': HOLDOUT_VIDEOS,
            'windows': windows_now, 'rejections': rejections,
        })
        return windows_now

    print(f"\n{'=' * 70}\nSAMPLING + EXTRACTING (seed={args.seed})\n{'=' * 70}")
    for video_name, (set_name, n_target) in all_videos.items():
        out_dir = args.dev_out if set_name == 'dev' else args.holdout_out

        if video_name in windows_by_video and len(windows_by_video[video_name]) >= n_target:
            print(f"  {video_name}: reusing {len(windows_by_video[video_name])} windows from manifest")
        else:
            video_path = os.path.join(args.source_dir, video_name)
            meta = source_meta[video_name]
            excluded = [tuple(r) for r in meta['excluded_ranges_sec']]
            print(f"  {video_name} ({set_name}, target={n_target}) ...")
            t0 = time_mod.time()
            selected = sample_windows_for_video(
                video_path, video_name, n_target, meta['duration_sec'], excluded,
                meta['fps'], model, rng, rejections)
            elapsed = time_mod.time() - t0

            new_entries = []
            for w in selected:
                start_int = int(round(w['start_sec']))
                stem = os.path.splitext(video_name)[0]
                out_path = os.path.join(out_dir, f'{stem}_w{start_int}.mp4')
                new_entries.append({
                    'source': video_name, 'set': set_name,
                    'start_sec': round(w['start_sec'], 2),
                    'end_sec': round(w['start_sec'] + w['duration_sec'], 2),
                    'duration_sec': round(w['duration_sec'], 2),
                    'output_path': out_path, 'attempt': w['attempt'],
                    'stratum': w['stratum'], 'stats': w['stats'],
                })
            windows_by_video[video_name] = new_entries
            n_attempts = sum(1 for r in rejections if r['video'] == video_name)
            print(f"    found {len(selected)}/{n_target} windows in {n_attempts} attempt(s), {elapsed:.1f}s")

            # Checkpoint immediately after EACH video's sampling -- this is a
            # multi-hour run across 5 large sources; without this, killing or
            # crashing partway through would lose every video sampled so far,
            # not just the one in progress.
            _checkpoint()
            print(f"    checkpointed to {args.manifest}")

        # Extract this video's clips right away rather than waiting for all
        # 5 videos to finish sampling -- so dev_clips/ becomes usable as soon
        # as the 3 dev sources are done, without waiting on holdout too, and
        # a later interruption doesn't leave sampled-but-unextracted windows.
        for w in windows_by_video.get(video_name, []):
            if os.path.exists(w['output_path']):
                continue
            print(f"    extracting {w['output_path']}  ({w['start_sec']:.1f}s - "
                  f"{w['end_sec']:.1f}s, {w['stratum']})")
            extract_clip(os.path.join(args.source_dir, w['source']), w['start_sec'],
                         w['duration_sec'], w['output_path'])

    windows = _checkpoint()
    print(f"\nSaved manifest: {args.manifest}")

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for video_name, (set_name, n_target) in all_videos.items():
        entries = windows_by_video.get(video_name, [])
        video_rejections = [r for r in rejections if r['video'] == video_name]
        reasons = {}
        for r in video_rejections:
            if r['reason']:
                reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
        print(f"  {video_name} [{set_name}]: {len(entries)}/{n_target} windows, "
              f"{len(video_rejections)} attempts, rejections={reasons}")
    print(f"\nTotal windows: {len(windows)}  "
          f"(dev={sum(1 for w in windows if w['set']=='dev')}, "
          f"holdout={sum(1 for w in windows if w['set']=='holdout')})")


if __name__ == '__main__':
    main()
