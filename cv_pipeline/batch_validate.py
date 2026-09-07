"""
batch_validate.py — batch validation harness for the data-level pipeline.

Runs every video in a folder through fast_setup.py's detect/camera/calibrate
stages plus the same data-level enrichment main.py does (position, camera
adjustment, view transform, calibration confidence, ball interpolation,
team resolution), then scores each clip against FIXED thresholds and writes
a per-clip scorecard + a cross-clip summary table.

Deliberately reuses existing pipeline/diagnostic code instead of
reimplementing it:
  - fast_setup.detect_and_track / camera_movement / calibrate  (the actual
    streaming, memory-safe detection/camera/calibration stages)
  - view_transformer.ViewTransformer.compute_frame_confidence  (in-bounds
    calibration-confidence metric, unchanged)
  - team_assigner.TeamAssigner.resolve_all_teams / finalize_fallback_
    assignments / merge_fragmented_tracks / apply_final_teams  (the exact
    two-pass team-resolution pipeline main.py runs)
  - analyze_danger_score.py's UNCERTAIN_GAP_MAX_FRAMES / find_uncertain_gaps
    convention for ball-detection gap-length (constant + algorithm copied
    below — the source function isn't independently importable without
    pulling in that script's much heavier render_output3/tactical_events
    import chain)
  - detect_id_swaps.py's STEP1/STEP2 algorithm (best_split + continuity
    check) — copied below, not imported, because that file is a bare
    argparse script with no importable function/return value. Reuses the
    per-player colour samples this run's TeamAssigner.resolve_all_teams
    already gathered (ta.pending_samples + the new ta.color_vote_frames —
    see team_assigner_color_bak.py) instead of re-fitting a second KMeans
    AND re-sampling every player's colour a second time across the whole
    clip, which is what the standalone script does and what an earlier
    version of this function also did — found to be a literal duplicate
    full-video pass while diagnosing a slow validation run.
  - compare_gray_frames.py's total_and_gray — copied below for the same
    reason (script-scope code, not a function it exports).
  - trackers.Tracker.add_position_to_tracks / interpolate_ball_positions —
    copied below rather than imported, because using the real methods
    would require constructing a full Tracker() (which loads the YOLO
    model) just to reach two model-independent helper methods.

Thresholds below are FIXED per the validation spec — this harness measures
today's system, it does not get tuned by what it finds. Camera-motion
"3+ simultaneous >25px jumps" is interpreted as a run of >=3 CONSECUTIVE
frames each individually over 25px (a sustained multi-frame burst) rather
than an isolated single-frame optical-flow spike — render_output6.py's own
comments document isolated single-frame spikes (up to ~72px observed) as
expected optical-flow noise that gets smoothed away, not a pipeline defect,
so the fail condition here targets sustained bursts instead.

Usage:
    python batch_validate.py <video_folder> [--max-frames N] [--out-dir DIR]
                              [--force] [--no-render] [--render-all]

DIAGNOSIS (2026-07-25) — team resolution was ~200x main.py's cost on the
same clip (1130s mean vs main.py's ~5.6s), confirmed via a profiled 200-frame
run: TeamAssigner.resolve_all_teams itself is fast (5.6s — almost purely
sequential frame access). TeamAssigner.merge_fragmented_tracks accounted for
210.1s of a 215.7s team-resolution total: 1,295 seeks at ~163ms average
(vs 5.4ms for a sequential continuation) — merge_fragmented_tracks's
_avg_color samples each MERGE CANDIDATE's full track-lifetime range at
stride 3, twice per candidate, in whatever order candidates were found (not
frame order), which is a genuinely seek-heavy access pattern on a
compressed mp4. main.py never hits this because it holds the whole clip as
a plain Python list up front — every access after that, sequential or not,
is O(1) list indexing, no video decode at all. Fix: for a --max-frames-
bounded window (this harness never processes the raw unbounded source),
decode the window once into a plain in-memory list for team resolution —
see _load_frames_for_team_resolution — reproducing main.py's cost profile
exactly because it does exactly what main.py does. get_player_color() is a
pure function of (frame pixels, bbox), so this changes nothing about WHAT
gets computed, only how the frame gets fetched.
"""

import argparse
import csv
import glob
import json
import math
import os
import pickle
import sys
import time
import traceback
from collections import defaultdict, OrderedDict

import cv2
import numpy as np
import pandas as pd

from fast_setup import (
    detect_and_track, camera_movement, calibrate,
    _video_frame_count, _video_fps, DETECT_EVERY, CALIB_EVERY,
)
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from team_assigner import TeamAssigner
from utils import get_center_of_bbox, get_foot_position

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}

STUB_ROOT = os.path.join('stubs', 'batch')

# ── Fixed thresholds (do not retune from batch findings) ───────────────────
TEAM_RESOLVED_PASS_PCT     = 0.85
TEAM_RESOLVED_DEGRADED_PCT = 0.70
CALIB_MEDIAN_PASS_PCT      = 0.90
CALIB_MEDIAN_DEGRADED_PCT  = 0.75
CAMERA_SPIKE_PX            = 15.0
CAMERA_SUSTAINED_PX        = 25.0
CAMERA_SUSTAINED_RUN       = 3       # frames, consecutive, to count as "simultaneous"

# Goalkeeper tolerance: flat constant (one keeper per team, generously
# doubled for track fragmentation/re-ID noise).
#
# A per-clip formula derived from outfield-track fragmentation rate was
# tried (2026-07-27) and reverted (2026-07-28): the correlation between
# team=3 track count and outfield fragmentation (n_merged_pairs +
# n_rejected_merge_candidates / raw_track_count) was only r=0.425
# (r^2=0.18) across the 39-clip dev batch -- too weak to justify a derived
# formula. In practice the scale constant had to be widened until it
# cleared BOTH flagged clips (BarcaMadridPT1_w529 gk=7, LiverpoolMadrid_w877
# gk=9), and LiverpoolMadrid_w877's own frag_rate (0.877) was unremarkable
# next to clips with similar/higher frag_rate and far fewer gk tracks (e.g.
# LiverpoolMadrid_w1717: frag_rate=0.905, gk=6) -- meaning the fit only
# cleared w877 at a scale wide enough to function as a disguised flat
# raise, not a genuine derivation from the mechanism.
#
# BarcaMadridPT1_w529 (gk=7) and LiverpoolMadrid_w877 (gk=9) are logged as
# genuine, unresolved FAILs under this flat threshold. If revisited, the
# right signal is goalkeeper-specific (camera proximity to the goal box,
# penalty-area congestion during the flagged windows), not an outfield-
# fragmentation proxy.
GOALKEEPER_TRACK_MAX = 6

# From analyze_danger_score.py — reused as a constant + reimplemented scan
# (see module docstring for why it's copied rather than imported).
UNCERTAIN_GAP_MAX_FRAMES = 15

# From detect_id_swaps.py — copied verbatim (module docstring above).
ID_SWAP_MIN_TRACK_SAMPLES = 10
ID_SWAP_MIN_SEGMENT       = 5
ID_SWAP_SEGMENT_PURITY_MIN = 0.88

_VERDICT_RANK = {'PASS': 0, 'DEGRADED': 1, 'FAIL': 2}


# ─────────────────────────────────────────────────────────────────────────
# Frame access — TeamAssigner needs tracks['players'][fn] AND
# video_frames[fn] together (resolve_all_teams reads them strictly in
# increasing fn order; merge_fragmented_tracks's _avg_color jumps between
# different pids' frame ranges). utils.read_video() loads the WHOLE clip
# into a Python list, which is what main.py does for the small Union Berlin
# clip but is exactly what fast_setup.py's docstring says is infeasible for
# a long clip (~250GB for the 42k-frame Liverpool/PSG video). This wraps a
# single cv2.VideoCapture with cheap sequential reads for the common case
# and an explicit reseek only when the access pattern actually jumps.
# ─────────────────────────────────────────────────────────────────────────
class SeekableFrameAccessor:
    """cache_size: small bounded LRU (default 32 frames, ~200MB at 1080p) —
    NOT for the dominant sequential pass (resolve_all_teams), which streams
    through in one direction and gets no benefit from caching, but for
    merge_fragmented_tracks's _avg_color, which reads stride-3 across each
    CANDIDATE PAIR's full track range and does so once per candidate a pid
    appears in — overlapping ranges across candidates would otherwise
    reseek+redecode the same frame from scratch every time. Kept small
    deliberately: this machine runs tight on free memory already (see
    batch_validate.py's own diagnostic notes), so an unbounded or large
    cache would trade one performance problem for a worse one."""
    def __init__(self, video_path, n_frames, cache_size=32):
        self.video_path = video_path
        self.n_frames = n_frames
        self._cap = cv2.VideoCapture(video_path)
        self._next_idx = 0
        self._cache = OrderedDict()
        self._cache_size = cache_size
        # Access-pattern profiling (always on — negligible overhead, a few
        # counter increments and time.time() calls per read). Added while
        # diagnosing team resolution taking ~200x main.py's cost: this is
        # what actually proved/measures where that cost goes (seek+redecode
        # vs cheap sequential continuation), see .stats_summary().
        self.stats = {'n_reads': 0, 'n_cache_hits': 0, 'n_sequential_reads': 0,
                      'n_seeks': 0, 'sequential_read_time': 0.0, 'seek_read_time': 0.0}

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.n_frames:
            raise IndexError(idx)
        if idx in self._cache:
            self._cache.move_to_end(idx)
            self.stats['n_cache_hits'] += 1
            return self._cache[idx]
        is_seek = idx != self._next_idx
        _t0 = time.time()
        if is_seek:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if not ret:
            raise IndexError(f"could not read frame {idx} of {self.video_path}")
        dt = time.time() - _t0
        self.stats['n_reads'] += 1
        if is_seek:
            self.stats['n_seeks'] += 1
            self.stats['seek_read_time'] += dt
        else:
            self.stats['n_sequential_reads'] += 1
            self.stats['sequential_read_time'] += dt
        self._next_idx = idx + 1
        self._cache[idx] = frame
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return frame

    def stats_summary(self):
        s = self.stats
        avg_seek = s['seek_read_time'] / s['n_seeks'] if s['n_seeks'] else 0.0
        avg_seq = s['sequential_read_time'] / s['n_sequential_reads'] if s['n_sequential_reads'] else 0.0
        return (f"reads={s['n_reads']} (seq={s['n_sequential_reads']}, seeks={s['n_seeks']}, "
               f"cache_hits={s['n_cache_hits']})  "
               f"seq_time={s['sequential_read_time']:.1f}s (avg {avg_seq*1000:.1f}ms/read)  "
               f"seek_time={s['seek_read_time']:.1f}s (avg {avg_seek*1000:.1f}ms/read)")

    def release(self):
        self._cap.release()


# Memory safety cap for holding an entire team-resolution window in RAM at
# once (see module docstring's DIAGNOSIS). 8GB is comfortably under what
# this machine typically has free while covering a properly --max-frames-
# bounded window (a 750-frame 1080p window is ~4.65GB) with margin for
# other processes. Only a forgotten/oversized --max-frames should trip the
# SeekableFrameAccessor fallback below.
TEAM_FRAME_LIST_MEMORY_CAP_BYTES = 8 * 1024 ** 3


def _load_frames_for_team_resolution(video_path, n_frames, width, height):
    """Decode the window ONCE, sequentially, into a plain in-memory list —
    this is the actual fix for team resolution's ~200x slowdown (see module
    docstring). TeamAssigner.resolve_all_teams / merge_fragmented_tracks /
    _scan_id_swaps only ever need __len__ / __getitem__(int), so a plain
    list is a drop-in replacement for SeekableFrameAccessor here, but every
    access — sequential OR merge_fragmented_tracks's arbitrary-range jumps
    — becomes O(1) list indexing instead of a video seek+redecode. This is
    exactly what main.py's read_video() does; the only reason this harness
    didn't do the same from the start is fast_setup.py's detect/camera/
    calibrate stages genuinely can't (a full 42k-frame clip would need
    ~250GB) — but those stages already stream past that constraint on their
    own, and by the time team resolution runs, n_frames is always whatever
    --max-frames bounded window this harness is processing, never the raw
    source. Falls back to the slower seeking accessor (with a warning) if
    the window is large enough that holding it all in memory would recreate
    that same memory-pressure problem.
    """
    est_bytes = n_frames * height * width * 3
    if est_bytes > TEAM_FRAME_LIST_MEMORY_CAP_BYTES:
        print(f"  [team] WARNING: {n_frames} frames at {width}x{height} would need "
              f"~{est_bytes / 1e9:.1f}GB in memory (cap "
              f"{TEAM_FRAME_LIST_MEMORY_CAP_BYTES / 1e9:.0f}GB) -- falling back to "
              f"seek-based access. This will be SLOW for team resolution (see module "
              f"docstring's DIAGNOSIS) -- pass a smaller --max-frames.")
        return SeekableFrameAccessor(video_path, n_frames)

    frames = []
    cap = cv2.VideoCapture(video_path)
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


# ─────────────────────────────────────────────────────────────────────────
# Small helpers copied from trackers/tracker.py's model-independent methods
# ─────────────────────────────────────────────────────────────────────────
def _add_position_to_tracks(tracks):
    for obj, object_tracks in tracks.items():
        for frame in object_tracks:
            for tid, info in frame.items():
                bbox = info['bbox']
                info['position'] = (get_center_of_bbox(bbox) if obj == 'ball'
                                     else get_foot_position(bbox))


def _interpolate_ball_positions(ball_positions):
    boxes = [x.get(1, {}).get('bbox', []) for x in ball_positions]
    df = pd.DataFrame(boxes, columns=['x1', 'y1', 'x2', 'y2']).interpolate().bfill()
    return [{1: {'bbox': b}} for b in df.to_numpy().tolist()]


# ─────────────────────────────────────────────────────────────────────────
# Ball detection-rate / gap-length — analyze_danger_score.py's convention
# ─────────────────────────────────────────────────────────────────────────
def _ball_gap_stats(ball_raw, n_frames, detect_every, fps):
    """ball_raw = tracks['ball'] straight out of fast_setup.detect_and_track,
    BEFORE Tracker-style pandas interpolation. A frame only counts as a
    real ("raw") detection if it's an actual detection-attempt frame
    (fn % detect_every == 0) and the primary model found the ball there —
    every other frame's ball bbox is fabricated (frame-skip gap-fill or
    pandas interpolation), not observed. Mirrors
    analyze_danger_score.find_uncertain_gaps.

    A flat "% frames with SOME ball value" is misleading on its own —
    pandas interpolation/bfill fills essentially every frame regardless of
    how long the underlying real-detection gap was, and this repo's own
    danger-score work already found long interpolated ball gaps to be the
    single biggest source of false danger-score peaks (see
    UNCERTAIN_GAP_MAX_FRAMES / find_uncertain_gaps above). So alongside the
    raw detection rate, this returns the actual GAP-LENGTH DISTRIBUTION
    (median/p90/max, count of gaps over 1s/3s) and, per-frame, the
    distance to the nearest real detection — reported as "% of frames
    within Ns of a real detection" for N in {1, 3} seconds, which is what
    actually determines how stale an interpolated ball position is at any
    given frame, not just whether pandas managed to fill a value there."""
    is_real = np.zeros(n_frames, dtype=bool)
    n_attempted = 0
    n_hit = 0
    for fn in range(n_frames):
        if fn % detect_every != 0:
            continue
        n_attempted += 1
        if fn < len(ball_raw) and ball_raw[fn].get(1) is not None:
            is_real[fn] = True
            n_hit += 1

    real_indices = np.flatnonzero(is_real)
    if len(real_indices) == 0:
        dist_to_real = np.full(n_frames, np.inf)
    else:
        all_idx = np.arange(n_frames)
        pos = np.searchsorted(real_indices, all_idx)
        pos_right = np.clip(pos, 0, len(real_indices) - 1)
        pos_left = np.clip(pos - 1, 0, len(real_indices) - 1)
        dist_right = np.abs(real_indices[pos_right] - all_idx)
        dist_left = np.abs(real_indices[pos_left] - all_idx)
        dist_to_real = np.minimum(dist_left, dist_right)

    gaps = []
    n_uncertain_gaps = 0
    gap_start = None
    for fn in range(n_frames):
        if not is_real[fn]:
            if gap_start is None:
                gap_start = fn
        elif gap_start is not None:
            gap_len = fn - gap_start
            gaps.append(gap_len)
            if gap_len > UNCERTAIN_GAP_MAX_FRAMES:
                n_uncertain_gaps += 1
            gap_start = None
    if gap_start is not None:
        gap_len = n_frames - gap_start
        gaps.append(gap_len)
        if gap_len > UNCERTAIN_GAP_MAX_FRAMES:
            n_uncertain_gaps += 1

    fps = fps if fps and fps > 0 else 25.0
    gaps_arr = np.array(gaps, dtype=np.float64) if gaps else np.array([0.0])
    max_gap = float(np.max(gaps_arr)) if gaps else 0.0
    median_gap = float(np.median(gaps_arr)) if gaps else 0.0
    p90_gap = float(np.percentile(gaps_arr, 90)) if gaps else 0.0
    n_gaps_over_1s = int(sum(1 for g in gaps if g > fps))
    n_gaps_over_3s = int(sum(1 for g in gaps if g > 3 * fps))

    return {
        'raw_detection_pct': (n_hit / n_attempted) if n_attempted else 0.0,
        'n_attempted': n_attempted,
        'n_hit': n_hit,
        'n_gaps': len(gaps),
        'max_gap_frames': int(max_gap), 'max_gap_sec': round(max_gap / fps, 2),
        'median_gap_frames': round(median_gap, 1), 'median_gap_sec': round(median_gap / fps, 2),
        'p90_gap_frames': round(p90_gap, 1), 'p90_gap_sec': round(p90_gap / fps, 2),
        'n_gaps_over_1s': n_gaps_over_1s, 'n_gaps_over_3s': n_gaps_over_3s,
        'n_gaps_over_uncertain_threshold': n_uncertain_gaps,
        'pct_frames_within_1s_of_real': round(float(np.mean(dist_to_real <= fps)), 4),
        'pct_frames_within_3s_of_real': round(float(np.mean(dist_to_real <= 3 * fps)), 4),
    }


def _ball_post_interp_pct(ball_interp, n_frames):
    n_valid = 0
    for fn in range(min(n_frames, len(ball_interp))):
        bbox = ball_interp[fn].get(1, {}).get('bbox')
        if bbox and len(bbox) == 4 and not any(
                (isinstance(v, float) and math.isnan(v)) for v in bbox):
            n_valid += 1
    return n_valid / n_frames if n_frames else 0.0


# ─────────────────────────────────────────────────────────────────────────
# Gray-frame / unresolved — compare_gray_frames.py's convention
# ─────────────────────────────────────────────────────────────────────────
def _total_and_gray(players):
    total = 0
    gray = 0
    for fd in players:
        for pid, info in fd.items():
            total += 1
            if info.get('team', 0) == 0:
                gray += 1
    return total, gray


# ─────────────────────────────────────────────────────────────────────────
# ID-swap scan — detect_id_swaps.py's STEP1/STEP2 algorithm, copied (see
# module docstring). Reuses this run's already-fitted TeamAssigner instead
# of re-fitting a second KMeans.
# ─────────────────────────────────────────────────────────────────────────
def _id_swap_best_split(samples):
    n = len(samples)
    guesses = [g for _, g in samples]
    best = None
    best_key = None
    for k in range(ID_SWAP_MIN_SEGMENT, n - ID_SWAP_MIN_SEGMENT + 1):
        segA, segB = guesses[:k], guesses[k:]
        majA = max(set(segA), key=segA.count)
        majB = max(set(segB), key=segB.count)
        if majA == majB:
            continue
        purA = segA.count(majA) / len(segA)
        purB = segB.count(majB) / len(segB)
        if purA < ID_SWAP_SEGMENT_PURITY_MIN or purB < ID_SWAP_SEGMENT_PURITY_MIN:
            continue
        weighted = (purA * len(segA) + purB * len(segB)) / n
        key = (min(purA, purB), min(len(segA), len(segB)))
        if best is None or key > best_key:
            best = (k, purA, purB, majA, majB, weighted)
            best_key = key
    return best


def _scan_id_swaps(ta, players_tracks):
    """Reuses ta.pending_samples (guess ints) + ta.color_vote_frames (the
    index-parallel frame numbers team_assigner_color_bak.py's
    resolve_all_teams now records specifically for this) instead of
    re-sampling every player's colour a second time across the whole clip —
    that used to be a literal second full-video pass duplicating
    resolve_all_teams's own work (same get_player_color() calls, same
    frames), found while diagnosing why a 750-frame validation run was so
    much slower than expected. No video frame access happens here at all
    anymore; only tracks['players'] (already in memory) for the
    majority-goalkeeper exclusion and the presence/gap check."""
    if not ta.locked_colors:
        return []

    gk_frac = {}
    for fd in players_tracks:
        for pid, det in fd.items():
            g, t = gk_frac.get(pid, (0, 0))
            gk_frac[pid] = (g + (1 if det.get('is_goalkeeper') else 0), t + 1)
    majority_gk = {pid for pid, (g, t) in gk_frac.items() if t and g / t > 0.5}

    present_by_pid = defaultdict(set)
    for fn, fd in enumerate(players_tracks):
        for pid in fd:
            present_by_pid[pid].add(fn)

    samples_by_pid = {}
    for pid, guesses in ta.pending_samples.items():
        if pid in majority_gk:
            continue
        fns = ta.color_vote_frames.get(pid, [])
        samples_by_pid[pid] = list(zip(fns, guesses))

    flagged = []
    for pid, samples in samples_by_pid.items():
        if len(samples) < ID_SWAP_MIN_TRACK_SAMPLES:
            continue
        result = _id_swap_best_split(samples)
        if result is None:
            continue
        k, purA, purB, majA, majB, weighted = result
        guesses = [g for _, g in samples]
        whole_maj = max(set(guesses), key=guesses.count)
        whole_purity = guesses.count(whole_maj) / len(guesses)
        split_before, split_after = samples[k - 1][0], samples[k][0]
        present = present_by_pid.get(pid, set())
        missing = [fn for fn in range(split_before, split_after + 1) if fn not in present]
        flagged.append({
            'pid': int(pid), 'n_samples': len(samples),
            'whole_purity': round(whole_purity, 3),
            'weighted_purity': round(weighted, 3),
            'improvement': round(weighted - whole_purity, 3),
            'split_before_fn': split_before, 'split_after_fn': split_after,
            'signature': 'weak' if missing else 'strong',
        })
    return flagged


# ─────────────────────────────────────────────────────────────────────────
# Camera-motion spike detection (no existing code does this comparison —
# see module docstring for the "3+ simultaneous >25px" interpretation)
# ─────────────────────────────────────────────────────────────────────────
def _camera_motion_stats(camera_movement_per_frame):
    mags = [math.hypot(dx, dy) for dx, dy in camera_movement_per_frame]
    spike_count = sum(1 for m in mags if m > CAMERA_SPIKE_PX)
    max_mag = max(mags) if mags else 0.0

    run = 0
    sustained_events = 0
    for m in mags:
        if m > CAMERA_SUSTAINED_PX:
            run += 1
        else:
            if run >= CAMERA_SUSTAINED_RUN:
                sustained_events += 1
            run = 0
    if run >= CAMERA_SUSTAINED_RUN:
        sustained_events += 1

    return {
        'spike_frame_count_over_15px': spike_count,
        'max_magnitude_px': round(max_mag, 2),
        'sustained_burst_events': sustained_events,  # runs of >=3 consecutive frames >25px
    }


# ─────────────────────────────────────────────────────────────────────────
# Verdicts
# ─────────────────────────────────────────────────────────────────────────
def _verdict_team(pct):
    if pct >= TEAM_RESOLVED_PASS_PCT:
        return 'PASS'
    if pct >= TEAM_RESOLVED_DEGRADED_PCT:
        return 'DEGRADED'
    return 'FAIL'


def _verdict_calib(median):
    if median >= CALIB_MEDIAN_PASS_PCT:
        return 'PASS'
    if median >= CALIB_MEDIAN_DEGRADED_PCT:
        return 'DEGRADED'
    return 'FAIL'


def _verdict_camera(sustained_events):
    return 'PASS' if sustained_events == 0 else 'FAIL'


def _count_distinct_goalkeepers(player_team_dict, merged_pairs):
    """Collapse ByteTrack ID fragments of the SAME physical goalkeeper into
    one logical count, using merge_fragmented_tracks's own accepted merges
    (temporal/spatial adjacency + appearance agreement -- the same
    evidence already proven for outfield players, see team_assigner's
    merge_fragmented_tracks docstring). This is a REPORTING-layer
    collapse, not a change to team resolution itself: player_team_dict
    still has one entry per raw track id (every frame still needs its own
    render color), but "how many distinct goalkeepers were tracked" is a
    different question than "how many track ids are labeled team=3" --
    merge_fragmented_tracks already answers the identity question via its
    'a'/'b' pairs, this just unions them via connected components. Each
    unlinked team=3 pid counts as its own goalkeeper."""
    gk_pids = {pid for pid, t in player_team_dict.items() if t == 3}
    parent = {pid: pid for pid in gk_pids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for pair in merged_pairs:
        a, b = pair['a'], pair['b']
        if a in gk_pids and b in gk_pids:
            union(a, b)

    return len({find(pid) for pid in gk_pids})


def _verdict_gk(track_count):
    return 'PASS' if track_count <= GOALKEEPER_TRACK_MAX else 'FAIL'


def _overall_verdict(verdicts):
    return max(verdicts, key=lambda v: _VERDICT_RANK[v])


# ─────────────────────────────────────────────────────────────────────────
# Lightweight rendering (DEGRADED/FAIL clips only, or --render-all)
# ─────────────────────────────────────────────────────────────────────────
def _draw_ellipse(frame, bbox, color, track_id=None):
    y2 = int(bbox[3])
    x_center = int((bbox[0] + bbox[2]) / 2)
    width = bbox[2] - bbox[0]
    cv2.ellipse(frame, (x_center, y2), (int(width), int(0.35 * width)),
                0.0, -45, 235, color, 2, cv2.LINE_4)
    if track_id is not None:
        cv2.putText(frame, str(track_id), (x_center - 10, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame


def render_sample_frames(video_path, tracks, calibration_confidence,
                          camera_movement_per_frame, n_frames, out_dir, n_samples=8):
    os.makedirs(out_dir, exist_ok=True)
    if n_frames <= 1:
        sample_indices = [0]
    else:
        sample_indices = sorted(set(
            int(round(i * (n_frames - 1) / (n_samples - 1))) for i in range(n_samples)))

    cap = cv2.VideoCapture(video_path)
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = frame.copy()
        for pid, info in tracks['players'][idx].items():
            color = info.get('team_color', (128, 128, 128))
            color = tuple(int(c) for c in color)
            _draw_ellipse(frame, info['bbox'], color, pid)
        ball = tracks['ball'][idx].get(1) if idx < len(tracks['ball']) else None
        if ball and ball.get('bbox'):
            bx = ball['bbox']
            cx, cy = int((bx[0] + bx[2]) / 2), int((bx[1] + bx[3]) / 2)
            cv2.circle(frame, (cx, cy), 8, (0, 255, 255), 2)
        conf = calibration_confidence.get(idx)
        mag = math.hypot(*camera_movement_per_frame[idx]) if idx < len(camera_movement_per_frame) else 0.0
        txt = f"frame {idx}"
        if conf is not None:
            txt += f"  calib_conf={conf:.0%}  cam_mag={mag:.1f}px"
        cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(out_dir, f"frame_{idx:06d}.jpg"), frame)
    cap.release()


# ─────────────────────────────────────────────────────────────────────────
# Per-clip cache (only the expensive fast_setup detect/camera/calibrate
# stages are cached — enrichment always re-runs so it reflects current code)
# ─────────────────────────────────────────────────────────────────────────
def _clip_cache_dir(video_path):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    d = os.path.join(STUB_ROOT, stem)
    os.makedirs(d, exist_ok=True)
    return d


def _load_or_run_fast_stages(video_path, n_frames, force):
    """Returns (tracks, camera_movement_per_frame, homography_per_frame,
    cache_used, pipeline_timing). pipeline_timing carries per-stage WALL
    time plus, for detect/calibrate, the per-inference-frame mean/median/
    p95/max distribution (see fast_setup._timing_stats) — cached alongside
    the stub data so a cache hit still reports the ORIGINAL run's timing
    rather than silently going blank."""
    cache_dir = _clip_cache_dir(video_path)
    meta_path = os.path.join(cache_dir, 'meta.pkl')
    tracks_path = os.path.join(cache_dir, 'tracks.pkl')
    cam_path = os.path.join(cache_dir, 'camera_movement.pkl')
    hom_path = os.path.join(cache_dir, 'homography.pkl')
    timing_path = os.path.join(cache_dir, 'timing.pkl')

    video_stat = os.stat(video_path)
    fingerprint = {
        'video_path': os.path.abspath(video_path),
        'n_frames': n_frames,
        'video_size': video_stat.st_size,
        'video_mtime': video_stat.st_mtime,
        'detect_every': DETECT_EVERY,
        'calib_every': CALIB_EVERY,
    }

    if not force and os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            cached_meta = pickle.load(f)
        if (cached_meta == fingerprint and os.path.exists(tracks_path)
                and os.path.exists(cam_path) and os.path.exists(hom_path)):
            print(f"  [cache] reusing cached detect/camera/calibrate stages for {video_path}")
            with open(tracks_path, 'rb') as f:
                tracks = pickle.load(f)
            with open(cam_path, 'rb') as f:
                camera_movement_per_frame = pickle.load(f)
            with open(hom_path, 'rb') as f:
                homography_per_frame = pickle.load(f)
            pipeline_timing = {}
            if os.path.exists(timing_path):
                with open(timing_path, 'rb') as f:
                    pipeline_timing = pickle.load(f)
            return (tracks, camera_movement_per_frame, homography_per_frame,
                    True, pipeline_timing)

    print(f"\n== Stage 1/3: detect+track ({video_path}, {n_frames} frames) ==")
    _t0 = time.time()
    tracks, n_detected, n_interp, _, detect_timing = detect_and_track(video_path, n_frames)
    detect_wall_sec = time.time() - _t0

    print(f"\n== Stage 2/3: camera movement ==")
    _t0 = time.time()
    camera_movement_per_frame = camera_movement(video_path, n_frames)
    camera_wall_sec = time.time() - _t0

    print(f"\n== Stage 3/3: pitch calibration ==")
    _t0 = time.time()
    homography_per_frame, n_calibrated, n_calib_interp, calib_timing = calibrate(video_path, n_frames)
    calib_wall_sec = time.time() - _t0

    with open(tracks_path, 'wb') as f:
        pickle.dump(tracks, f)
    with open(cam_path, 'wb') as f:
        pickle.dump(camera_movement_per_frame, f)
    with open(hom_path, 'wb') as f:
        pickle.dump(homography_per_frame, f)
    with open(meta_path, 'wb') as f:
        pickle.dump(fingerprint, f)

    pipeline_timing = {
        'detect_wall_sec': round(detect_wall_sec, 1),
        'camera_wall_sec': round(camera_wall_sec, 1),
        'calibrate_wall_sec': round(calib_wall_sec, 1),
        'detect_frame_timing': detect_timing,
        'calibrate_frame_timing': calib_timing,
    }
    with open(timing_path, 'wb') as f:
        pickle.dump(pipeline_timing, f)

    tracks['_n_detected'] = n_detected
    tracks['_n_interp'] = n_interp
    tracks['_n_calibrated'] = n_calibrated
    tracks['_n_calib_interp'] = n_calib_interp
    return (tracks, camera_movement_per_frame, homography_per_frame,
            False, pipeline_timing)


# ─────────────────────────────────────────────────────────────────────────
# Per-clip pipeline
# ─────────────────────────────────────────────────────────────────────────
def process_clip(video_path, max_frames, out_dir, force=False,
                  render_mode='auto', n_render_samples=8):
    """render_mode: 'auto' (only DEGRADED/FAIL), 'all', 'none'."""
    t0 = time.time()
    clip_name = os.path.splitext(os.path.basename(video_path))[0]

    n_frames_total = _video_frame_count(video_path)
    fps = _video_fps(video_path)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    n_frames = min(n_frames_total, max_frames) if max_frames else n_frames_total

    tracks, camera_movement_per_frame, homography_per_frame, cache_used, pipeline_timing = \
        _load_or_run_fast_stages(video_path, n_frames, force)

    n_detected = tracks.pop('_n_detected', None)
    n_interp = tracks.pop('_n_interp', None)
    n_calibrated = tracks.pop('_n_calibrated', None)
    n_calib_interp = tracks.pop('_n_calib_interp', None)

    # ── position / camera-adjust / view-transform (same steps main.py runs) ──
    _t_stage = time.time()
    _add_position_to_tracks(tracks)

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        raise IOError(f"Cannot read first frame of {video_path}")
    cam_estimator = CameraMovementEstimator(first_frame)
    cam_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    view_transformer = ViewTransformer()
    fallback_H = view_transformer.persepctive_trasnformer.astype(np.float32)
    fallback_H_inv = np.linalg.inv(fallback_H).astype(np.float32)
    homography_per_frame = {
        fn: h if h is not None else (fallback_H, fallback_H_inv)
        for fn, h in homography_per_frame.items()
    }
    view_transformer.add_transformed_position_to_tracks(tracks, homography_per_frame)
    calibration_confidence = view_transformer.compute_frame_confidence(tracks)
    enrichment_wall_sec = time.time() - _t_stage

    conf_vals = list(calibration_confidence.values())[:n_frames]
    conf_mean = float(np.mean(conf_vals)) if conf_vals else 0.0
    conf_median = float(np.median(conf_vals)) if conf_vals else 0.0
    conf_pct_below_70 = (sum(1 for v in conf_vals if v < 0.70) / len(conf_vals)) if conf_vals else 0.0

    # ── Ball: capture RAW state before pandas interpolation overwrites it ──
    _t_stage = time.time()
    ball_raw = tracks['ball']
    ball_gap_stats = _ball_gap_stats(ball_raw, n_frames, DETECT_EVERY, fps)
    tracks['ball'] = _interpolate_ball_positions(tracks['ball'])
    ball_post_pct = _ball_post_interp_pct(tracks['ball'], n_frames)
    ball_wall_sec = time.time() - _t_stage

    # ── Team resolution (exact main.py sequence) ────────────────────────────
    # Sub-stage timed + printed individually (not just a single team_wall_sec
    # total) after team resolution turned out to be ~200x main.py's cost on
    # the same clip (1130s mean vs main.py's ~5.6s) — see module docstring's
    # DIAGNOSIS note for the access-pattern root cause and fix.
    _t_stage = time.time()
    frames = _load_frames_for_team_resolution(video_path, n_frames, width, height)
    print(f"  [team] frames loaded ({len(frames)}) in {time.time() - _t_stage:.1f}s")

    _frame_stats = lambda: frames.stats_summary() if hasattr(frames, 'stats_summary') else 'in-memory list, O(1) access, no seeking'

    _t_sub = time.time()
    ta = TeamAssigner()
    ta.resolve_all_teams(frames, tracks['players'], tracks['referees'])
    resolve_wall_sec = time.time() - _t_sub
    print(f"  [team] evidence gathering (resolve_all_teams): {resolve_wall_sec:.1f}s  "
          f"[{_frame_stats()}]")

    _t_sub = time.time()
    fallback_assignments = ta.finalize_fallback_assignments()
    fallback_wall_sec = time.time() - _t_sub
    print(f"  [team] fallback: {fallback_wall_sec:.1f}s  ({len(fallback_assignments)} assigned)")

    _t_sub = time.time()
    merged_pairs, rejected_pairs = ta.merge_fragmented_tracks(
        frames, tracks['players'], exclude_pairs=[])
    merge_wall_sec = time.time() - _t_sub
    print(f"  [team] merge (relink fragments): {merge_wall_sec:.1f}s  "
          f"({len(merged_pairs)} merged, {len(rejected_pairs)} rejected)  "
          f"[{_frame_stats()}]")

    _t_sub = time.time()
    id_swaps = _scan_id_swaps(ta, tracks['players'])
    id_swap_wall_sec = time.time() - _t_sub
    print(f"  [team] id-swap scan: {id_swap_wall_sec:.1f}s  ({len(id_swaps)} flagged)")

    _t_sub = time.time()
    ta.apply_final_teams(tracks['players'])
    apply_wall_sec = time.time() - _t_sub
    print(f"  [team] apply final teams: {apply_wall_sec:.1f}s")
    if hasattr(frames, 'release'):
        frames.release()
    team_wall_sec = time.time() - _t_stage

    all_pids = set()
    for fd in tracks['players']:
        all_pids.update(fd.keys())
    resolved_pids = {pid for pid in all_pids if ta.player_team_dict.get(pid, 0) != 0}
    n_resolved = len(resolved_pids)
    n_unresolved = len(all_pids) - n_resolved
    resolved_pct = (n_resolved / len(all_pids)) if all_pids else 1.0

    gray_total, gray_count = _total_and_gray(tracks['players'])
    gray_pct = (gray_count / gray_total) if gray_total else 0.0

    cluster_sep = None
    if 1 in ta.locked_colors and 2 in ta.locked_colors:
        cluster_sep = float(np.linalg.norm(ta.locked_colors[1] - ta.locked_colors[2]))

    gk_track_count = _count_distinct_goalkeepers(ta.player_team_dict, merged_pairs)

    camera_stats = _camera_motion_stats(camera_movement_per_frame[:n_frames])

    # ── Verdicts (only the 4 categories the spec gives thresholds for) ─────
    team_verdict = _verdict_team(resolved_pct)
    calib_verdict = _verdict_calib(conf_median)
    camera_verdict = _verdict_camera(camera_stats['sustained_burst_events'])
    gk_verdict = _verdict_gk(gk_track_count)
    overall = _overall_verdict([team_verdict, calib_verdict, camera_verdict, gk_verdict])

    _t_stage = time.time()
    do_render = (render_mode == 'all') or (render_mode == 'auto' and overall != 'PASS')
    render_dir = None
    if do_render:
        render_dir = os.path.join(out_dir, clip_name, 'render')
        render_sample_frames(video_path, tracks, calibration_confidence,
                              camera_movement_per_frame, n_frames, render_dir,
                              n_samples=n_render_samples)
    render_wall_sec = time.time() - _t_stage

    id_swap_strong = sum(1 for f in id_swaps if f['signature'] == 'strong')
    id_swap_weak = sum(1 for f in id_swaps if f['signature'] == 'weak')

    pipeline_timing = {
        **pipeline_timing,
        'enrichment_wall_sec': round(enrichment_wall_sec, 1),
        'ball_wall_sec': round(ball_wall_sec, 1),
        'team_wall_sec': round(team_wall_sec, 1),
        'team_resolve_wall_sec': round(resolve_wall_sec, 1),
        'team_fallback_wall_sec': round(fallback_wall_sec, 1),
        'team_merge_wall_sec': round(merge_wall_sec, 1),
        'team_id_swap_wall_sec': round(id_swap_wall_sec, 1),
        'team_apply_wall_sec': round(apply_wall_sec, 1),
        'team_frame_access_stats': dict(frames.stats) if hasattr(frames, 'stats') else None,
        'render_wall_sec': round(render_wall_sec, 1),
    }

    scorecard = {
        'clip': clip_name,
        'video_path': video_path,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'cache_used': cache_used,
        'processing_time_sec': round(time.time() - t0, 1),
        'video': {
            'fps': fps, 'resolution': f"{width}x{height}",
            'n_frames_total': n_frames_total, 'n_frames_processed': n_frames,
        },
        'pipeline': {
            'detect_every': DETECT_EVERY, 'calib_every': CALIB_EVERY,
            'n_frames_detected': n_detected, 'n_frames_interp_detect': n_interp,
            'n_frames_calibrated': n_calibrated, 'n_frames_interp_calib': n_calib_interp,
        },
        'timing': pipeline_timing,
        'team': {
            'cluster_separation': round(cluster_sep, 2) if cluster_sep is not None else None,
            'n_player_tracks_total': len(all_pids),
            'n_resolved': n_resolved, 'n_unresolved': n_unresolved,
            'resolved_pct': round(resolved_pct, 4),
            'gray_frame_pct': round(gray_pct, 4),
            'n_fallback_assigned': len(fallback_assignments),
            'n_merged_pairs': len(merged_pairs),
            'n_rejected_merge_candidates': len(rejected_pairs),
            'verdict': team_verdict,
        },
        'calibration': {
            'mean_confidence': round(conf_mean, 4), 'median_confidence': round(conf_median, 4),
            'pct_frames_below_70': round(conf_pct_below_70, 4),
            'verdict': calib_verdict,
        },
        'camera_motion': {**camera_stats, 'verdict': camera_verdict},
        'identity': {
            'raw_track_count': len(all_pids),
            'n_flagged_candidates': len(id_swaps),
            'strong_signature': id_swap_strong, 'weak_signature': id_swap_weak,
            # Normalized so candidate counts are comparable across differently-
            # sized windows (e.g. 750 vs 1500 frames) without doing the math
            # by hand every time — added after an 8-strong-in-750-frames vs
            # 6-strong-in-1500-frames comparison came up during validation.
            'strong_signature_per_100_frames': round(id_swap_strong / n_frames * 100, 3) if n_frames else None,
            'weak_signature_per_100_frames': round(id_swap_weak / n_frames * 100, 3) if n_frames else None,
            'flagged': id_swaps,
            'verdict': None,   # informational only — no threshold given
        },
        'goalkeepers': {
            'track_count': gk_track_count,
            'expected_max': GOALKEEPER_TRACK_MAX,
            # Explicit margin so a boundary pass (e.g. exactly at the ceiling)
            # doesn't read identically to a comfortable one in the report.
            'margin_label': (f"{gk_track_count}/{GOALKEEPER_TRACK_MAX} — over limit" if gk_track_count > GOALKEEPER_TRACK_MAX
                              else f"{gk_track_count}/{GOALKEEPER_TRACK_MAX} — at limit" if gk_track_count == GOALKEEPER_TRACK_MAX
                              else f"{gk_track_count}/{GOALKEEPER_TRACK_MAX} — {GOALKEEPER_TRACK_MAX - gk_track_count} under limit"),
            'verdict': gk_verdict,
        },
        'ball': {
            'raw_detection_pct': round(ball_gap_stats['raw_detection_pct'], 4),
            'post_interpolation_pct': round(ball_post_pct, 4),
            'n_gaps': ball_gap_stats['n_gaps'],
            'median_gap_frames': ball_gap_stats['median_gap_frames'],
            'median_gap_sec': ball_gap_stats['median_gap_sec'],
            'p90_gap_frames': ball_gap_stats['p90_gap_frames'],
            'p90_gap_sec': ball_gap_stats['p90_gap_sec'],
            'max_gap_frames': ball_gap_stats['max_gap_frames'],
            'max_gap_sec': ball_gap_stats['max_gap_sec'],
            'n_gaps_over_1s': ball_gap_stats['n_gaps_over_1s'],
            'n_gaps_over_3s': ball_gap_stats['n_gaps_over_3s'],
            'n_gaps_over_uncertain_threshold': ball_gap_stats['n_gaps_over_uncertain_threshold'],
            'pct_frames_within_1s_of_real': ball_gap_stats['pct_frames_within_1s_of_real'],
            'pct_frames_within_3s_of_real': ball_gap_stats['pct_frames_within_3s_of_real'],
            'verdict': None,   # informational only — known limitation, not scored
        },
        'overall_verdict': overall,
        'rendered': do_render,
        'render_dir': render_dir,
    }
    return scorecard


# ─────────────────────────────────────────────────────────────────────────
# Batch driver
# ─────────────────────────────────────────────────────────────────────────
def _find_videos(folder):
    paths = []
    for ext in VIDEO_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f'*{ext}')))
    return sorted(paths)


def _scorecard_to_row(clip_name, scorecard):
    """The scorecard.json -> summary.csv row mapping. Pulled out of main()'s
    per-clip loop so --build-baseline can produce a byte-for-byte-comparable
    summary.csv from EXISTING scorecards, without re-running the pipeline."""
    return {
        'clip': clip_name, 'status': 'OK',
        'n_frames': scorecard['video']['n_frames_processed'],
        'fps': round(scorecard['video']['fps'], 2),
        'resolution': scorecard['video']['resolution'],
        'team_resolved_pct': f"{scorecard['team']['resolved_pct']:.1%}",
        'team_verdict': scorecard['team']['verdict'],
        'calib_median_pct': f"{scorecard['calibration']['median_confidence']:.1%}",
        'calib_verdict': scorecard['calibration']['verdict'],
        'camera_sustained_events': scorecard['camera_motion']['sustained_burst_events'],
        'camera_verdict': scorecard['camera_motion']['verdict'],
        'gk_margin': scorecard['goalkeepers']['margin_label'],
        'gk_verdict': scorecard['goalkeepers']['verdict'],
        'ball_raw_pct': f"{scorecard['ball']['raw_detection_pct']:.1%}",
        'ball_median_gap_sec': scorecard['ball']['median_gap_sec'],
        'ball_max_gap_sec': scorecard['ball']['max_gap_sec'],
        'ball_gaps_over_3s': scorecard['ball']['n_gaps_over_3s'],
        'ball_pct_within_3s': f"{scorecard['ball']['pct_frames_within_3s_of_real']:.1%}",
        'id_swap_strong': scorecard['identity']['strong_signature'],
        'id_swap_strong_per_100f': scorecard['identity']['strong_signature_per_100_frames'],
        'id_swap_weak': scorecard['identity']['weak_signature'],
        'overall_verdict': scorecard['overall_verdict'],
        'rendered': scorecard['rendered'],
    }


def _write_summary(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'summary.csv')
    md_path = os.path.join(out_dir, 'summary.md')

    fields = ['clip', 'status', 'n_frames', 'fps', 'resolution',
              'team_resolved_pct', 'team_verdict',
              'calib_median_pct', 'calib_verdict',
              'camera_sustained_events', 'camera_verdict',
              'gk_margin', 'gk_verdict',
              'ball_raw_pct', 'ball_median_gap_sec', 'ball_max_gap_sec',
              'ball_gaps_over_3s', 'ball_pct_within_3s',
              'id_swap_strong', 'id_swap_strong_per_100f', 'id_swap_weak',
              'overall_verdict', 'rendered']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fields})

    with open(md_path, 'w') as f:
        f.write('# Batch validation summary\n\n')
        f.write('| ' + ' | '.join(fields) + ' |\n')
        f.write('|' + '---|' * len(fields) + '\n')
        for row in rows:
            f.write('| ' + ' | '.join(str(row.get(k, '')) for k in fields) + ' |\n')

    return csv_path, md_path


def build_baseline(reports_dir, baseline_dir):
    """Snapshot every scorecard.json under reports_dir into baseline_dir as
    a committed golden baseline: scorecard.json only (no render/ images --
    those are large, regenerable, and not what comparison reads), plus a
    combined summary.csv/md built via the same _scorecard_to_row mapping a
    live run uses, so compare_to_baseline() is comparing like-for-like
    columns rather than two independently-evolved formats."""
    scorecard_paths = sorted(glob.glob(os.path.join(reports_dir, '*', 'scorecard.json')))
    if not scorecard_paths:
        raise SystemExit(f"No scorecard.json files found under {reports_dir}")

    os.makedirs(baseline_dir, exist_ok=True)
    rows = []
    saved_clips = []
    for path in scorecard_paths:
        clip_name = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            scorecard = json.load(f)
        clip_dir = os.path.join(baseline_dir, clip_name)
        os.makedirs(clip_dir, exist_ok=True)
        with open(os.path.join(clip_dir, 'scorecard.json'), 'w') as f:
            json.dump(scorecard, f, indent=2, default=str)
        rows.append(_scorecard_to_row(clip_name, scorecard))
        saved_clips.append(clip_name)

    csv_path, md_path = _write_summary(rows, baseline_dir)
    return saved_clips, csv_path, md_path


# Verdict columns worth flagging on ANY change, not just the overall one --
# a sub-verdict flip (e.g. calib_verdict PASS->DEGRADED) can hide behind an
# unchanged overall_verdict if another category was already the worse one.
_BASELINE_VERDICT_FIELDS = ['team_verdict', 'calib_verdict', 'camera_verdict',
                            'gk_verdict', 'overall_verdict']
# Percentage-point columns worth a tolerance-gated drift check.
_BASELINE_PCT_FIELDS = ['team_resolved_pct', 'calib_median_pct']


def _read_summary_csv(path):
    with open(path, newline='') as f:
        return {row['clip']: row for row in csv.DictReader(f)}


def _parse_pct(s):
    return float(str(s).strip().rstrip('%'))


def compare_to_baseline(new_csv_path, baseline_dir, tolerance=5.0):
    """Diff a new run's summary.csv against baseline_dir/summary.csv. Flags:
    - any of _BASELINE_VERDICT_FIELDS changing for a clip present in both
    - any of _BASELINE_PCT_FIELDS moving by more than `tolerance` points
    - a baseline clip missing from the new run entirely
    Clips in the new run but not the baseline are reported but NOT flagged
    (new dev/holdout clips aren't a regression). Returns the flag list."""
    baseline_csv = os.path.join(baseline_dir, 'summary.csv')
    if not os.path.exists(baseline_csv):
        raise SystemExit(f"No baseline found at {baseline_csv} -- build one first with --build-baseline")

    baseline_rows = _read_summary_csv(baseline_csv)
    new_rows = _read_summary_csv(new_csv_path)

    flags = []
    for clip, base_row in baseline_rows.items():
        if clip not in new_rows:
            flags.append((clip, 'missing_from_new_run', '', ''))
            continue
        new_row = new_rows[clip]
        for field in _BASELINE_VERDICT_FIELDS:
            old_v, new_v = base_row.get(field), new_row.get(field)
            if old_v != new_v:
                flags.append((clip, f'{field}_changed', old_v, new_v))
        for field in _BASELINE_PCT_FIELDS:
            try:
                old_pct, new_pct = _parse_pct(base_row[field]), _parse_pct(new_row[field])
            except (KeyError, ValueError):
                continue
            delta = new_pct - old_pct
            if abs(delta) > tolerance:
                flags.append((clip, f'{field}_moved_{delta:+.1f}pts', base_row[field], new_row[field]))

    new_clips = sorted(set(new_rows) - set(baseline_rows))

    print(f"\n{'=' * 70}\nBASELINE COMPARISON\n{'=' * 70}")
    print(f"  baseline: {baseline_csv}  ({len(baseline_rows)} clips)")
    print(f"  new run:  {new_csv_path}  ({len(new_rows)} clips)")
    print(f"  tolerance: {tolerance:.1f} points on {', '.join(_BASELINE_PCT_FIELDS)}")
    if new_clips:
        print(f"\n  clips in new run not in baseline (informational, not a regression):")
        for c in new_clips:
            print(f"    {c}")
    if not flags:
        print(f"\n  NO REGRESSIONS -- all {len(baseline_rows)} baseline clips match within tolerance.")
    else:
        print(f"\n  {len(flags)} FLAG(S):")
        for clip, reason, old, new in flags:
            print(f"    {clip:45s} {reason:35s} {old!s:>16s} -> {new!s}")
    return flags


def _print_console_table(rows):
    fields = ['clip', 'status', 'n_frames', 'team_resolved_pct', 'team_verdict',
              'calib_median_pct', 'calib_verdict', 'camera_verdict',
              'gk_margin', 'gk_verdict', 'id_swap_strong_per_100f',
              'overall_verdict']
    widths = {f: max(len(f), *(len(str(r.get(f, ''))) for r in rows)) if rows else len(f)
              for f in fields}
    header = '  '.join(f.ljust(widths[f]) for f in fields)
    print(header)
    print('-' * len(header))
    for r in rows:
        print('  '.join(str(r.get(f, '')).ljust(widths[f]) for f in fields))


_TIMING_STAGE_KEYS = ['detect_wall_sec', 'camera_wall_sec', 'calibrate_wall_sec',
                      'enrichment_wall_sec', 'ball_wall_sec', 'team_wall_sec', 'render_wall_sec']


def _print_and_save_pilot_extrapolation(scorecards, target_sizes, out_dir):
    """PILOT MODE: extrapolate full-batch wall-clock cost from a small
    number of real, end-to-end-processed windows. Only meaningful run on
    an otherwise-idle machine — see the --extrapolate-to help text and
    [[feedback_state_run_contamination]] in project memory; this
    deliberately does NOT run on a busy machine's numbers."""
    ok = [s for s in scorecards if s is not None]
    if not ok:
        print("\n[pilot] No successfully-processed clips to extrapolate from.")
        return

    totals = [s['processing_time_sec'] for s in ok]
    mean_total = float(np.mean(totals))
    stage_means = {}
    for key in _TIMING_STAGE_KEYS:
        vals = [(s.get('timing') or {}).get(key) for s in ok]
        vals = [v for v in vals if v is not None]
        stage_means[key] = float(np.mean(vals)) if vals else 0.0

    lines = []
    lines.append(f"PILOT EXTRAPOLATION — from {len(ok)} window(s): "
                 f"{', '.join(s['clip'] for s in ok)}")
    lines.append(f"  mean total wall time / window: {mean_total:.1f}s ({mean_total / 60:.1f} min)")
    lines.append("  stage breakdown (mean per window):")
    for key in _TIMING_STAGE_KEYS:
        lines.append(f"    {key:22s}: {stage_means[key]:8.1f}s")
    lines.append("")
    lines.append("  extrapolated full-batch cost:")
    extrapolations = {}
    for n in target_sizes:
        est_sec = mean_total * n
        extrapolations[n] = est_sec
        lines.append(f"    {n:4d} windows -> {est_sec:9.0f}s  "
                     f"({est_sec / 60:7.1f} min, {est_sec / 3600:5.2f} hr)")

    text = "\n".join(lines)
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)

    md_path = os.path.join(out_dir, 'pilot_extrapolation.md')
    json_path = os.path.join(out_dir, 'pilot_extrapolation.json')
    with open(md_path, 'w') as f:
        f.write("# Pilot extrapolation\n\n```\n" + text + "\n```\n")
    with open(json_path, 'w') as f:
        json.dump({
            'pilot_clips': [s['clip'] for s in ok],
            'n_pilot_windows': len(ok),
            'mean_total_wall_sec': mean_total,
            'stage_mean_wall_sec': stage_means,
            'extrapolated_sec_by_batch_size': {str(n): v for n, v in extrapolations.items()},
        }, f, indent=2)
    print(f"\nSaved pilot extrapolation: {md_path}, {json_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('video_folder', nargs='?', default=None,
                         help='Folder of video files to validate (not needed with '
                              '--build-baseline / --compare-to-baseline)')
    parser.add_argument('--max-frames', type=int, default=None,
                         help='Cap frames processed per clip (default: full clip)')
    parser.add_argument('--out-dir', default='batch_reports',
                         help='Output directory for scorecards/summary (default: batch_reports)')
    parser.add_argument('--force', action='store_true',
                         help='Ignore cached detect/camera/calibrate stubs, recompute')
    parser.add_argument('--no-render', dest='render_mode', action='store_const', const='none',
                         default='auto', help='Never auto-render, even for DEGRADED/FAIL clips')
    parser.add_argument('--render-all', dest='render_mode', action='store_const', const='all',
                         help='Render sample frames for every clip, not just DEGRADED/FAIL')
    parser.add_argument('--extrapolate-to', default=None,
                         help='PILOT MODE: comma-separated target batch sizes (e.g. "25,50") to '
                              'extrapolate full-batch cost to, from this run\'s own per-clip wall '
                              'times. Point video_folder at 2-3 representative clips on an '
                              'otherwise-idle machine for a real (uncontaminated) estimate — '
                              'timings from a busy/shared machine are not valid for sizing.')
    parser.add_argument('--build-baseline', metavar='REPORTS_DIR', default=None,
                         help='Snapshot every scorecard.json under REPORTS_DIR into '
                              '--baseline-dir as a committed golden baseline, then exit '
                              '(no video processing). E.g. --build-baseline batch_reports')
    parser.add_argument('--compare-to-baseline', metavar='SUMMARY_CSV', default=None,
                         help='Diff SUMMARY_CSV (a prior run\'s summary.csv) against '
                              '--baseline-dir/summary.csv, flag verdict changes or '
                              'team_resolved_pct/calib_median_pct drift beyond '
                              '--regression-tolerance, then exit. Exits 1 if any flags.')
    parser.add_argument('--baseline-dir', default='batch_reports_baseline',
                         help='Baseline directory for --build-baseline / --compare-to-baseline '
                              '(default: batch_reports_baseline)')
    parser.add_argument('--regression-tolerance', type=float, default=5.0,
                         help='Percentage-point tolerance for team_resolved_pct/calib_median_pct '
                              'drift before --compare-to-baseline flags a clip (default: 5.0)')
    args = parser.parse_args()

    if args.build_baseline:
        saved_clips, csv_path, md_path = build_baseline(args.build_baseline, args.baseline_dir)
        print(f"Saved {len(saved_clips)} clip(s) to baseline: {args.baseline_dir}")
        for c in saved_clips:
            print(f"  {c}")
        print(f"\nBaseline summary: {csv_path}, {md_path}")
        return

    if args.compare_to_baseline:
        flags = compare_to_baseline(args.compare_to_baseline, args.baseline_dir,
                                     args.regression_tolerance)
        sys.exit(1 if flags else 0)

    if not args.video_folder:
        parser.error('video_folder is required unless --build-baseline or --compare-to-baseline is given')

    os.makedirs(STUB_ROOT, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    video_paths = _find_videos(args.video_folder)
    if not video_paths:
        print(f"No video files found in {args.video_folder}")
        return

    print(f"Found {len(video_paths)} video(s) in {args.video_folder}")

    rows = []
    all_scorecards = []
    for video_path in video_paths:
        clip_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n{'=' * 70}\nProcessing: {video_path}\n{'=' * 70}")
        try:
            scorecard = process_clip(video_path, args.max_frames, args.out_dir,
                                      force=args.force, render_mode=args.render_mode)
        except Exception as e:
            print(f"  [ERROR] {clip_name} failed: {e}")
            traceback.print_exc()
            rows.append({'clip': clip_name, 'status': 'ERROR', 'error': str(e)})
            continue

        clip_dir = os.path.join(args.out_dir, clip_name)
        os.makedirs(clip_dir, exist_ok=True)
        with open(os.path.join(clip_dir, 'scorecard.json'), 'w') as f:
            json.dump(scorecard, f, indent=2, default=str)

        print(f"\n  -- {clip_name} scorecard --")
        print(f"  video: {scorecard['video']['n_frames_processed']} frames @ "
              f"{scorecard['video']['fps']:.2f}fps, {scorecard['video']['resolution']}")
        print(f"  team:      resolved={scorecard['team']['resolved_pct']:.1%}  "
              f"sep={scorecard['team']['cluster_separation']}  "
              f"[{scorecard['team']['verdict']}]")
        print(f"  calib:     median={scorecard['calibration']['median_confidence']:.1%}  "
              f"[{scorecard['calibration']['verdict']}]")
        print(f"  camera:    spikes>15px={scorecard['camera_motion']['spike_frame_count_over_15px']}  "
              f"sustained_bursts={scorecard['camera_motion']['sustained_burst_events']}  "
              f"[{scorecard['camera_motion']['verdict']}]")
        print(f"  goalkeepers: {scorecard['goalkeepers']['margin_label']}  "
              f"[{scorecard['goalkeepers']['verdict']}]")
        print(f"  identity:  raw_tracks={scorecard['identity']['raw_track_count']}  "
              f"flagged={scorecard['identity']['n_flagged_candidates']} "
              f"(strong={scorecard['identity']['strong_signature']} "
              f"[{scorecard['identity']['strong_signature_per_100_frames']}/100f], "
              f"weak={scorecard['identity']['weak_signature']} "
              f"[{scorecard['identity']['weak_signature_per_100_frames']}/100f])")
        print(f"  ball:      raw={scorecard['ball']['raw_detection_pct']:.1%}  "
              f"gaps: median={scorecard['ball']['median_gap_sec']}s "
              f"p90={scorecard['ball']['p90_gap_sec']}s "
              f"max={scorecard['ball']['max_gap_sec']}s "
              f"(n={scorecard['ball']['n_gaps']}, >1s={scorecard['ball']['n_gaps_over_1s']}, "
              f">3s={scorecard['ball']['n_gaps_over_3s']})")
        print(f"             within 1s of real detection: "
              f"{scorecard['ball']['pct_frames_within_1s_of_real']:.1%}  "
              f"within 3s: {scorecard['ball']['pct_frames_within_3s_of_real']:.1%}")
        timing = scorecard.get('timing') or {}
        if timing:
            print(f"  timing:    detect_wall={timing.get('detect_wall_sec')}s  "
                  f"camera_wall={timing.get('camera_wall_sec')}s  "
                  f"calibrate_wall={timing.get('calibrate_wall_sec')}s")
            dft = timing.get('detect_frame_timing') or {}
            cft = timing.get('calibrate_frame_timing') or {}
            if dft:
                print(f"    detect/frame:    mean={dft.get('mean_sec'):.3f}s "
                      f"median={dft.get('median_sec'):.3f}s p95={dft.get('p95_sec'):.3f}s "
                      f"max={dft.get('max_sec'):.3f}s (n={dft.get('n')})")
            if cft:
                print(f"    calibrate/frame: mean={cft.get('mean_sec'):.3f}s "
                      f"median={cft.get('median_sec'):.3f}s p95={cft.get('p95_sec'):.3f}s "
                      f"max={cft.get('max_sec'):.3f}s (n={cft.get('n')})")
        print(f"  OVERALL:   {scorecard['overall_verdict']}"
              f"{'  (rendered sample frames)' if scorecard['rendered'] else ''}")

        rows.append(_scorecard_to_row(clip_name, scorecard))
        all_scorecards.append(scorecard)

    print(f"\n{'=' * 70}\nBATCH SUMMARY ({len(rows)} clip(s))\n{'=' * 70}")
    _print_console_table(rows)
    csv_path, md_path = _write_summary(rows, args.out_dir)
    print(f"\nSaved summary: {csv_path}, {md_path}")

    if args.extrapolate_to:
        target_sizes = [int(x) for x in args.extrapolate_to.split(',') if x.strip()]
        _print_and_save_pilot_extrapolation(all_scorecards, target_sizes, args.out_dir)


if __name__ == '__main__':
    main()
