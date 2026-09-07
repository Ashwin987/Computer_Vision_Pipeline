"""
run_cv_analysis.py — standalone subprocess wrapper around the main.py CV
pipeline (detection/tracking, two-pass team resolution, goalkeeper
fragmentation-merge, camera-motion correction, calibration-confidence
speed tagging, referee exclusion, all 6 render outputs) for ONE video
segment at a time.

Designed to be invoked as a subprocess by an external caller (e.g. an AI
Tactical Analyzer) that polls status.json for progress and reads
stats.json once status.json says "complete" — but every piece of that
protocol is exercised by running this file directly, so it's fully
testable standalone right now.

Usage:
    python run_cv_analysis.py --video <path_to_segment> --output-dir <dir> [--match-name <name>]

    --output-dir is the BASE directory (default: output_videos); the
    actual outputs land in <output-dir>/<match-name>/, following the
    per-match folder convention in output_videos/README.txt. --match-name
    defaults to a folder-safe slug of the video's filename.

Ball-fallback note: unlike main.py's cached, stub-driven runs, tracking
here ALWAYS forces a fresh detection pass (read_from_stub=False) — the
tiled ball-fallback pass only ever triggers inside get_object_tracks'
fresh-detection branch (see trackers/tracker.py), so reading from a stub
would silently skip it. Affordable here specifically because this
pipeline targets short (~30s) segments, not full matches — see
ball_fallback.py.

Never lets an exception exit without writing status.json — every stage
of main() below the initial arg/path setup runs inside one try/except
that writes {"status": "error", ...} on any failure before re-raising.
"""
import argparse
import json
import os
import pickle
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone

import cv2
import numpy as np

MODEL_PATH             = 'models/best.pt'
CALIBRATOR_MODEL_PATH  = 'pose/pitch_keypoints_v3/weights/best.pt'

# (stage_id, human label) — order is the order main() executes them in.
# stage_num/total_stages in status.json are derived from this list's
# length/position, not hardcoded, so adding/removing a stage can't drift
# out of sync with what's reported.
STAGES = [
    ('init',                      'Reading video'),
    ('calibration',               'Pitch calibration (homography)'),
    ('tracking',                  'Detecting & tracking players/ball (incl. tiled ball-fallback pass)'),
    ('camera_motion',             'Estimating camera movement'),
    ('position_transform',        'Transforming positions to pitch coordinates'),
    ('speed_distance',            'Computing player speed & distance'),
    ('team_resolution',           'Resolving team colors (two-pass)'),
    ('team_fallback_merge',       'Fallback assignment + track-fragment merge'),
    ('ball_possession',           'Assigning ball possession'),
    ('match_events',              'Detecting match events'),
    ('tactical_events',           'Computing & ranking tactical events'),
    ('render_main_video',         'Rendering main tracking video'),
    ('render_tactical_carousel',  'Rendering tactical-events carousel'),
    ('render_stamina_panel',      'Rendering player stamina panel'),
    ('render_tactical_map',       'Rendering per-player tactical map'),
    ('render_pitch_heatmap',      'Rendering team pitch-control heatmap'),
    ('render_movement_trails',    'Rendering movement trails'),
    ('stats',                     'Computing summary stats'),
    ('finalize',                  'Finalizing outputs'),
]
STAGE_LABEL = dict(STAGES)
STAGE_NUM   = {sid: i + 1 for i, (sid, _) in enumerate(STAGES)}
TOTAL_STAGES = len(STAGES)

# Content -> final filename mapping. render_output*.py all write to fixed
# 'output_videos/outputN.avi' paths (no output-dir parameter exists on
# any of them), so this pipeline renders into that shared root like
# main.py always has, then moves+renames into the match folder as a
# final step — same rename-after-render pattern main.py itself already
# uses for its own "shorter_"-prefixed outputs. NOTE: because the
# intermediate filenames are shared/global, two run_cv_analysis.py
# invocations must not be run concurrently against the same repo working
# directory (fine for the current one-video-at-a-time subprocess usage;
# would need render_output*.py to accept an output path to parallelize).
_RENAME_MAP = [
    ('output_videos/output_video.avi', 'output1.avi'),   # main tracking video
    ('output_videos/output1.avi',      'output2.avi'),   # tactical-events carousel (render_output1.py)
    ('output_videos/output2.avi',      'output3.avi'),   # player stamina panel (render_output2.py)
    ('output_videos/output3.avi',      'output4.avi'),   # per-player tactical map (render_output3.py)
    ('output_videos/output4.avi',      'output5.avi'),   # team pitch-control heatmap (render_output4.py)
    ('output_videos/output6.avi',      'output6.avi'),   # movement trails (render_output6.py)
]


def _now():
    return datetime.now(timezone.utc).isoformat()


def slugify(name):
    """Folder-safe match name derived from a video path's filename."""
    name = os.path.splitext(os.path.basename(name))[0]
    name = name.lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name or 'match'


class StatusWriter:
    """Writes status.json atomically (tmp file + os.replace) so a polling
    caller never observes a half-written JSON file mid-write."""

    # Throttle for update_substage: write at most once per this many
    # seconds OR this many frames_done, whichever comes first (plus
    # always on force=True) -- frequent enough to tell "alive" from
    # "hung" on the two genuinely slow stages this exists for
    # (calibration ~2-5s/frame, ball-fallback ~8-22s/frame), without
    # adding meaningful I/O overhead on any stage that processes frames
    # faster than that.
    SUBSTAGE_MIN_INTERVAL_SEC = 3.0
    SUBSTAGE_MIN_FRAME_STEP = 10

    def __init__(self, path):
        self.path = path
        self.run_started_at = _now()
        self._current_stage_id = None
        self._last_substage_write_t = 0.0
        self._last_substage_frames_done = None

    def _write(self, data):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)

    def start_stage(self, stage_id):
        self._current_stage_id = stage_id
        self._last_substage_write_t = 0.0
        self._last_substage_frames_done = None
        self._write({
            'status':       'running',
            'stage':        stage_id,
            'stage_num':    STAGE_NUM[stage_id],
            'total_stages': TOTAL_STAGES,
            'label':        STAGE_LABEL[stage_id],
            'started_at':   self.run_started_at,
            'updated_at':   _now(),
        })
        print(f"[{STAGE_NUM[stage_id]}/{TOTAL_STAGES}] {STAGE_LABEL[stage_id]}", flush=True)

    def update_substage(self, frames_done, frames_total, force=False):
        """Adds a 'substage' block to the current stage's status.json
        entry: {frames_done, frames_total, pct, last_updated}. Must be
        called after start_stage (uses its stage_id/label/stage_num).
        Throttled per SUBSTAGE_MIN_INTERVAL_SEC/SUBSTAGE_MIN_FRAME_STEP
        above unless force=True (pass force=True on the final update of a
        stage so pct reliably reaches exactly 100 instead of stopping one
        throttle-interval short)."""
        now = time.time()
        enough_time = (now - self._last_substage_write_t) >= self.SUBSTAGE_MIN_INTERVAL_SEC
        enough_frames = (self._last_substage_frames_done is None or
                         (frames_done - self._last_substage_frames_done) >= self.SUBSTAGE_MIN_FRAME_STEP)
        if not (force or enough_time or enough_frames):
            return
        self._last_substage_write_t = now
        self._last_substage_frames_done = frames_done
        stage_id = self._current_stage_id
        pct = round(frames_done / frames_total * 100, 1) if frames_total else 0.0
        self._write({
            'status':       'running',
            'stage':        stage_id,
            'stage_num':    STAGE_NUM[stage_id],
            'total_stages': TOTAL_STAGES,
            'label':        STAGE_LABEL[stage_id],
            'started_at':   self.run_started_at,
            'updated_at':   _now(),
            'substage': {
                'frames_done':  frames_done,
                'frames_total': frames_total,
                'pct':          pct,
                'last_updated': _now(),
            },
        })

    def complete(self, outputs, stats_file):
        self._write({
            'status':       'complete',
            'stage_num':    TOTAL_STAGES,
            'total_stages': TOTAL_STAGES,
            'started_at':   self.run_started_at,
            'updated_at':   _now(),
            'outputs':      outputs,
            'stats_file':   stats_file,
        })

    def error(self, message, stage_id=None):
        self._write({
            'status':     'error',
            'stage':      stage_id,
            'message':    message,
            'started_at': self.run_started_at,
            'updated_at': _now(),
        })


def _smooth_positions(tracks, window=1):
    """Verbatim copy of main.py's helper — moving-average over
    position_transformed. window=1 (main.py's current default call) is a
    no-op smoothing-wise; kept parameterized for parity with main.py
    rather than special-cased away."""
    from collections import defaultdict
    half = window // 2
    for obj in tracks:
        if obj in ('ball', 'referees'):
            continue
        pos_by_id = defaultdict(dict)
        for fn, frame in enumerate(tracks[obj]):
            for tid, info in frame.items():
                p = info.get('position_transformed')
                if p is not None:
                    pos_by_id[tid][fn] = p
        for tid, pos_dict in pos_by_id.items():
            fn_set = set(pos_dict)
            for fn in list(fn_set):
                nbrs = [pos_dict[f] for f in range(fn - half, fn + half + 1)
                        if f in fn_set]
                if nbrs:
                    tracks[obj][fn][tid]['position_transformed'] = [
                        sum(p[0] for p in nbrs) / len(nbrs),
                        sum(p[1] for p in nbrs) / len(nbrs),
                    ]


# ── stats.json builders ─────────────────────────────────────────────────

def _build_calibration_stats(calibration_confidence_per_frame):
    vals = list(calibration_confidence_per_frame.values())
    n = len(vals)
    low  = sum(1 for v in vals if v < 0.70)
    med  = sum(1 for v in vals if 0.70 <= v < 0.90)
    high = sum(1 for v in vals if v >= 0.90)
    return {
        'frames':           n,
        'mean_confidence':  round(sum(vals) / n, 4) if n else None,
        'frames_high':      high,
        'frames_medium':    med,
        'frames_low':       low,
        'pct_high':         round(high / n * 100, 1) if n else None,
        'pct_medium':       round(med / n * 100, 1) if n else None,
        'pct_low':          round(low / n * 100, 1) if n else None,
    }


def _build_ball_stats(ball_pre_interp, ball_fallback, n_frames):
    """ball_pre_interp must be captured BEFORE
    Tracker.interpolate_ball_positions runs -- that call rebuilds every
    frame's ball dict as bare {1: {"bbox": x}}, unconditionally dropping
    the 'fallback' flag (see trackers/tracker.py's interpolate_ball_positions),
    so reading it from the post-interpolation tracks would make every
    frame look like a primary hit."""
    primary_hits  = sum(1 for b in ball_pre_interp if b.get(1) is not None and not b[1].get('fallback'))
    fallback_hits = sum(1 for b in ball_pre_interp if b.get(1) is not None and b[1].get('fallback'))
    final_hits    = primary_hits + fallback_hits
    still_missed  = n_frames - final_hits
    fb = ball_fallback.summary() if ball_fallback is not None else {
        'n_triggered': 0, 'n_recovered': 0, 'recovery_rate': None,
        'total_fallback_sec': 0.0,
    }
    return {
        'n_frames':                    n_frames,
        'primary_hits':                primary_hits,
        'primary_detection_rate_pct':  round(primary_hits / n_frames * 100, 1) if n_frames else None,
        'fallback_triggered':          fb['n_triggered'],
        'fallback_recovered':          fb['n_recovered'],
        'fallback_recovery_rate_pct':  round(fb['recovery_rate'] * 100, 1) if fb['recovery_rate'] is not None else None,
        'fallback_total_sec':          round(fb['total_fallback_sec'], 1),
        'final_hits_combined':         final_hits,
        'final_detection_rate_pct':    round(final_hits / n_frames * 100, 1) if n_frames else None,
        'still_missed_after_fallback': still_missed,
    }


def _build_team_resolution_stats(team_assigner, tracks, n_merged_pairs):
    pid_team = {}
    for frame in tracks['players']:
        for pid, info in frame.items():
            pid_team[pid] = info.get('team', 0)
    total = len(pid_team)
    n1  = sum(1 for t in pid_team.values() if t == 1)
    n2  = sum(1 for t in pid_team.values() if t == 2)
    ngk = sum(1 for t in pid_team.values() if t == 3)
    nun = total - n1 - n2 - ngk
    resolved = n1 + n2 + ngk

    def _color(team_id):
        c = team_assigner.locked_colors.get(team_id)
        return [round(float(v), 1) for v in c] if c is not None else None

    return {
        'total_tracked_players':   total,
        'resolved_team1':          n1,
        'resolved_team2':          n2,
        'goalkeepers':             ngk,
        'unresolved':              nun,
        'resolution_rate_pct':     round(resolved / total * 100, 1) if total else None,
        'fallback_assigned_count': len(team_assigner.fallback_assigned),
        'merged_fragment_pairs':   n_merged_pairs,
        'team_colors_bgr':         {'team1': _color(1), 'team2': _color(2)},
    }


def _build_player_stats(tracks):
    per_player = {}
    for frame in tracks['players']:
        for pid, info in frame.items():
            team = info.get('team', 0)
            if team == 0:
                continue   # never confidently resolved -- not a real dashboard row
            p = per_player.setdefault(pid, {
                'team': team, 'top_speed_kmh': 0.0, 'top_speed_confidence': 'high',
                'total_distance_m': 0.0, 'frames_tracked': 0, 'speed_sum': 0.0,
            })
            p['frames_tracked'] += 1
            speed = info.get('speed') or 0.0
            p['speed_sum'] += speed
            if speed > p['top_speed_kmh']:
                p['top_speed_kmh'] = speed
                p['top_speed_confidence'] = info.get('speed_confidence', 'high')
            dist = info.get('distance') or 0.0
            if dist > p['total_distance_m']:   # cumulative -- max seen so far = latest
                p['total_distance_m'] = dist

    players = []
    for pid, p in per_player.items():
        avg_speed = p['speed_sum'] / p['frames_tracked'] if p['frames_tracked'] else 0.0
        players.append({
            'player_id':            int(pid),
            'team':                 int(p['team']),
            'top_speed_kmh':        round(float(p['top_speed_kmh']), 2),
            'top_speed_confidence': p['top_speed_confidence'],
            'avg_speed_kmh':        round(float(avg_speed), 2),
            'total_distance_m':     round(float(p['total_distance_m']), 1),
            'frames_tracked':       p['frames_tracked'],
        })
    players.sort(key=lambda r: -r['top_speed_kmh'])
    return players


def _build_match_events_stats(match_events):
    counts = Counter(e['type'] for e in match_events)
    return {'counts': dict(counts), 'total': len(match_events)}


def _build_tactical_events_stats(tactical_events_detector, ranked_windows, top_n=10):
    counts = dict(tactical_events_detector.event_counts)
    total = sum(counts.values())
    all_events = []
    for widx, evs in ranked_windows.items():
        for ev in evs:
            e = dict(ev)
            e['window'] = widx
            all_events.append(e)
    all_events.sort(key=lambda e: e.get('score', 0), reverse=True)
    highlights = [{
        'type':      e.get('type'),
        'player_id': e.get('player_id'),
        'window':    e.get('window'),
        'score':     round(float(e.get('score', 0)), 2),
        'intensity': round(float(e.get('intensity', 0)), 2),
        'metric':    e.get('metric'),
    } for e in all_events[:top_n]]
    return {'counts': counts, 'total': total, 'highlights': highlights}


def build_stats(*, match_name, video_path, n_frames, fps,
                 calibration_confidence_per_frame, tracks, ball_pre_interp, ball_fallback,
                 team_assigner, n_merged_pairs, match_events,
                 tactical_events_detector, ranked_windows, n_transitions):
    return {
        'match_name':    match_name,
        'video_path':    video_path,
        'generated_at':  _now(),
        'video': {
            'n_frames':     n_frames,
            'fps':          fps,
            'duration_sec': round(n_frames / fps, 1) if fps else None,
        },
        'calibration':     _build_calibration_stats(calibration_confidence_per_frame),
        'ball':             _build_ball_stats(ball_pre_interp, ball_fallback, n_frames),
        'team_resolution':  _build_team_resolution_stats(team_assigner, tracks, n_merged_pairs),
        'players':          _build_player_stats(tracks),
        'match_events':     _build_match_events_stats(match_events),
        'tactical_events':  _build_tactical_events_stats(tactical_events_detector, ranked_windows),
        'transitions':      {'count': n_transitions},
    }


# ── pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(video_path, final_dir, match_name, status):
    from utils import read_video, save_video
    from trackers import Tracker
    from team_assigner import TeamAssigner
    from player_ball_assigner import PlayerBallAssigner
    from camera_movement_estimator import CameraMovementEstimator
    from view_transformer import ViewTransformer
    from speed_and_distance_estimator import SpeedAndDistance_Estimator
    from match_events import MatchEventsDetector
    from transition_detector import TransitionDetector

    stub_prefix = f'stubs/cv_analysis_{match_name}'
    CAL_STUB   = f'{stub_prefix}_homography.pkl'
    CAM_STUB   = f'{stub_prefix}_camera_movement.pkl'
    TRACK_STUB = f'{stub_prefix}_tracks.pkl'
    os.makedirs('stubs', exist_ok=True)

    # ── init ──────────────────────────────────────────────────────────────
    status.start_stage('init')
    video_frames = read_video(video_path)
    n_frames = len(video_frames)
    _cap = cv2.VideoCapture(video_path)
    fps = float(_cap.get(cv2.CAP_PROP_FPS)) or 25.0
    _cap.release()
    print(f"[run_cv_analysis] {n_frames} frames @ {fps:.2f} fps")

    view_transformer = ViewTransformer()

    # ── calibration ───────────────────────────────────────────────────────
    status.start_stage('calibration')
    if os.path.exists(CAL_STUB):
        with open(CAL_STUB, 'rb') as f:
            homography_per_frame = pickle.load(f)
    else:
        from pitch_calibrator import SoccerNetCalibrator
        sc = SoccerNetCalibrator(CALIBRATOR_MODEL_PATH)

        _calib_state = {'done': 0}

        def _calibration_progress(sampled_done, sampled_total):
            _calib_state['done'] = sampled_done
            status.update_substage(sampled_done, sampled_total)

        homography_per_frame = sc.calibrate_video(video_path, progress_callback=_calibration_progress)
        # Force a final write with frames_total pinned to the actual done
        # count -- guarantees pct reaches exactly 100 even if the video's
        # own frame-count metadata (used to estimate sampled_total up
        # front) was slightly off.
        status.update_substage(_calib_state['done'], _calib_state['done'], force=True)
        with open(CAL_STUB, 'wb') as f:
            pickle.dump(homography_per_frame, f)
    _fb_H     = view_transformer.persepctive_trasnformer.astype(np.float32)
    _fb_H_inv = np.linalg.inv(_fb_H).astype(np.float32)
    homography_per_frame = {
        fn: h if h is not None else (_fb_H, _fb_H_inv)
        for fn, h in homography_per_frame.items()
    }

    # ── tracking (+ mandatory tiled ball-fallback) ───────────────────────────
    status.start_stage('tracking')
    # use_ball_fallback=True + read_from_stub=False (always, regardless of
    # whether TRACK_STUB already exists from a prior run of this same
    # video/match_name) -- the fallback call site inside get_object_tracks
    # only ever runs in the fresh-detection branch, so reading from a stub
    # would silently skip the stage this function exists to guarantee.
    tracker = Tracker(MODEL_PATH, use_ball_fallback=True)

    def _tracking_progress(frame_num, total_frames):
        fb = tracker.ball_fallback
        if fb is None or fb.n_triggered == 0:
            return   # nothing triggered yet -- no meaningful fallback substage to report
        is_final = frame_num >= total_frames - 1
        if is_final:
            frames_total = fb.n_triggered   # exact, known for certain now
        else:
            # Total trigger count isn't knowable until every frame's primary
            # detection has run -- estimate it by extrapolating the trigger
            # rate observed so far across the whole clip, same idea tqdm
            # uses for its own ETA. Self-corrects as frame_num climbs and is
            # pinned to the exact count on the last frame (is_final above).
            frac_done = max((frame_num + 1) / total_frames, 1e-6)
            frames_total = max(fb.n_triggered, round(fb.n_triggered / frac_done))
        status.update_substage(fb.n_triggered, frames_total, force=is_final)

    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=False, stub_path=TRACK_STUB, video_path=video_path,
        progress_callback=_tracking_progress)
    tracker.add_position_to_tracks(tracks)

    # ── camera movement ──────────────────────────────────────────────────
    status.start_stage('camera_motion')
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames, read_from_stub=os.path.exists(CAM_STUB), stub_path=CAM_STUB)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # ── position transform + calibration confidence ──────────────────────
    status.start_stage('position_transform')
    view_transformer.add_transformed_position_to_tracks(tracks, homography_per_frame)
    _smooth_positions(tracks, window=1)
    calibration_confidence_per_frame = view_transformer.compute_frame_confidence(tracks)
    # Snapshot BEFORE interpolate_ball_positions -- it rebuilds every frame's
    # ball dict from scratch and drops the 'fallback' flag unconditionally
    # (see _build_ball_stats' docstring), so this is the only point where
    # primary-vs-fallback-vs-still-missed can still be told apart.
    ball_pre_interp = tracks['ball']
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # ── speed / distance ─────────────────────────────────────────────────
    status.start_stage('speed_distance')
    speed_and_distance_estimator = SpeedAndDistance_Estimator(fps=fps)
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(
        tracks, calibration_confidence_per_frame=calibration_confidence_per_frame)

    # ── team resolution (two-pass, referee-color-distance exclusion) ────────
    status.start_stage('team_resolution')
    team_assigner = TeamAssigner()
    team_assigner.resolve_all_teams(video_frames, tracks['players'], tracks['referees'])

    # ── fallback assignment + fragment merge (incl. goalkeeper fragments) ───
    status.start_stage('team_fallback_merge')
    team_assigner.finalize_fallback_assignments()
    merged_pairs, _rejected_pairs = team_assigner.merge_fragmented_tracks(
        video_frames, tracks['players'], exclude_pairs=list(tracker.applied_splits.items()))
    team_assigner.apply_final_teams(tracks['players'])

    # ── ball possession ──────────────────────────────────────────────────
    status.start_stage('ball_possession')
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
    team_ball_control = np.array(team_ball_control)

    # ── match events ──────────────────────────────────────────────────────
    status.start_stage('match_events')
    events_detector = MatchEventsDetector(fps=fps)
    match_events = events_detector.detect_events(tracks)

    # ── tactical events ───────────────────────────────────────────────────
    status.start_stage('tactical_events')
    from render_output3 import STEP as _tac_step, _FALLBACK_PITCH_VERTS as _tac_fallback_verts
    from tactical_events.space_control import (
        build_pitch_mask, build_sampling_grid, compute_space_control_per_frame,
        compute_pitch_verts_from_tracks,
    )
    from tactical_events.tactical_events_detector import TacticalEventsDetector
    from tactical_events.event_ranking import rank_events_by_window

    _tac_h, _tac_w = video_frames[0].shape[:2]
    _tac_pitch_verts = compute_pitch_verts_from_tracks(
        tracks, _tac_h, _tac_w, fallback_verts=_tac_fallback_verts)
    _tac_pitch_mask = build_pitch_mask(_tac_pitch_verts, _tac_h, _tac_w)
    _tac_grid, _tac_grid_rows, _tac_grid_cols, _tac_row_idx, _tac_col_idx = \
        build_sampling_grid(_tac_pitch_mask, _tac_step)
    space_control_per_frame = compute_space_control_per_frame(tracks, _tac_grid, top_frac=0.10)

    tactical_events_detector = TacticalEventsDetector(fps=fps)
    events_by_frame = tactical_events_detector.detect(tracks, space_control_per_frame)
    ranked_windows, window_frames = rank_events_by_window(
        events_by_frame, tracks, space_control_per_frame, fps=fps)

    # ── render: main tracking video ──────────────────────────────────────
    status.start_stage('render_main_video')
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame)
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(
        output_video_frames, tracks)
    output_video_frames = events_detector.draw_events(output_video_frames, match_events)

    transition_detector = TransitionDetector(fps=fps)
    transitions = transition_detector.detect_transitions(team_ball_control, tracks)
    output_video_frames = transition_detector.draw_transitions(output_video_frames, transitions)

    os.makedirs('output_videos', exist_ok=True)
    save_video(output_video_frames, 'output_videos/output_video.avi')

    original_w = video_frames[0].shape[1]
    annotated_frames = [frame[:, :original_w] for frame in output_video_frames]

    # ── render: remaining 5 outputs ──────────────────────────────────────
    status.start_stage('render_tactical_carousel')
    from render_output1 import render_output1
    render_output1(tracks, fps=fps, annotated_frames=annotated_frames,
                   ranked_windows=ranked_windows, window_frames=window_frames)

    status.start_stage('render_stamina_panel')
    from render_output2 import render_output2
    render_output2(output_video_frames, tracks, team_ball_control, fps=fps,
                   homography_per_frame=homography_per_frame)

    status.start_stage('render_tactical_map')
    from render_output3 import render_output3
    render_output3(video_frames, tracks, team_ball_control, transitions, fps=fps,
                   homography_per_frame=homography_per_frame)

    status.start_stage('render_pitch_heatmap')
    from render_output4 import render_output4
    render_output4(video_frames, tracks, team_ball_control, view_transformer,
                   fps=fps, annotated_frames=annotated_frames,
                   homography_per_frame=homography_per_frame)

    status.start_stage('render_movement_trails')
    from render_output6 import render_output6
    render_output6(video_frames, tracks, team_ball_control, view_transformer,
                   fps=fps, annotated_frames=annotated_frames,
                   homography_per_frame=homography_per_frame,
                   camera_movement_per_frame=camera_movement_per_frame)

    # ── stats ─────────────────────────────────────────────────────────────
    status.start_stage('stats')
    stats = build_stats(
        match_name=match_name, video_path=video_path, n_frames=n_frames, fps=fps,
        calibration_confidence_per_frame=calibration_confidence_per_frame,
        tracks=tracks, ball_pre_interp=ball_pre_interp, ball_fallback=tracker.ball_fallback,
        team_assigner=team_assigner,
        n_merged_pairs=len(merged_pairs), match_events=match_events,
        tactical_events_detector=tactical_events_detector, ranked_windows=ranked_windows,
        n_transitions=len(transitions),
    )
    stats_path = os.path.join(final_dir, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    # ── finalize: move rendered outputs into the match folder ───────────────
    status.start_stage('finalize')
    outputs = {}
    for src, dst_name in _RENAME_MAP:
        dst = os.path.join(final_dir, dst_name)
        if os.path.exists(src):
            os.replace(src, dst)
            outputs[dst_name] = dst
        else:
            print(f"[run_cv_analysis] WARNING: expected render output missing: {src}")

    return outputs, stats_path


def main():
    parser = argparse.ArgumentParser(
        description='Standalone CV analysis pipeline wrapper (subprocess-callable).')
    parser.add_argument('--video', required=True, help='Path to the video segment to analyze')
    parser.add_argument('--output-dir', default='output_videos',
                        help='Base output directory; a <match-name> subfolder is created '
                             'inside it, per output_videos/README.txt (default: output_videos)')
    parser.add_argument('--match-name', default=None,
                        help='Folder-safe match name; derived from the video filename if omitted')
    args = parser.parse_args()

    match_name = args.match_name or slugify(args.video)
    final_dir = os.path.join(args.output_dir, match_name)
    os.makedirs(final_dir, exist_ok=True)

    status = StatusWriter(os.path.join(final_dir, 'status.json'))

    if not os.path.exists(args.video):
        status.error(f'Video file not found: {args.video}')
        print(f"ERROR: video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    try:
        t0 = time.time()
        outputs, stats_path = run_pipeline(args.video, final_dir, match_name, status)
        status.complete(outputs, stats_path)
        print(f"[run_cv_analysis] done in {time.time() - t0:.1f}s -> {final_dir}")
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        status.error(f'{type(e).__name__}: {e}\n{tb}')
        sys.exit(1)


if __name__ == '__main__':
    main()
