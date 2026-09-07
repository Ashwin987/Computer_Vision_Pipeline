"""
tactical_events/event_ranking.py

Selection/ranking layer for the tactical-events panel. Buckets the raw
per-frame events already produced by TacticalEventsDetector into
20-second windows, scores each event as BASE_WEIGHT[type] + intensity
(0-5, how far the triggering measurement exceeded its threshold), and
keeps the top TOP_PER_WINDOW per window (sorted descending by score).

This does NOT change detection: tactical_events_detector.py's
thresholds, cooldowns, and rising-edge triggers are untouched. This
module only re-reads the already-computed tracks / space_control_per_frame
data at each event's (frame, player_id) to score it — it re-derives a
raw magnitude (e.g. the speed at that frame), not the detection logic
itself (no run-counters, no rising-edge state, no re-thresholding).

Called ONCE from main.py, after TacticalEventsDetector.detect() and
before any rendering. render_output1.py's carousel display just looks
up the current 20-second window's already-ranked batch.
"""

import numpy as np
from collections import defaultdict

from tactical_events.tactical_events_detector import (
    SPRINT_KMH, BURST_DELTA_KMH, BURST_WINDOW_SEC, RECOVERY_KMH,
    LATERAL_DIST_M, LATERAL_WINDOW_SEC, DROP_DIST_M, DROP_WINDOW_SEC,
    BREAK_KMH, ISOLATED_DIST_M, PRESS_DIST_M, PRESS_MIN_KMH, OVERLAP_MIN_KMH,
)

WINDOW_SEC      = 20.0
TOP_PER_WINDOW  = 10
MAX_PER_TYPE    = 3     # diversity cap: no more than this many of one event type per window

BASE_WEIGHT = {
    'BREAK':    10,
    'ISOLATED': 9,
    'SPACE':    8,
    'PRESS':    8,
    'OVERLAP':  7,
    'LATERAL_RUN': 6,
    'RECOVERY': 6,
    'SPRINT':   5,
    'DROP':     4,
    'BURST':    3,
}

# Linear 0-5 intensity ramp per type: value<=threshold -> 0, value>=max_ref -> 5.
_SPRINT_MAX      = 34.0   # km/h  (27 km/h threshold -> low, 34 -> near max)
_RECOVERY_MAX    = 27.0   # km/h  (20 km/h threshold)
_BREAK_MAX       = 27.0   # km/h  (20 km/h threshold)
_BURST_MAX       = 25.0   # km/h delta (15 km/h threshold)
_ISOLATED_MAX    = 45.0   # metres (25m threshold -> low, 45m -> near max)
_LATERAL_MAX     = 35.0   # metres (20m threshold)
_DROP_MAX        = 30.0   # metres (15m threshold)
_SPACE_MAX_RATIO = 4.0    # player's share of the pitch vs. the average share
_OVERLAP_MAX     = 25.0   # km/h  (15 km/h threshold)


def _ramp(value, threshold, max_ref):
    """Linear 0-5 ramp: value<=threshold -> 0.0, value>=max_ref -> 5.0."""
    if max_ref <= threshold:
        return 0.0
    frac = (value - threshold) / (max_ref - threshold)
    return float(np.clip(frac, 0.0, 1.0) * 5.0)


def _pos_at(tracks, fn, pid):
    if fn < 0 or fn >= len(tracks['players']):
        return None
    info = tracks['players'][fn].get(pid)
    if not info:
        return None
    p = info.get('position_transformed')
    if p is None:
        return None
    try:
        return (float(p[0]), float(p[1]))
    except Exception:
        return None


def _intensity_and_metric(ev, tracks, space_control_per_frame, fps):
    """Return (intensity 0-5, human-readable metric string) for one event."""
    fn, pid, etype = ev['frame'], ev['player_id'], ev['type']
    info = tracks['players'][fn].get(pid, {}) if fn < len(tracks['players']) else {}
    speed = info.get('speed')
    speed = float(speed) if speed is not None else 0.0
    pos_now = _pos_at(tracks, fn, pid)

    if etype == 'SPRINT':
        return _ramp(speed, SPRINT_KMH, _SPRINT_MAX), f"{speed:.1f} km/h"

    if etype == 'RECOVERY':
        return _ramp(speed, RECOVERY_KMH, _RECOVERY_MAX), f"{speed:.1f} km/h"

    if etype == 'BREAK':
        return _ramp(speed, BREAK_KMH, _BREAK_MAX), f"{speed:.1f} km/h"

    if etype == 'BURST':
        w = max(1, int(round(BURST_WINDOW_SEC * fps)))
        info0 = tracks['players'][fn - w].get(pid) if fn - w >= 0 else None
        s0 = info0.get('speed') if info0 else None
        delta = speed - float(s0) if s0 is not None else 0.0
        return _ramp(delta, BURST_DELTA_KMH, _BURST_MAX), f"+{delta:.1f} km/h"

    if etype == 'ISOLATED':
        team = info.get('team', 0)
        best = None
        for pid2, info2 in tracks['players'][fn].items():
            if pid2 == pid or info2.get('team', 0) != team:
                continue
            p2 = info2.get('position_transformed')
            if p2 is None or pos_now is None:
                continue
            try:
                d = np.hypot(pos_now[0] - float(p2[0]), pos_now[1] - float(p2[1]))
            except Exception:
                continue
            if best is None or d < best:
                best = d
        best = best if best is not None else ISOLATED_DIST_M
        return _ramp(best, ISOLATED_DIST_M, _ISOLATED_MAX), f"{best:.1f}m to nearest teammate"

    if etype == 'SPACE':
        sc = space_control_per_frame[fn].get(pid) if fn < len(space_control_per_frame) else None
        n_active = len(space_control_per_frame[fn]) if fn < len(space_control_per_frame) else 1
        frac = sc['frac'] if sc else 0.0
        avg = 1.0 / n_active if n_active else 1.0
        ratio = (frac / avg) if avg > 0 else 1.0
        return _ramp(ratio, 1.0, _SPACE_MAX_RATIO), f"{ratio:.2f}x avg space"

    if etype == 'LATERAL_RUN':
        w = max(1, int(round(LATERAL_WINDOW_SEC * fps)))
        p0 = _pos_at(tracks, fn - w, pid)
        dist = abs(pos_now[1] - p0[1]) if (pos_now is not None and p0 is not None) else LATERAL_DIST_M
        return _ramp(dist, LATERAL_DIST_M, _LATERAL_MAX), f"{dist:.1f}m lateral"

    if etype == 'DROP':
        w = max(1, int(round(DROP_WINDOW_SEC * fps)))
        p0 = _pos_at(tracks, fn - w, pid)
        dist = abs(p0[0] - pos_now[0]) if (pos_now is not None and p0 is not None) else 0.0
        return _ramp(dist, DROP_DIST_M, _DROP_MAX), f"{dist:.1f}m backward"

    if etype == 'PRESS':
        # opp_best/opp_best_pid: the (nearest) opponent this player is
        # closing down, matching the detector's own per-opponent grouping.
        # "N pressers" must then count CO-PRESSING TEAMMATES against that
        # SAME opponent (the detector's actual trigger condition), not
        # nearby opponents — counting opponents here previously produced a
        # display metric that didn't match what actually triggered PRESS.
        team = info.get('team', 0)
        opp_best = None
        opp_best_pid = None
        for pid2, info2 in tracks['players'][fn].items():
            if info2.get('team', 0) == team or info2.get('team', 0) not in (1, 2):
                continue
            p2 = info2.get('position_transformed')
            if p2 is None or pos_now is None:
                continue
            try:
                d = np.hypot(pos_now[0] - float(p2[0]), pos_now[1] - float(p2[1]))
            except Exception:
                continue
            if d <= PRESS_DIST_M and (opp_best is None or d < opp_best):
                opp_best = d
                opp_best_pid = pid2
        if opp_best is None:
            return 0.0, "press"

        opp_pos = tracks['players'][fn].get(opp_best_pid, {}).get('position_transformed')
        presser_count = 0
        if opp_pos is not None:
            for pid3, info3 in tracks['players'][fn].items():
                if info3.get('team', 0) != team:
                    continue
                p3 = info3.get('position_transformed')
                spd3 = info3.get('speed')
                if p3 is None or spd3 is None or spd3 < PRESS_MIN_KMH:
                    continue
                try:
                    d3 = np.hypot(float(p3[0]) - float(opp_pos[0]), float(p3[1]) - float(opp_pos[1]))
                except Exception:
                    continue
                if d3 <= PRESS_DIST_M:
                    presser_count += 1

        closing = PRESS_DIST_M - opp_best
        proximity_score = _ramp(closing, 0.0, PRESS_DIST_M)
        bonus = max(0, presser_count - 1) * 1.0
        intensity = float(np.clip(proximity_score + bonus, 0.0, 5.0))
        return intensity, f"{opp_best:.1f}m closed, {presser_count} pressers"

    if etype == 'OVERLAP':
        return _ramp(speed, OVERLAP_MIN_KMH, _OVERLAP_MAX), f"{speed:.1f} km/h run"

    return 0.0, ""


def rank_events_by_window(events_by_frame, tracks, space_control_per_frame, fps):
    """Run ONCE. Buckets every event into WINDOW_SEC-second windows, scores
    each as BASE_WEIGHT + intensity, then selects the top TOP_PER_WINDOW per
    window subject to a diversity cap: at most MAX_PER_TYPE of any single
    event type. Selection walks the score-sorted list top-down, skipping an
    event once its type has already reached the cap, so the instances kept
    per type are still its highest-scoring ones. The final batch is sorted
    descending by score (fewer than TOP_PER_WINDOW, or zero, is fine).

    Returns
    -------
    ranked_windows : dict window_idx -> list of scored-event dicts:
        {'frame', 'type', 'player_id', 'base_weight', 'intensity', 'score', 'metric'}
    window_frames  : int, frames per window (caller computes
                     window_idx = frame_num // window_frames)
    """
    window_frames = max(1, int(round(WINDOW_SEC * fps)))

    windows = defaultdict(list)
    for fn, evs in events_by_frame.items():
        for ev in evs:
            windows[fn // window_frames].append(ev)

    ranked_windows = {}
    for widx, evs in windows.items():
        scored = []
        for ev in evs:
            base = BASE_WEIGHT.get(ev['type'], 0)
            intensity, metric = _intensity_and_metric(ev, tracks, space_control_per_frame, fps)
            scored.append({
                **ev,
                'base_weight': base,
                'intensity': round(intensity, 2),
                'score': round(base + intensity, 2),
                'metric': metric,
            })
        scored.sort(key=lambda e: -e['score'])

        batch = []
        type_counts = defaultdict(int)
        for ev in scored:
            if len(batch) >= TOP_PER_WINDOW:
                break
            if type_counts[ev['type']] >= MAX_PER_TYPE:
                continue   # this type already has its MAX_PER_TYPE slots filled
            batch.append(ev)
            type_counts[ev['type']] += 1

        batch.sort(key=lambda e: -e['score'])
        ranked_windows[widx] = batch

    return ranked_windows, window_frames
