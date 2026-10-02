#!/usr/bin/env python3
"""
verify_evidence_citations.py — standalone, re-runnable auditor that checks
every evidence citation the dashboard shows against the actual underlying
data it claims to represent.

Why this exists: a real bug shipped where a Data Dashboard card read "PSG
Primary Zone: Attacking Third" while the evidence clip for that exact minute
showed PSG nowhere near the final third. The clip that exposed it has since
been removed from the UI (dashboard/app.py's _render_minute_evidence) -
showing one playable minute out of dozens implied the other minutes' claims
were just as trustworthy, which was never true. This script is the actual
fix: it doesn't need a person to watch a clip and notice something's off, it
directly recomputes or cross-checks every citation against raw numbers, for
every match, every time it's run.

Two independent audits, per curated match:

1. COACH REPORT — every tactical-event highlight (<=10 per match, each
   carrying a real `frame` number) has its stored metric/intensity
   re-derived FROM SCRATCH from the cached tracking stub (tracks.pkl +
   camera_movement.pkl + homography.pkl), using the exact same enrichment
   chain and the exact same event_ranking._intensity_and_metric() function
   that produced the original value. Also re-runs full event detection and
   checks the recomputed event COUNTS against stats.json's — if those
   don't match, the cached stub has drifted from stats.json and nothing
   downstream (frame numbers included) can be trusted without
   re-running backfill_tactical_event_frames_to_stats.py first.

2. DATA DASHBOARD — two checks against raw_data (what bundle.json's
   per-minute Gemini analysis is built from):
   a. GROUND-TRUTH ZONE CHECK, for every minute per match that has real CV
      tracking coverage - originally just the one 30s "peak momentum"
      window per match, now also every extra minute
      extend_cv_coverage.py has processed (read from each match's
      dashboard/curated_matches/<match>/_extra_cv_windows.json, if present):
      recomputes each team's real attacking direction from actual tracked
      player positions (the same mean-x heuristic tactical_events_detector.py
      already uses to tell BREAK/transition direction), buckets the real
      tracked ball position into pitch thirds relative to that direction,
      and compares the result to the ball_zone Gemini assigned that minute.
      This is a direct, data-grounded check of the exact directionality
      pattern asked for, for every minute real ground truth exists for.
   b. INTERNAL CONSISTENCY CHECK, every minute, both matches, no tracking
      data required: a minute whose ball_zone says a team was in their
      attacking_third should show a non-trivial amount of that same team's
      own attack_sec; one labeled defensive_third for them shouldn't show
      that team spending most of the minute "attacking" per their own
      number. Flags any minute where this relationship is clearly
      inverted — a proxy for the same directionality-bug class, run across
      every minute in both matches, not just the one with CV ground truth.

Mirrors (and must be kept in sync with — see each function's docstring for
the exact app.py line range mirrored) a few pure, side-effect-free pieces of
dashboard/app.py: compute_dashboard_df's derived columns and the Global
Control tab's Primary Zone mode-selection logic. Not imported directly
because dashboard/app.py runs Streamlit calls (st.set_page_config etc.) at
module import time and can't be imported outside a Streamlit run.

Usage:
    python verify_evidence_citations.py                       # both matches
    python verify_evidence_citations.py --match liverpool_psg
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from view_transformer.view_transformer import ViewTransformer
from speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistance_Estimator
from tactical_events.space_control import (
    build_pitch_mask, build_sampling_grid, compute_space_control_per_frame,
    compute_pitch_verts_from_tracks,
)
from tactical_events.tactical_events_detector import TacticalEventsDetector
from tactical_events.event_ranking import rank_events_by_window, _intensity_and_metric
from render_output3 import STEP as _TAC_STEP, _FALLBACK_PITCH_VERTS as _TAC_FALLBACK_VERTS
from utils.bbox_utils import get_center_of_bbox, get_foot_position

REPO_ROOT = Path(__file__).parent.parent
STUBS_DIR = Path(__file__).parent / "stubs"
CV_OUTPUT_DIR = Path(__file__).parent / "output_videos"
CLIPS_DIR = Path(__file__).parent / "extra_coverage_clips"
CURATED_DIR = REPO_ROOT / "dashboard" / "curated_matches"


def extra_windows_path(match_name):
    return CURATED_DIR / match_name / "_extra_cv_windows.json"


def load_extra_windows(match_name):
    """Extra minutes extend_cv_coverage.py has processed beyond the
    original single peak-momentum window, if any - see that script's
    docstring for how/why these get added."""
    p = extra_windows_path(match_name)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

PITCH_X_MAX = 105.0   # same convention as view_transformer.py / tactical_events_detector.py
THIRD = PITCH_X_MAX / 3.0

# Internal-consistency heuristic thresholds (Data Dashboard check 2b) — a
# minute out of 60s. Not a hard physical law, just a sanity bound: a team
# credited with 60 - ATTACK_SEC_LOW seconds or more of NOT attacking can't
# plausibly also be the team whose ball_zone says they spent the minute in
# the opponent's third with the ball; the inverse for defensive_third.
ATTACK_SEC_LOW_THRESHOLD = 10.0
ATTACK_SEC_HIGH_THRESHOLD = 45.0

MATCHES = [
    {
        "name": "liverpool_psg",
        "cv_name": "liverpool_psg_verified",
    },
    {
        "name": "barca_madrid_pt1",
        "cv_name": "barca_madrid_pt1_verified",
    },
]


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


def load_enriched(cv_name, video_path):
    """Rebuilds tracks (with team/position/position_transformed/speed) and
    space_control_per_frame from the cached stub, exactly the same
    enrichment chain backfill_tactical_event_frames_to_stats.py uses (team
    reattachment from stats.json, position, camera-movement adjustment,
    view transform, speed/distance) — confirmed there that the cached
    tracks.pkl is genuinely raw (bbox + is_goalkeeper only), so every one of
    these steps is required before the data is usable for anything below."""
    tracks_path = STUBS_DIR / f"cv_analysis_{cv_name}_tracks.pkl"
    camera_path = STUBS_DIR / f"cv_analysis_{cv_name}_camera_movement.pkl"
    homography_path = STUBS_DIR / f"cv_analysis_{cv_name}_homography.pkl"
    stats_path = CV_OUTPUT_DIR / cv_name / "stats.json"
    for p in (tracks_path, camera_path, homography_path, stats_path):
        if not p.exists():
            raise FileNotFoundError(f"required file missing: {p}")

    tracks = _load_pickle(tracks_path)
    camera_movement_per_frame = _load_pickle(camera_path)
    homography_per_frame = _load_pickle(homography_path)
    homography_per_frame = {f: h for f, h in homography_per_frame.items() if h is not None}

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    fps = stats["video"]["fps"]
    team_by_pid = {p["player_id"]: p["team"] for p in stats["players"]}

    for frame in tracks["players"]:
        for pid, info in frame.items():
            info["team"] = team_by_pid.get(pid, 0)

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

    return tracks, space_control_per_frame, fps, stats, team_by_pid


# ─────────────────────────── Coach Report audit ────────────────────────────

def verify_coach_report(match_name, tracks, space_control_per_frame, fps, stats):
    """Re-derives every tactical-event highlight's metric/intensity with
    event_ranking._intensity_and_metric — the SAME function that computed
    the original value, re-run on the reconstructed tracks/space_control —
    and flags any stored value it can't reproduce. Also independently
    re-runs full detection to sanity-check the stored event COUNTS, since a
    highlight's frame number is only trustworthy if the stub is still the
    one stats.json was actually built from."""
    findings = []

    detector = TacticalEventsDetector(fps=fps)
    events_by_frame = detector.detect(tracks, space_control_per_frame)
    ranked_windows, _ = rank_events_by_window(events_by_frame, tracks, space_control_per_frame, fps=fps)
    recomputed_counts = dict(detector.event_counts)
    stored_counts = stats.get("tactical_events", {}).get("counts", {})
    if recomputed_counts != stored_counts:
        findings.append({
            "match": match_name, "scope": "coach_report_integrity",
            "claim": f"stats.json tactical_events.counts = {stored_counts}",
            "actual": f"recomputed from cached tracks.pkl = {recomputed_counts}",
            "detail": "cached tracking stub no longer matches stats.json - every highlight's frame "
                      "number in this match is untrustworthy until backfill_tactical_event_frames_to_stats.py "
                      "is re-run.",
        })

    highlights = stats.get("tactical_events", {}).get("highlights", [])
    for h in highlights:
        fn = h.get("frame")
        pid = h.get("player_id")
        etype = h.get("type")
        if fn is None:
            findings.append({
                "match": match_name, "scope": "coach_report",
                "claim": f"{etype} / player {pid} — no frame recorded",
                "actual": "can't verify: this is exactly the 'frame not recorded' case "
                          "_format_event_evidence already reports plainly instead of fabricating one",
                "detail": "not a mismatch, informational only.",
                "informational": True,
            })
            continue
        if fn >= len(tracks["players"]):
            findings.append({
                "match": match_name, "scope": "coach_report",
                "claim": f"{etype} / player {pid} / frame {fn}",
                "actual": f"frame {fn} is out of range for this clip ({len(tracks['players'])} frames)",
                "detail": "stored frame number doesn't exist in the reconstructed tracks - can't verify.",
            })
            continue
        ev = {"frame": fn, "player_id": pid, "type": etype}
        recomputed_intensity, recomputed_metric = _intensity_and_metric(ev, tracks, space_control_per_frame, fps)
        recomputed_intensity = round(recomputed_intensity, 2)
        stored_metric = h.get("metric")
        stored_intensity = h.get("intensity")
        metric_ok = (recomputed_metric == stored_metric)
        try:
            intensity_ok = abs(float(stored_intensity) - recomputed_intensity) < 0.05
        except (TypeError, ValueError):
            intensity_ok = False
        if not (metric_ok and intensity_ok):
            findings.append({
                "match": match_name, "scope": "coach_report",
                "claim": f"{etype} / player {pid} / frame {fn} — metric='{stored_metric}', intensity={stored_intensity}",
                "actual": f"re-derived from tracks at that exact frame: metric='{recomputed_metric}', intensity={recomputed_intensity}",
                "detail": "stats.json's stored value doesn't match what event_ranking._intensity_and_metric "
                          "computes from the actual tracking data at this event's own frame.",
            })
    return findings


# ───────────────────────── Data Dashboard audit ────────────────────────────

def compute_attack_dir(tracks, team_by_pid):
    """Same heuristic as tactical_events_detector.py's own 'Attacking
    direction per team' block: per player, mean world-x across every frame
    they have a valid transformed position; per team, mean of those
    per-player means. The team sitting deeper (lower mean x) attacks toward
    +x. Mirrors cv_pipeline/tactical_events/tactical_events_detector.py
    lines ~143-162 exactly, just recomputed here independently from the
    freshly-reconstructed tracks rather than trusted from stats.json (which
    doesn't store it at all)."""
    pid_xs = defaultdict(list)
    for frame in tracks["players"]:
        for pid, info in frame.items():
            p = info.get("position_transformed")
            if p is not None:
                pid_xs[pid].append(float(p[0]))
    team_x = {1: [], 2: []}
    for pid, xs in pid_xs.items():
        t = team_by_pid.get(pid)
        if t not in (1, 2) or not xs:
            continue
        team_x[t].append(float(np.mean(xs)))
    mean_x = {t: (float(np.mean(team_x[t])) if team_x[t] else PITCH_X_MAX / 2) for t in (1, 2)}
    if mean_x[1] <= mean_x[2]:
        return {1: 1.0, 2: -1.0}
    return {1: -1.0, 2: 1.0}


def real_ball_zone_for_team(tracks, team_num, attack_dir):
    """Buckets every frame's real tracked ball position into a pitch third
    RELATIVE TO team_num's own attacking direction (exactly what the
    Gemini prompt defines 'ball_zone' to mean — 'relative to the team in
    possession'), and returns the majority third plus the full per-third
    frame-count breakdown and how many frames actually had a usable ball
    position (ball detection isn't 100%, especially after a miss with no
    interpolation re-applied here on purpose, to only ever use real
    detected positions, never an interpolated guess, as ground truth)."""
    d = attack_dir[team_num]
    counts = {"attacking_third": 0, "middle_third": 0, "defensive_third": 0}
    n = 0
    for frame in tracks["ball"]:
        info = frame.get(1)
        if not info:
            continue
        p = info.get("position_transformed")
        if p is None:
            continue
        x = float(p[0])
        n += 1
        if d == 1.0:
            if x >= 2 * THIRD:
                counts["attacking_third"] += 1
            elif x >= THIRD:
                counts["middle_third"] += 1
            else:
                counts["defensive_third"] += 1
        else:
            if x <= THIRD:
                counts["attacking_third"] += 1
            elif x <= 2 * THIRD:
                counts["middle_third"] += 1
            else:
                counts["defensive_third"] += 1
    if n == 0:
        return None, counts, 0
    majority = max(counts, key=counts.get)
    return majority, counts, n


def verify_zone_ground_truth(match_name, bundle, seg_ts, tracks, team_by_pid, window_frames, window_fps):
    """The direct check the user asked for: for a minute that has real CV
    tracking coverage (originally only the single 30s 'peak momentum' window
    per match; now also every extra window extend_cv_coverage.py has
    processed - see verify_zone_ground_truth_all below), recompute the REAL
    ball-zone (from actual tracked ball + player positions, see above) and
    compare it to the ball_zone Gemini assigned that same minute in
    raw_data. This is the check that caught the known PSG 'Attacking Third'
    bug.

    window_frames/window_fps: this window's own actual frame count/fps, used
    to report real coverage (e.g. '30s of a 60s minute' for the original
    peak-momentum windows vs. the full 60s for extend_cv_coverage.py's
    windows, which process the whole minute, not just the first half)."""
    findings = []
    raw_data = bundle.get("raw_data", [])
    row = next((r for r in raw_data if r.get("timestamp") == seg_ts), None)
    if row is None:
        findings.append({
            "match": match_name, "scope": "data_dashboard_zone_ground_truth",
            "claim": f"cv window timestamp={seg_ts}",
            "actual": "no raw_data row has this timestamp",
            "detail": "can't ground-truth-check this CV window's minute - no matching raw_data row.",
        })
        return findings

    color_a, color_b = bundle["color_a"].lower(), bundle["color_b"].lower()
    cv_team_mapping = bundle.get("cv_team_mapping", {})
    inv_map = {v: int(k) for k, v in cv_team_mapping.items()}  # {'team_a': 1, 'team_b': 2} or reversed

    poss = str(row.get("team_in_possession", "")).lower()
    if poss == color_a:
        poss_key, poss_name = "team_a", bundle["team_a"]
    elif poss == color_b:
        poss_key, poss_name = "team_b", bundle["team_b"]
    else:
        findings.append({
            "match": match_name, "scope": "data_dashboard_zone_ground_truth",
            "claim": f"minute {seg_ts}: team_in_possession='{row.get('team_in_possession')}'",
            "actual": "neither team - no possession to ground-truth the zone claim against",
            "detail": "informational only, not a mismatch.",
            "informational": True,
        })
        return findings

    cv_num = inv_map.get(poss_key)
    claimed_zone = row.get("ball_zone")
    attack_dir = compute_attack_dir(tracks, team_by_pid)
    real_zone, counts, n = real_ball_zone_for_team(tracks, cv_num, attack_dir)

    window_sec = window_frames / window_fps if window_fps else 0
    coverage_note = (f"covers the full 60s minute" if window_sec >= 59
                      else f"covers only the first ~{window_sec:.0f}s of this 60s minute")
    window_note = f"ground truth {coverage_note} - {n} frames had a detected ball position"

    if real_zone is None:
        findings.append({
            "match": match_name, "scope": "data_dashboard_zone_ground_truth",
            "claim": f"minute {seg_ts}: {poss_name} Primary Zone data point = '{claimed_zone}'",
            "actual": "no usable ball position in any frame of the CV window - can't ground-truth this claim",
            "detail": window_note,
        })
        return findings

    match_ok = (real_zone == claimed_zone)
    finding = {
        "match": match_name, "scope": "data_dashboard_zone_ground_truth",
        "claim": f"minute {seg_ts}: raw_data says {poss_name} (in possession) ball_zone = '{claimed_zone}'",
        "actual": f"real tracked ball position says '{real_zone}' "
                  f"(frame breakdown: {counts}, {window_note})",
        "detail": "CONFIRMED mismatch between the Gemini-labeled zone and where the ball actually, "
                  "verifiably was, relative to this team's own real tracked attacking direction."
                  if not match_ok else "real tracked ball position matches the claimed zone.",
    }
    if not match_ok:
        findings.append(finding)
    else:
        finding["informational"] = True
        findings.append(finding)
    return findings


def verify_zone_ground_truth_all(match_name, match_cfg, bundle, primary_tracks, primary_team_by_pid,
                                  primary_window_frames, primary_fps):
    """Runs verify_zone_ground_truth for the original peak-momentum window
    (tracks already loaded by run_match for the Coach Report check, reused
    here rather than reloaded) PLUS every extra window
    extend_cv_coverage.py has recorded in this match's
    _extra_cv_windows.json, if any. Each extra window gets its own
    load_enriched() call (its own clip, its own cached stub) - this is the
    only place coverage actually grows beyond the single original minute."""
    findings = []
    findings += verify_zone_ground_truth(
        match_name, bundle, bundle.get("cv_segment_timestamp"),
        primary_tracks, primary_team_by_pid, primary_window_frames, primary_fps)

    for w in load_extra_windows(match_name):
        cv_name, ts = w["cv_name"], w["timestamp"]
        clip_path = CLIPS_DIR / f"{cv_name}.mp4"
        if not clip_path.exists():
            findings.append({
                "match": match_name, "scope": "data_dashboard_zone_ground_truth",
                "claim": f"extra window {ts} ({cv_name})",
                "actual": f"clip missing at {clip_path} - can't re-enrich its tracks",
                "detail": "recorded in _extra_cv_windows.json but the clip isn't on disk - re-run "
                          "extend_cv_coverage.py for this target.",
            })
            continue
        tracks, _, fps, _, team_by_pid = load_enriched(cv_name, clip_path)
        findings += verify_zone_ground_truth(
            match_name, bundle, ts, tracks, team_by_pid, len(tracks["players"]), fps)
    return findings


def verify_attack_sec_consistency(match_name, bundle):
    """Every minute, both matches, no tracking data needed: flags a minute
    whose ball_zone claims a team was pinned in their attacking_third (with
    the ball) while that same team's own attack_sec says they barely
    attacked at all that minute, or the mirror case for defensive_third -
    the same directionality-bug class as the ground-truth check above, just
    checkable on every minute instead of only the one with CV coverage."""
    findings = []
    color_a, color_b = bundle["color_a"].lower(), bundle["color_b"].lower()
    team_a_name, team_b_name = bundle["team_a"], bundle["team_b"]
    for row in bundle.get("raw_data", []):
        ts = row.get("timestamp")
        poss = str(row.get("team_in_possession", "")).lower()
        zone = row.get("ball_zone")
        if poss == color_a:
            own_sec, team_label = row.get("team_a_attack_sec"), team_a_name
        elif poss == color_b:
            own_sec, team_label = row.get("team_b_attack_sec"), team_b_name
        else:
            continue
        try:
            own_sec = float(own_sec)
        except (TypeError, ValueError):
            continue
        if zone == "attacking_third" and own_sec < ATTACK_SEC_LOW_THRESHOLD:
            findings.append({
                "match": match_name, "scope": "data_dashboard_consistency",
                "claim": f"minute {ts}: {team_label} (in possession) ball_zone = 'attacking_third'",
                "actual": f"{team_label}'s own team_{'a' if poss==color_a else 'b'}_attack_sec this minute = {own_sec:.0f}s (out of 60s)",
                "detail": f"PLAUSIBLE mismatch: a team credited with only {own_sec:.0f}s of attacking play "
                          f"this minute is an unlikely candidate for 'pinned the opponent in their third with "
                          f"the ball' at the same time (threshold: <{ATTACK_SEC_LOW_THRESHOLD:.0f}s flagged).",
            })
        if zone == "defensive_third" and own_sec > ATTACK_SEC_HIGH_THRESHOLD:
            findings.append({
                "match": match_name, "scope": "data_dashboard_consistency",
                "claim": f"minute {ts}: {team_label} (in possession) ball_zone = 'defensive_third'",
                "actual": f"{team_label}'s own team_{'a' if poss==color_a else 'b'}_attack_sec this minute = {own_sec:.0f}s (out of 60s)",
                "detail": f"PLAUSIBLE mismatch: a team credited with {own_sec:.0f}s of attacking play this "
                          f"minute is an unlikely candidate for 'had the ball pinned in their own defensive "
                          f"third' at the same time (threshold: >{ATTACK_SEC_HIGH_THRESHOLD:.0f}s flagged).",
            })
    return findings


# ───────────────────────────── Global Control mirror ───────────────────────

def compute_dashboard_df(raw_df, color_a, color_b):
    """Verbatim mirror of dashboard/app.py:488-521's compute_dashboard_df.
    Kept here only so this script can recompute the exact same Primary-Zone
    mode-selection app.py's Global Control cards use, as the 'claim' text
    the checks above are run against. MUST be kept in sync with app.py if
    that function's formulas ever change."""
    df = raw_df.copy()
    df.replace(["", " ", "Unknown", "N/A", None], np.nan, inplace=True)
    df = df.ffill().bfill()
    df["team_a_has_ball"] = (df["team_in_possession"].str.lower() == color_a.lower()).astype(int)
    df["team_b_has_ball"] = (df["team_in_possession"].str.lower() == color_b.lower()).astype(int)
    return df


def primary_zone(df, has_ball_col):
    """Mirrors app.py:5145-5152 / 5157-5164 exactly (the Global Control
    'Primary Zone' card's own mode-selection logic, incl. its >=2 floor)."""
    poss = df[df[has_ball_col] == 1]["ball_zone"]
    if poss.empty or poss.value_counts().iloc[0] < 2:
        return None, []
    zone_raw = poss.value_counts().index[0]
    minutes = df.loc[(df[has_ball_col] == 1) & (df["ball_zone"] == zone_raw), "timestamp"].tolist()
    return zone_raw, minutes


def report_primary_zone_claims(match_name, bundle):
    """Not a check by itself (no independent oracle for a code-aggregation
    bug without a second implementation) - prints exactly what the
    'Primary Zone' cards claim for this match so it's visible alongside the
    ground-truth and consistency findings above, and confirms the >=2-row
    floor and mode logic actually ran on real data rather than asserting it
    from reading the code alone."""
    df = compute_dashboard_df(pd.DataFrame(bundle["raw_data"]), bundle["color_a"], bundle["color_b"])
    ta_zone, ta_minutes = primary_zone(df, "team_a_has_ball")
    tb_zone, tb_minutes = primary_zone(df, "team_b_has_ball")
    print(f"    Primary Zone claims as the UI would compute them:")
    print(f"      {bundle['team_a']}: {ta_zone or 'None'}  (minutes: {', '.join(ta_minutes) if ta_minutes else '-'})")
    print(f"      {bundle['team_b']}: {tb_zone or 'None'}  (minutes: {', '.join(tb_minutes) if tb_minutes else '-'})")


# ──────────────────────────────── driver ───────────────────────────────────

def run_match(match_cfg):
    name, cv_name = match_cfg["name"], match_cfg["cv_name"]
    bundle_path = CURATED_DIR / name / "bundle.json"
    video_path = CURATED_DIR / name / "peak_momentum_segment.mp4"
    with open(bundle_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    print(f"\n{'=' * 70}\n{name}  ({bundle['team_a']} vs {bundle['team_b']})\n{'=' * 70}")

    print(f"\n[Coach Report] re-deriving every tactical-event highlight from the cached tracking stub...")
    tracks, space_control_per_frame, fps, stats, team_by_pid = load_enriched(cv_name, video_path)
    coach_findings = verify_coach_report(name, tracks, space_control_per_frame, fps, stats)
    n_coach_checked = len(stats.get("tactical_events", {}).get("highlights", []))
    n_coach_real = sum(1 for f in coach_findings if not f.get("informational"))
    print(f"  {n_coach_checked} highlight(s) checked, {n_coach_real} real mismatch(es)")

    print(f"\n[Data Dashboard] ground-truth zone check against real tracked ball position...")
    zone_findings = verify_zone_ground_truth_all(name, match_cfg, bundle, tracks, team_by_pid,
                                                   len(tracks["players"]), fps)
    n_zone_checked = 1 + len(load_extra_windows(name))
    n_zone_real = sum(1 for f in zone_findings if not f.get("informational"))
    print(f"  {n_zone_checked} minute(s) with real CV coverage checked, {n_zone_real} real mismatch(es)")

    print(f"\n[Data Dashboard] internal consistency check across all {len(bundle.get('raw_data', []))} minutes...")
    consistency_findings = verify_attack_sec_consistency(name, bundle)
    print(f"  {len(consistency_findings)} plausible mismatch(es)")

    report_primary_zone_claims(name, bundle)

    return coach_findings + zone_findings + consistency_findings


def print_findings(all_findings):
    real = [f for f in all_findings if not f.get("informational")]
    info = [f for f in all_findings if f.get("informational")]

    print(f"\n{'=' * 70}\nRESULTS: {len(real)} mismatch(es) found across both matches\n{'=' * 70}")
    if not real:
        print("No mismatches found.")
    for i, f in enumerate(real, 1):
        print(f"\n[{i}] {f['match']} — {f['scope']}")
        print(f"    CLAIM : {f['claim']}")
        print(f"    ACTUAL: {f['actual']}")
        print(f"    {f['detail']}")

    if info:
        print(f"\n--- {len(info)} informational note(s) (not mismatches) ---")
        for f in info:
            print(f"  {f['match']} / {f['scope']}: {f['claim']} -> {f['actual']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", choices=[m["name"] for m in MATCHES], default=None,
                     help="only check this match (default: both curated matches)")
    args = ap.parse_args()

    matches = [m for m in MATCHES if m["name"] == args.match] if args.match else MATCHES
    all_findings = []
    for m in matches:
        all_findings.extend(run_match(m))
    print_findings(all_findings)
