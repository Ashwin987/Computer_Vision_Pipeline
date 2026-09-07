"""
analyze_danger_score.py — per-frame "danger score" (0-100): how
threatening the current attacking situation is, from spatial data only.

Standalone, additive analysis script. Does NOT import, modify, or touch
main.py, fast_setup.py, fast_render_*.py, or any render_output*.py file.
Reuses (imports, doesn't modify):
  - analyze_formation_window.load_window_tracks / assign_teams_for_window
    / SOURCES / PITCH_LENGTH / PITCH_WIDTH / PITCH_MARGIN — the same
    cached-stub loading, position_transformed computation, outlier
    filtering, and per-frame jersey-colour team assignment already
    proven there.
  - tactical_events.space_control.nearest_player_per_cell — the exact
    nearest-player "lightweight Voronoi" algorithm render_output4.py and
    main.py's SPACE tactical event use, applied here to a real-world
    (metres) grid over the attacking third instead of their pixel-space/
    camera-view grid (this script needs no camera frames at all, so the
    world-space position_transformed data is the natural coordinate
    space — no pitch mask or perspective handling needed for a plain
    axis-aligned rectangle).

Deliberately skips render_output3._link_identities(): unlike formation
detection, none of the five danger signals need to know a player is
"the same identity" across frames — each frame's spatial arrangement
(who is where, which team, ball position) is scored independently. This
sidesteps the identity-fragmentation problems formation analysis hit.

Five per-frame signals (each normalized to 0-100, see NORMALIZATION
constants below for the caps used and why):
  a. ball_proximity      - closer to the goal being attacked = higher
  b. space_control       - attacking team's share of nearest-player
                            control over the attacking-third grid
  c. compactness         - defending team's convex-hull (or bounding-box,
                            if <3 defenders) area; LARGER area (stretched,
                            disorganized defence) = MORE danger, so this
                            is used directly (not inverted) as a danger
                            driver — "inverting" compactness-as-goodness
                            into danger-as-badness is exactly a direct
                            area->score mapping, not a 1/area one
  d. numerical_advantage  - (attackers - defenders) inside the attacking
                            third
  e. ball_velocity        - signed closing speed of the ball toward the
                            goal being attacked (km/h; interpolated ball
                            trajectory, mirroring Tracker.interpolate_
                            ball_positions' pandas interpolate+bfill, but
                            applied to world-metres positions rather than
                            pixel bboxes so a gap doesn't compound
                            per-frame homography differences)

"Attacking team" is decided fresh every frame (item 3): whichever team
has more aggregate territorial presence past halfway into the opponent's
half; ties (commonly 0-0, e.g. a settled midfield exchange) fall back to
which side of the pitch the ball is currently on.

Usage:
    python analyze_danger_score.py [--source {fastpipe,main}] [start_frame] [end_frame]
    (defaults to --source main, the whole 750-frame 121364_0.mp4 clip)
"""

import math
import os
import time
from collections import Counter

import cv2
import numpy as np
import pandas as pd

from analyze_formation_window import (
    load_window_tracks, assign_teams_for_window,
    PITCH_LENGTH, PITCH_WIDTH, PITCH_MARGIN, SOURCES,
)
from tactical_events.space_control import nearest_player_per_cell

GOAL_Y = PITCH_WIDTH / 2.0        # 34.0m — goal-mouth y-coordinate at both ends
FINAL_THIRD_M = PITCH_LENGTH / 3.0  # 35.0m — "final third" / attacking-third depth
ZONE_GRID_STEP_M = 2.0            # world-space grid resolution for space_control

# Hand-tuned starting weights (sum to 1.0) — ball proximity and space
# control near goal weighted highest per spec, as the two most directly
# meaningful signals; the other three are secondary/contextual.
WEIGHTS = {
    'ball_proximity':      0.30,
    'space_control':       0.25,
    'compactness':         0.15,
    'numerical_advantage': 0.15,
    'ball_velocity':       0.15,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "WEIGHTS must sum to 1.0"

# ── Normalization caps: raw signal value that maps to a 0-100 score ────────
# ball_proximity: farthest a ball can realistically be from a goal mouth —
# the opposite corner of the pitch to the far goal post's y — normalizes
# distance-to-goal down to [0,1] before scaling to 100.
MAX_BALL_DIST_M = math.hypot(PITCH_LENGTH, PITCH_WIDTH / 2.0)   # ~110.4m
# compactness: heuristic ceiling for a very stretched/disorganized back
# line's convex-hull area; a compact back four/five is typically 200-800m^2.
MAX_HULL_AREA_M2 = 3000.0
# numerical_advantage: a +/-5 attacker-vs-defender overload in the final
# third is already an extreme, rare situation.
MAX_NUMERICAL_ADV = 5.0
# ball_velocity: closing speed ceiling in km/h — a hard driven pass/shot
# rarely closes distance-to-goal faster than this.
MAX_CLOSING_SPEED_KMH = 80.0

SMOOTH_WINDOW_FRAMES = 12   # ~0.5s @ 24fps, within the requested 10-15 range
PEAK_MIN_SEPARATION_SECONDS = 2.0   # distinct "moments", not the same peak 5x
# A peak must clear this smoothed score to count as "notable" at all — set
# from the short-clip runs' observed top-5 range (56.5-62.8), so it's a
# defensible "this is a real peak" bar rather than an arbitrary number.
DANGER_PEAK_THRESHOLD = 55.0

# Frames within this many seconds of the loaded window's start or end are
# excluded from PEAK REPORTING ONLY (not from the score itself, not from
# the CSV log, not from the sparkline). Diagnosed on frame 42148 (13.1s
# before the full match's final frame): a "9v2 overload" that checked out
# as spatially real (no duplicates, no misfired zone math, legitimate team
# locks) but was actually stoppage-time/end-of-clip congregation near a
# goal — a corner scramble, celebration, or players leaving the pitch —
# not organized attacking play. The model has no notion of match phase,
# so spatial clustering right at a clip boundary reliably saturates every
# signal without being a real tactical chance. This is a blunt, honest
# boundary exclusion (we don't know WHY the boundary is chaotic, just that
# clip start/end footage is systematically unlike in-play football), not
# an attempt to model stoppage time specifically — a genuine chance that
# happens to occur in the last 20s of actual match time would also be
# excluded here, which is an accepted, explicit trade-off, not a claim
# that no real danger ever happens near a clip boundary.
BOUNDARY_EXCLUSION_SECONDS = 20.0

# Console/sparkline readability: a 750-frame clip and a 42,475-frame full
# match need very different bin sizes — these targets keep either one to a
# reasonably sized terminal print instead of hardcoding a fixed-size bin
# that's fine for one and unusable for the other.
TARGET_SPARKLINE_BARS = 150

# ══════════════════════════════════════════════════════════════════════════
# THREE-CATEGORY REDESIGN
#
# Every frame now gets exactly one category — SHOT, SET_PIECE, OPEN_PLAY, or
# NONE — decided in that precedence order (a shot at the end of a corner is
# tagged SHOT, not SET_PIECE; the dead-ball buildup frames before it are
# still SET_PIECE). This replaces the previous design, which only tagged
# SET_PIECE as an annotation on top of frames that had ALREADY qualified as
# an open-play peak via the composite formula — meaning a real dead-ball
# congregation could still get counted (and top-ranked!) as "organized
# attacking danger" as long as its composite score cleared the bar. Now a
# SET_PIECE frame is never scored by the open-play formula's output at all;
# it gets its own confidence number instead. The composite formula, its
# weights, and SMOOTH_WINDOW_FRAMES/DANGER_PEAK_THRESHOLD are UNCHANGED —
# they still define OPEN_PLAY exactly as before, just no longer catch
# everything indiscriminately.
# ══════════════════════════════════════════════════════════════════════════

# ── Set-piece detection ─────────────────────────────────────────────────────
# Signature: a dead ball. Corners/free kicks pause play — the ball sits
# nearly still while players walk into a static shape near goal — right
# before the delivery spikes every open-play signal at once (exactly what
# happened at the render-window peaks investigated previously). Two
# identity-free signals (no per-player tracking needed, consistent with
# this script's original design): ball speed over a lookback window, and
# how stable the defending team's hull area is over that same window (a
# settled defensive shape barely changes; dynamic attacking/defensive runs
# do). Lookback shortened from an earlier 3.0s to match the spec's "1-2
# seconds" for how long a restart actually sits dead before delivery.
SET_PIECE_LOOKBACK_SECONDS = 1.5
SET_PIECE_BALL_STATIC_KMH = 8.0        # a frame counts as "ball static" below this speed
SET_PIECE_BALL_STATIC_FRACTION = 0.5   # >= this fraction of the lookback must be static
# Required (not just corroborating, unlike the earlier design): a hull area
# that's still visibly moving isn't a settled dead-ball shape yet.
SET_PIECE_HULL_STABLE_COV = 0.20
# Corners/free-kick deliveries into the box originate close to goal; a dead
# ball at the halfway line (injury stoppage, offside pause) isn't a "corner/
# free kick positioning" situation even if the ball is briefly still.
SET_PIECE_MAX_DISTANCE_M = 45.0

# ── Shot detection: independent of the smoothed composite score ────────────
# A real shot is a short, sharp burst. Two distinct smoothing regimes now
# exist side by side: SMOOTH_WINDOW_FRAMES (12, ~0.5s) for the open-play
# composite, and this much shorter one for shots — short enough that a
# shot's spike isn't diluted into the surrounding build-up, but long enough
# (3-5 frames, ~0.12-0.2s) to reject the single-frame ball-tracking
# glitches discovered while building this (see BALL_MAX_PLAUSIBLE_SPEED_KMH
# below): a genuine shot's ball keeps moving fast for several consecutive
# frames as it travels; an isolated bad detection does not.
SHOT_SMOOTH_WINDOW_FRAMES = 4
SHOT_MIN_SPEED_KMH = 60.0        # below this, treat as a fast pass/cross, not a strike
SHOT_MAX_DISTANCE_M = 30.0       # must originate from realistic shooting range
SHOT_SPEED_CEILING_KMH = 120.0   # normalizes the velocity component; shots can exceed the
                                  # danger formula's own MAX_CLOSING_SPEED_KMH=80 cap
# shot_score blends velocity and proximity (task spec: "based on velocity
# magnitude AND proximity to goal"), not velocity alone — a 90km/h strike
# from 8m and a 90km/h strike from 29m aren't equally dangerous.
SHOT_VELOCITY_WEIGHT = 0.7
SHOT_PROXIMITY_WEIGHT = 0.3
SHOT_MIN_SEPARATION_SECONDS = 1.5   # distinct strikes, not the same spike counted twice

# closing_speed_kmh is a frame-to-frame finite difference with no outlier
# filtering — a single bad ball detection (misdetected object, brief jump)
# produces a spurious multi-hundred-km/h reading. The composite formula
# never noticed because velocity_raw is _clip01'd, so a 2000km/h glitch and
# a genuine 85km/h strike both just saturate to the same 100/100 — but a
# feature reading the raw number directly surfaces it immediately (observed:
# up to 2279km/h on the short clip, pre-fix). The fastest officially
# recorded football strikes are ~130km/h; gate shot/set-piece candidates on
# ball_speed_kmh (unsigned, same finite-difference math) so a tracking
# glitch can't masquerade as either. Does NOT touch compute_frame_signals
# or the composite formula.
BALL_MAX_PLAUSIBLE_SPEED_KMH = 150.0

# ── UNCERTAIN: long ball-detection gaps ─────────────────────────────────────
# Diagnosed directly on the 26:12 shot (frame 39300, LiverpoolPSG_short.mp4):
# the ball had zero RAW (non-interpolated) detections for 47 consecutive
# frames (1.88s) immediately after the strike. Every category detector
# still ran on that stretch anyway, scoring a fabricated straight-line
# interpolated path as if it were real ball motion — the shot's actual
# high-speed phase was structurally invisible, and worse, nothing flagged
# that the numbers being shown were made up. UNCERTAIN pre-empts every
# other category (it's checked FIRST in classify_frames) so a long gap can
# never masquerade as SHOT/SET_PIECE/OPEN_PLAY/NONE. It's an internal/notes
# category only — see run()'s docstring note on why it must never appear
# as an on-screen label in the eventual video overlay.
UNCERTAIN_GAP_MAX_FRAMES = 15   # > this many consecutive non-real frames = uncertain (0.6s @ 25fps)

TARGET_CONSOLE_PREVIEW_LINES = 80

DEFAULT_SOURCE = 'main'


# ── Data extraction ─────────────────────────────────────────────────────────

def _valid_position(pos):
    """Apply the same degenerate-homography outlier filter + pitch-rectangle
    clip used throughout analyze_formation_window.py. Returns (x, y) or None."""
    if pos is None:
        return None
    x, y = float(pos[0]), float(pos[1])
    if not (-PITCH_MARGIN <= x <= PITCH_LENGTH + PITCH_MARGIN
            and -PITCH_MARGIN <= y <= PITCH_WIDTH + PITCH_MARGIN):
        return None
    return min(max(x, 0.0), PITCH_LENGTH), min(max(y, 0.0), PITCH_WIDTH)


# Diagnosis (see analyze_danger_score's investigation on 121364_0.mp4):
# 31.7% of frames had a physically impossible >11-per-team raw pid count.
# Checking every anomalous frame's pairwise same-team distances found only
# 96/238 (40.3%) resolve if two IDs within this radius are treated as one
# real player (a fragmented/duplicate detection — e.g. a ByteTrack ID
# handoff where a fading old id and its replacement briefly co-exist).
# The remaining 142/238 (59.7%) are NOT proximity duplicates — genuinely
# separate, spread-out extra detections (most likely a referee or other
# non-outfield-player detection slipping into a team's jersey-colour
# cluster) that this fix does NOT and cannot resolve; those frames are
# still flagged as anomalous after dedup, honestly, rather than force-
# capped to look clean.
WITHIN_FRAME_DEDUP_RADIUS_M = 2.0

# A "player" detection within this distance of a tracked referee in the
# SAME frame is very likely the referee itself having slipped into a
# team's jersey-colour cluster (referees wear a third, distinct kit, but
# get compared only against the two team centroids), rather than a real
# outfield player. Diagnosed as a candidate cause of the remaining
# post-dedup anomalies; empirically only resolves 3/142 of them (see
# extract_frame_players docstring) — most of what's left is something
# else entirely (a persistently-tracked extra identity with a normal,
# continuous on-pitch trajectory — not near any referee, not near any
# teammate — that neither this nor the within-frame dedup can catch).
REFEREE_EXCLUSION_RADIUS_M = 1.5

# Any "player" detection within this distance of the pitch boundary is
# excluded — but ONLY the two long SIDELINES (y=0 / y=PITCH_WIDTH), not
# the goal lines (x=0 / x=PITCH_LENGTH). "Touchline" in football
# specifically means the sideline, where technical areas/coaching staff
# actually stand; the goal lines have no such exclusion — real defenders
# and goalkeepers stand right up against their OWN goal line constantly
# during normal play (corners, deep blocks, keeper distribution), so
# excluding near x=0/105 too was tried first and caused a real
# regression: mean defenders-per-frame dropped from 7.69 to 6.98, with
# some frames down to just 2 detected defenders and one peak showing
# "0v0 in the final third" — legitimate deep defenders were being
# stripped, not the misclassified extra identity. Restricting to the
# sidelines only fixes that collateral damage.
#
# CAVEAT, checked before trusting this fix: the specific case this was
# proposed to explain (pid 85, present frames 221-558 on team 2) does
# NOT actually match a "stands on the touchline, barely moves" profile.
# Its measured trajectory spans dx=49.7m, dy=68.0m (touchline to
# touchline) with a MEAN distance of 20.79m from the nearest boundary —
# only 11 of 228 frames (4.8%) are even within 2m of an edge, and most of
# those are exactly y=0.0 or y=68.0 (i.e. the PITCH_MARGIN clip boundary
# itself — a sign of an underlying degenerate-homography outlier for
# that one frame, not a genuine touchline detection). This filter is
# still implemented as specified below (it's a reasonable general-purpose
# guard and does catch real touchline-adjacent detections), but it should
# NOT be expected to resolve pid 85 or the 257-341 frame cluster — see
# the re-run results in this script's calling report for confirmation.
TOUCHLINE_EXCLUSION_M = 1.5


def _dedup_same_team(team_points):
    """team_points: [(pid, x, y), ...] for ONE team in ONE frame. Any two
    points within WITHIN_FRAME_DEDUP_RADIUS_M of each other are treated as
    the same real player (not merged across teams — a nearby attacker and
    defender are real contested play, not a duplicate). Keeps the
    lowest-pid representative per merged group; does not average, per
    spec ("keeping only one of them")."""
    n = len(team_points)
    if n <= 1:
        return list(team_points)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = team_points[i][1], team_points[i][2]
            xj, yj = team_points[j][1], team_points[j][2]
            if math.hypot(xi - xj, yi - yj) < WITHIN_FRAME_DEDUP_RADIUS_M:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    return [team_points[min(idxs, key=lambda i: team_points[i][0])]
            for idxs in groups.values()]


def _referee_positions(tracks, frame_num):
    """[(x, y), ...] for every tracked referee in this frame with a valid
    position_transformed."""
    frame_data = tracks['referees'][frame_num] if frame_num < len(tracks['referees']) else {}
    out = []
    for _rid, info in frame_data.items():
        xy = _valid_position(info.get('position_transformed'))
        if xy is not None:
            out.append(xy)
    return out


def _near_touchline(x, y):
    """True if (x, y) is within TOUCHLINE_EXCLUSION_M of a SIDELINE
    (y=0 or y=PITCH_WIDTH) — deliberately NOT the goal lines (x=0/
    PITCH_LENGTH), where real defenders/goalkeepers legitimately stand
    during normal play. See TOUCHLINE_EXCLUSION_M's comment."""
    return y <= TOUCHLINE_EXCLUSION_M or y >= PITCH_WIDTH - TOUCHLINE_EXCLUSION_M


def extract_frame_players(tracks, frame_num):
    """[(pid, team, x, y), ...] for this frame's players with a resolved
    team (1/2) and a valid position_transformed, after (1) excluding
    detections within TOUCHLINE_EXCLUSION_M of the pitch boundary, (2)
    excluding detections within REFEREE_EXCLUSION_RADIUS_M of a tracked
    referee that frame, then (3) within-frame same-team proximity
    deduplication (see _dedup_same_team). No identity linking ACROSS
    frames — every frame is still independent."""
    frame_data = tracks['players'][frame_num] if frame_num < len(tracks['players']) else {}
    referees = _referee_positions(tracks, frame_num)
    by_team = {1: [], 2: []}
    for pid, info in frame_data.items():
        team = info.get('team', 0)
        if team not in (1, 2):
            continue
        xy = _valid_position(info.get('position_transformed'))
        if xy is None:
            continue
        x, y = xy
        if _near_touchline(x, y):
            continue   # likely a technical-area person, not a real player
        if any(math.hypot(x - rx, y - ry) < REFEREE_EXCLUSION_RADIUS_M for rx, ry in referees):
            continue   # very likely the referee, not a real player
        by_team[team].append((pid, x, y))

    out = []
    for team in (1, 2):
        for pid, x, y in _dedup_same_team(by_team[team]):
            out.append((pid, team, x, y))
    return out


def diagnose_frame(tracks, frame_num):
    """Print every player's raw pid/team/position for `frame_num`, the
    pairwise same-team distances under WITHIN_FRAME_DEDUP_RADIUS_M, and
    the before/after dedup counts — the exact diagnostic used to
    establish WITHIN_FRAME_DEDUP_RADIUS_M's coverage (40.3% of anomalies)."""
    frame_data = tracks['players'][frame_num] if frame_num < len(tracks['players']) else {}
    raw_by_team = {1: [], 2: []}
    for pid, info in frame_data.items():
        team = info.get('team', 0)
        if team not in (1, 2):
            continue
        xy = _valid_position(info.get('position_transformed'))
        if xy is None:
            continue
        raw_by_team[team].append((pid, xy[0], xy[1]))

    referees = _referee_positions(tracks, frame_num)
    print(f"=== frame {frame_num} diagnostic ===")
    print(f"  Referees this frame: {referees}")
    for team in (1, 2):
        pts = sorted(raw_by_team[team], key=lambda p: p[0])
        print(f"  Team {team}: {len(pts)} raw detections")
        for pid, x, y in pts:
            flags = []
            if _near_touchline(x, y):
                flags.append("within touchline-exclusion margin")
            if any(math.hypot(x - rx, y - ry) < REFEREE_EXCLUSION_RADIUS_M for rx, ry in referees):
                flags.append("within referee-exclusion radius")
            flag = f"  <- {', '.join(flags)}" if flags else ""
            print(f"    pid={pid:5d}  x={x:6.2f}  y={y:6.2f}{flag}")
        print(f"  Team {team} same-team pairs within "
              f"{WITHIN_FRAME_DEDUP_RADIUS_M}m:")
        found = False
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = math.hypot(pts[i][1] - pts[j][1], pts[i][2] - pts[j][2])
                if d < WITHIN_FRAME_DEDUP_RADIUS_M:
                    print(f"    pid {pts[i][0]} <-> pid {pts[j][0]}: {d:.2f}m")
                    found = True
        if not found:
            print("    (none)")
        after_touchline = [(pid, x, y) for pid, x, y in pts if not _near_touchline(x, y)]
        after_ref = [(pid, x, y) for pid, x, y in after_touchline
                     if not any(math.hypot(x - rx, y - ry) < REFEREE_EXCLUSION_RADIUS_M
                                for rx, ry in referees)]
        deduped = _dedup_same_team(after_ref)
        print(f"  Team {team}: {len(pts)} raw -> {len(after_touchline)} after touchline-exclusion "
              f"-> {len(after_ref)} after referee-exclusion -> {len(deduped)} after dedup")


def build_ball_trajectory(tracks, n):
    """World-space (x, y) for every frame, gaps filled by linear
    interpolation + edge back/forward-fill — the same interpolate+bfill
    approach as Tracker.interpolate_ball_positions, applied to the
    already-transformed world-metres position instead of the raw pixel
    bbox (avoids compounding per-frame homography differences across a
    detection gap)."""
    xs = np.full(n, np.nan)
    ys = np.full(n, np.nan)
    for fn in range(n):
        ball_frame = tracks['ball'][fn] if fn < len(tracks['ball']) else {}
        info = ball_frame.get(1)
        if not info:
            continue
        xy = _valid_position(info.get('position_transformed'))
        if xy is None:
            continue
        xs[fn], ys[fn] = xy
    xs = pd.Series(xs).interpolate(limit_direction='both').to_numpy()
    ys = pd.Series(ys).interpolate(limit_direction='both').to_numpy()
    return xs, ys


def compute_ball_speeds(ball_xs, ball_ys, fps):
    """Total (unsigned) ball speed magnitude per frame, km/h — distinct
    from compute_frame_signals' closing_speed_kmh, which is the SIGNED
    component toward whatever goal is being attacked that frame. This is
    plain speed regardless of direction, used for set-piece "is the ball
    dead" detection."""
    n = len(ball_xs)
    speeds = np.zeros(n)
    for i in range(1, n):
        d = math.hypot(ball_xs[i] - ball_xs[i - 1], ball_ys[i] - ball_ys[i - 1])
        speeds[i] = d * fps * 3.6
    if n > 1:
        speeds[0] = speeds[1]
    return speeds


def compute_shot_signal(rows, ball_speeds, fps):
    """Short-window-smoothed SIGNED closing speed (km/h, toward the goal
    being attacked that frame) — see SHOT_SMOOTH_WINDOW_FRAMES for why this
    is a much shorter window than the open-play composite's smoothing.

    Sanitizes closing_speed_kmh BEFORE smoothing: gating only the frame
    being scored isn't enough — a single implausible-speed frame (see
    BALL_MAX_PLAUSIBLE_SPEED_KMH) ANYWHERE inside a 4-frame rolling window
    still drags the mean itself into the hundreds-of-km/h range even when
    the center frame's own reading is perfectly sane (observed: a 516km/h
    "smoothed" signal on a frame whose own raw value was -4.7km/h). Bad
    frames are replaced via interpolation from their plausible neighbours
    before the rolling mean runs, the same interpolate-through-outliers
    pattern build_ball_trajectory already uses for position data."""
    raw = np.array([r['closing_speed_kmh'] for r in rows])
    implausible = ball_speeds > BALL_MAX_PLAUSIBLE_SPEED_KMH
    sanitized = raw.copy()
    sanitized[implausible] = np.nan
    sanitized = pd.Series(sanitized).interpolate(limit_direction='both').to_numpy()
    return pd.Series(sanitized).rolling(
        SHOT_SMOOTH_WINDOW_FRAMES, center=True, min_periods=1).mean().to_numpy()


def is_shot_frame(rows, shot_signal, ball_speeds, index):
    """(is_shot, shot_score_or_None). shot_score blends velocity and
    proximity per SHOT_VELOCITY_WEIGHT/SHOT_PROXIMITY_WEIGHT."""
    r = rows[index]
    speed = float(shot_signal[index])
    if speed < SHOT_MIN_SPEED_KMH:
        return False, None
    if ball_speeds[index] > BALL_MAX_PLAUSIBLE_SPEED_KMH:
        return False, None   # tracking glitch — see BALL_MAX_PLAUSIBLE_SPEED_KMH
    if r['ball_dist_to_goal_m'] > SHOT_MAX_DISTANCE_M:
        return False, None

    velocity_component = 100.0 * max(0.0, min(1.0, speed / SHOT_SPEED_CEILING_KMH))
    proximity_component = 100.0 * max(0.0, min(
        1.0, 1.0 - r['ball_dist_to_goal_m'] / SHOT_MAX_DISTANCE_M))
    shot_score = (SHOT_VELOCITY_WEIGHT * velocity_component
                  + SHOT_PROXIMITY_WEIGHT * proximity_component)
    return True, shot_score


def is_set_piece_frame(rows, ball_speeds, index, fps):
    """(is_set_piece, (confidence, evidence)_or_None). Identity-free — only
    ball speed and the already-computed defending-team hull area, no
    per-player tracking. `confidence` reflects how clearly this matches a
    dead-ball signature, deliberately NOT the open-play danger score (a
    static congregation isn't "how dangerous", it's "how confidently is
    this a restart")."""
    lookback_frames = round(SET_PIECE_LOOKBACK_SECONDS * fps)
    lo = max(0, index - lookback_frames)
    window_speeds = ball_speeds[lo:index]
    if len(window_speeds) < 3:
        return False, None

    # static_fraction is a threshold count, robust to the occasional
    # tracking-glitch speed spike (see BALL_MAX_PLAUSIBLE_SPEED_KMH) on its
    # own — a glitch frame is just correctly counted as "not static". The
    # reported median (not mean) below is for the same reason: a couple of
    # glitch frames in the lookback window would otherwise drag a plain
    # mean up to a physically nonsensical number while barely moving the
    # median, which is what's actually representative of a mostly-still ball.
    static_fraction = float(np.mean(window_speeds < SET_PIECE_BALL_STATIC_KMH))
    if static_fraction < SET_PIECE_BALL_STATIC_FRACTION:
        return False, None

    hull_areas = np.array([rows[j]['hull_area_m2'] for j in range(lo, index)])
    mean_hull = float(hull_areas.mean()) if len(hull_areas) else 0.0
    hull_cov = float(hull_areas.std() / mean_hull) if mean_hull > 1e-6 else 0.0
    if hull_cov >= SET_PIECE_HULL_STABLE_COV:
        return False, None   # players still moving — not a settled dead-ball shape yet

    if rows[index]['ball_dist_to_goal_m'] > SET_PIECE_MAX_DISTANCE_M:
        return False, None   # dead ball, but not near a goal — not corner/free-kick positioning

    confidence = 100.0 * static_fraction
    evidence = {
        'ball_static_fraction': static_fraction,
        'hull_area_cov': hull_cov,
        'median_ball_speed_kmh': float(np.median(window_speeds)),
    }
    return True, (confidence, evidence)


def find_uncertain_gaps(tracks, n, start_frame, detect_every):
    """Runs of > UNCERTAIN_GAP_MAX_FRAMES consecutive frames with no RAW
    (non-interpolated) ball detection. A frame only counts as a real
    detection if it's one of the source's actual detection-attempt frames
    (absolute frame_num % detect_every == 0 — always true for detect_every=1,
    i.e. main.py's every-frame source) AND the ball was actually found that
    frame; every other frame's ball bbox — frame-skip gap-fill OR a
    genuinely missed detection — is interpolated/fabricated position data,
    not observed.

    Returns [(gap_start_local, gap_end_local_inclusive), ...].
    """
    is_real = np.zeros(n, dtype=bool)
    for fn in range(n):
        if (start_frame + fn) % detect_every != 0:
            continue
        ball_frame = tracks['ball'][fn] if fn < len(tracks['ball']) else {}
        if ball_frame.get(1) is not None:
            is_real[fn] = True

    gaps = []
    gap_start = None
    for fn in range(n):
        if not is_real[fn]:
            if gap_start is None:
                gap_start = fn
        elif gap_start is not None:
            gap_len = fn - gap_start
            if gap_len > UNCERTAIN_GAP_MAX_FRAMES:
                gaps.append((gap_start, fn - 1))
            gap_start = None
    if gap_start is not None and (n - gap_start) > UNCERTAIN_GAP_MAX_FRAMES:
        gaps.append((gap_start, n - 1))   # gap runs to the end of the loaded window

    return gaps


def classify_frames(rows, ball_speeds, fps, uncertain_mask=None):
    """Every frame gets exactly one category, in precedence order
    UNCERTAIN > SHOT > SET_PIECE > OPEN_PLAY > NONE. UNCERTAIN pre-empts
    everything else — see UNCERTAIN_GAP_MAX_FRAMES's comment for why a
    frame with no real underlying ball position must never be allowed to
    masquerade as any other category, however its fabricated interpolated
    data happens to score. SET_PIECE frames still never fall through to be
    counted as OPEN_PLAY (satisfying "do NOT score them using the open-play
    formula"). Mutates each row in place with 'category', 'shot_score',
    'set_piece_confidence', 'set_piece_evidence'.

    NOTE for the (not-yet-updated) video overlay: category == 'UNCERTAIN'
    must render as a blank/neutral state — no score, no on-screen label,
    per spec. This function only produces the classification; the notes
    file (written by run()) is where UNCERTAIN windows are meant to be
    reported, not the video."""
    n = len(rows)
    shot_signal = compute_shot_signal(rows, ball_speeds, fps)
    counts = {'UNCERTAIN': 0, 'SHOT': 0, 'SET_PIECE': 0, 'OPEN_PLAY': 0, 'NONE': 0}

    for i in range(n):
        r = rows[i]
        r['shot_score'] = None
        r['shot_signal_kmh'] = float(shot_signal[i])   # the SMOOTHED value the decision is based on —
        r['set_piece_confidence'] = None                # report this, not raw closing_speed_kmh, to
        r['set_piece_evidence'] = None                   # avoid an apparently-contradictory display

        if uncertain_mask is not None and uncertain_mask[i]:
            r['category'] = 'UNCERTAIN'
            counts['UNCERTAIN'] += 1
            continue

        is_shot, shot_score = is_shot_frame(rows, shot_signal, ball_speeds, i)
        if is_shot:
            r['category'] = 'SHOT'
            r['shot_score'] = shot_score
            counts['SHOT'] += 1
            continue

        is_sp, sp_data = is_set_piece_frame(rows, ball_speeds, i, fps)
        if is_sp:
            r['category'] = 'SET_PIECE'
            r['set_piece_confidence'], r['set_piece_evidence'] = sp_data
            counts['SET_PIECE'] += 1
            continue

        if r['danger_smoothed'] >= DANGER_PEAK_THRESHOLD:
            r['category'] = 'OPEN_PLAY'
            counts['OPEN_PLAY'] += 1
        else:
            r['category'] = 'NONE'
            counts['NONE'] += 1

    return counts


def select_top_events(rows, category, score_key, fps, top_n,
                       min_sep_seconds=PEAK_MIN_SEPARATION_SECONDS,
                       clean_only=False, boundary_frames=0):
    """Greedy, min-separated top-N selection WITHIN one category, ranked by
    `score_key`. Generalizes the single-category peak selection the
    previous design only did for OPEN_PLAY."""
    n = len(rows)
    min_sep_frames = round(min_sep_seconds * fps)
    idxs = [i for i in range(n) if rows[i]['category'] == category]
    idxs.sort(key=lambda i: -(rows[i][score_key] or 0.0))

    selected = []
    for i in idxs:
        if boundary_frames and (i < boundary_frames or i >= n - boundary_frames):
            continue
        if clean_only and rows[i].get('count_anomaly'):
            continue
        if all(abs(i - s) >= min_sep_frames for s in selected):
            selected.append(i)
        if len(selected) >= top_n:
            break
    return selected


def determine_goal_sides(tracks, n):
    """Once per clip: which goal (x≈0 or x≈105) each team defends,
    decided by relative pooled mean x between the two teams (see
    analyze_formation_window.determine_goal_sides for why an absolute
    halfway threshold is unsafe) — computed here directly over raw
    per-frame team labels since no identity linking/merged-sid data
    exists in this script."""
    sums, counts = {1: 0.0, 2: 0.0}, {1: 0, 2: 0}
    for fn in range(n):
        for _pid, team, x, _y in extract_frame_players(tracks, fn):
            sums[team] += x
            counts[team] += 1
    means = {t: (sums[t] / counts[t] if counts[t] else None) for t in (1, 2)}
    order = sorted((1, 2), key=lambda t: means[t] if means[t] is not None else float('inf'))
    return {order[0]: True, order[1]: False}, means   # True = defends x≈0


# ── Zone grid (world-space, attacking third) ────────────────────────────────

def _build_zone_grid(x_lo, x_hi):
    xs = np.arange(x_lo, x_hi + ZONE_GRID_STEP_M / 2, ZONE_GRID_STEP_M)
    ys = np.arange(0.0, PITCH_WIDTH + ZONE_GRID_STEP_M / 2, ZONE_GRID_STEP_M)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)


# Precompute both possible attacking-third grids once (only two possible
# zones exist for a fixed pair of goal ends) rather than rebuilding per frame.
_ZONE_GRID_LOW  = _build_zone_grid(0.0, FINAL_THIRD_M)                    # near x=0 goal
_ZONE_GRID_HIGH = _build_zone_grid(PITCH_LENGTH - FINAL_THIRD_M, PITCH_LENGTH)  # near x=105 goal


def _hull_or_bbox_area(points):
    """Convex hull area (cv2, already a project dependency) for >=3 points;
    bounding-box area for exactly 2; 0 for 0-1 (can't form a 2D shape)."""
    pts = np.array(points, dtype=np.float32)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts)
        return float(cv2.contourArea(hull))
    if len(pts) == 2:
        w = float(abs(pts[0, 0] - pts[1, 0]))
        h = float(abs(pts[0, 1] - pts[1, 1]))
        return w * h
    return 0.0


def _clip01(v):
    return max(0.0, min(1.0, v))


# ── Per-frame danger score ──────────────────────────────────────────────────

def compute_frame_signals(players, ball_xy, prev_ball_xy, defends_low, fps):
    """Returns dict of raw signal values, normalized 0-100 scores, weighted
    contributions, and bookkeeping (attacking_team, n_attackers, n_defenders)
    for one frame. `players` is extract_frame_players()'s output list."""
    team1_players = [(x, y) for _pid, t, x, y in players if t == 1]
    team2_players = [(x, y) for _pid, t, x, y in players if t == 2]

    mid = PITCH_LENGTH / 2.0
    attacks_high = {t: defends_low[t] for t in (1, 2)}  # defend low -> attack high

    def territorial_presence(team_pts, team):
        if attacks_high[team]:
            return sum(max(0.0, x - mid) for x, _y in team_pts)
        return sum(max(0.0, mid - x) for x, _y in team_pts)

    presence1 = territorial_presence(team1_players, 1)
    presence2 = territorial_presence(team2_players, 2)

    if presence1 != presence2:
        attacking_team = 1 if presence1 > presence2 else 2
    else:
        # Tie (commonly 0-0, e.g. a settled midfield exchange): fall back to
        # which side of the pitch the ball is currently on.
        wants_high = ball_xy[0] > mid
        attacking_team = 1 if attacks_high[1] == wants_high else 2

    defending_team = 2 if attacking_team == 1 else 1
    attacker_pts = team1_players if attacking_team == 1 else team2_players
    defender_pts = team2_players if attacking_team == 1 else team1_players

    target_goal_x = 0.0 if defends_low[defending_team] else PITCH_LENGTH
    target_goal = (target_goal_x, GOAL_Y)

    # a. ball proximity to goal
    ball_dist = math.hypot(ball_xy[0] - target_goal[0], ball_xy[1] - target_goal[1])
    ball_proximity_raw = _clip01(1.0 - ball_dist / MAX_BALL_DIST_M)

    # b. attacking space control near goal (final-third grid, nearest-player)
    grid = _ZONE_GRID_LOW if target_goal_x == 0.0 else _ZONE_GRID_HIGH
    all_pts = attacker_pts + defender_pts
    all_teams = ([attacking_team] * len(attacker_pts)) + ([defending_team] * len(defender_pts))
    if len(all_pts) >= 2:
        positions = np.array(all_pts, dtype=np.float32)
        nearest_idx, _ = nearest_player_per_cell(grid, positions)
        owner_teams = np.array(all_teams)[nearest_idx]
        space_control_raw = float(np.mean(owner_teams == attacking_team))
    else:
        space_control_raw = 0.0   # not enough players to judge — neutral/no evidence

    # c. defensive compactness (LARGER area = more danger; see module docstring)
    hull_area = _hull_or_bbox_area(defender_pts)
    compactness_raw = _clip01(hull_area / MAX_HULL_AREA_M2)

    # d. numerical advantage inside the attacking third
    zone_lo, zone_hi = (0.0, FINAL_THIRD_M) if target_goal_x == 0.0 \
        else (PITCH_LENGTH - FINAL_THIRD_M, PITCH_LENGTH)
    n_att_zone = sum(1 for x, _y in attacker_pts if zone_lo <= x <= zone_hi)
    n_def_zone = sum(1 for x, _y in defender_pts if zone_lo <= x <= zone_hi)
    advantage = n_att_zone - n_def_zone
    numerical_raw = _clip01((advantage + MAX_NUMERICAL_ADV) / (2 * MAX_NUMERICAL_ADV))

    # e. ball velocity toward goal (signed closing speed, km/h)
    prev_dist = math.hypot(prev_ball_xy[0] - target_goal[0], prev_ball_xy[1] - target_goal[1])
    closing_speed_kmh = (prev_dist - ball_dist) * fps * 3.6
    velocity_raw = _clip01((closing_speed_kmh + MAX_CLOSING_SPEED_KMH) / (2 * MAX_CLOSING_SPEED_KMH))

    scores = {
        'ball_proximity':      100.0 * ball_proximity_raw,
        'space_control':       100.0 * space_control_raw,
        'compactness':         100.0 * compactness_raw,
        'numerical_advantage': 100.0 * numerical_raw,
        'ball_velocity':       100.0 * velocity_raw,
    }
    contributions = {k: WEIGHTS[k] * v for k, v in scores.items()}
    danger = sum(contributions.values())

    # No identity linking/deduplication happens in this script (by design —
    # see module docstring), so a raw-pid count above 11 for one team in a
    # single frame is physically impossible and signals a real artifact:
    # most commonly a ByteTrack ID handoff where a fading old id and its
    # replacement briefly co-exist in the same frame, or a referee/opponent
    # momentarily misclassified into the wrong jersey-colour cluster. This
    # doesn't get silently corrected — it's flagged so numerical_advantage
    # (and, to a lesser extent, space_control) can be discounted for that
    # specific frame rather than trusted at face value.
    count_anomaly = len(attacker_pts) > 11 or len(defender_pts) > 11

    return {
        'attacking_team': attacking_team,
        'n_attackers': len(attacker_pts),
        'n_defenders': len(defender_pts),
        'n_att_zone': n_att_zone,
        'n_def_zone': n_def_zone,
        'count_anomaly': count_anomaly,
        'ball_dist_to_goal_m': ball_dist,
        'closing_speed_kmh': closing_speed_kmh,
        'hull_area_m2': hull_area,
        'scores': scores,
        'contributions': contributions,
        'danger': danger,
    }


# ── Sparkline ────────────────────────────────────────────────────────────────

_SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def sparkline(values):
    vmax = 100.0  # fixed 0-100 scale, not per-series min/max, so bars are comparable across runs
    out = []
    for v in values:
        level = int(_clip01(v / vmax) * (len(_SPARK_CHARS) - 1))
        out.append(_SPARK_CHARS[level])
    return "".join(out)


# ── Main ─────────────────────────────────────────────────────────────────────

def run(source=DEFAULT_SOURCE, start_frame=0, end_frame=10**9, out_csv=None,
        top_n=5, clean_only=True, notes_path=None):
    print(f"analyze_danger_score: source='{source}'  requested window [{start_frame}:{end_frame})")

    t0 = time.time()
    tracks, video_path, fps, start_frame, end_frame = load_window_tracks(
        start_frame, end_frame, source=source)
    n = end_frame - start_frame
    print(f"Loaded + position-transformed window [{start_frame}:{end_frame}) "
          f"({n} frames) in {time.time() - t0:.2f}s")

    t1 = time.time()
    print(f"Streaming {video_path} for window frames only (jersey-colour team assignment)...")
    assign_teams_for_window(tracks, video_path, start_frame, end_frame)
    print(f"Team assignment took {time.time() - t1:.1f}s")

    defends_low, goal_side_means = determine_goal_sides(tracks, n)
    print(f"Goal sides (pooled mean x): team1={goal_side_means[1]}, team2={goal_side_means[2]}  "
          f"-> team1 defends {'x≈0' if defends_low[1] else 'x≈105'}, "
          f"team2 defends {'x≈0' if defends_low[2] else 'x≈105'}")

    ball_xs, ball_ys = build_ball_trajectory(tracks, n)
    ball_speeds = compute_ball_speeds(ball_xs, ball_ys, fps)

    print("\nFormula: danger = "
          + " + ".join(f"{w:.2f}*{k}" for k, w in WEIGHTS.items())
          + "  (each signal normalized to 0-100 before weighting)")
    for k, w in WEIGHTS.items():
        print(f"  {k:22s} weight={w:.2f}")

    rows = []
    for fn in range(n):
        players = extract_frame_players(tracks, fn)
        ball_xy = (ball_xs[fn], ball_ys[fn])
        prev_ball_xy = (ball_xs[fn - 1], ball_ys[fn - 1]) if fn > 0 else ball_xy
        result = compute_frame_signals(players, ball_xy, prev_ball_xy, defends_low, fps)
        result['frame'] = start_frame + fn
        result['t_sec'] = fn / fps
        result['ball_speed_kmh'] = float(ball_speeds[fn])
        rows.append(result)

    raw_scores = np.array([r['danger'] for r in rows])
    smoothed = pd.Series(raw_scores).rolling(
        SMOOTH_WINDOW_FRAMES, center=True, min_periods=1).mean().to_numpy()
    smoothed_contribs = {
        k: pd.Series([r['contributions'][k] for r in rows]).rolling(
            SMOOTH_WINDOW_FRAMES, center=True, min_periods=1).mean().to_numpy()
        for k in WEIGHTS
    }
    for i, r in enumerate(rows):
        r['danger_smoothed'] = float(smoothed[i])
        r['contributions_smoothed'] = {k: float(smoothed_contribs[k][i]) for k in WEIGHTS}

    n_anomalous = sum(1 for r in rows if r['count_anomaly'])
    print(f"\nCount anomaly (>11 raw detections for one team, post dedup/referee/touchline "
          f"exclusion): {n_anomalous}/{n} frames ({100.0 * n_anomalous / n:.1f}%)")

    # ── Console preview: duration-adaptive so a 42k-frame match doesn't dump
    # thousands of lines — targets ~TARGET_CONSOLE_PREVIEW_LINES rows either way.
    preview_every = max(1, round(n / TARGET_CONSOLE_PREVIEW_LINES))
    print(f"\nFrame-by-frame danger score (every {preview_every} frames shown here "
          f"[~{preview_every / fps:.1f}s]; full per-frame log written to CSV):")
    print(f"  {'frame':>6}  {'t':>9}  {'raw':>6}  {'smoothed':>8}  attacking-team")
    for r in rows[::preview_every]:
        mm, ss = divmod(r['t_sec'], 60)
        hh, mm = divmod(int(mm), 60)
        print(f"  {r['frame']:6d}  {hh:02d}:{mm:02d}:{ss:05.2f}  {r['danger']:6.1f}  "
              f"{r['danger_smoothed']:8.1f}  team {r['attacking_team']}")

    # ── Sparkline: duration-adaptive bin size (~TARGET_SPARKLINE_BARS bars) ──
    total_secs = n / fps
    bin_seconds = max(1, round(total_secs / TARGET_SPARKLINE_BARS))
    per_second = pd.Series(smoothed).groupby(np.arange(n) // round(fps * bin_seconds)).mean()
    total_secs = len(per_second)
    n_bars = len(per_second)
    total_min = total_secs / 60.0
    print(f"\nDanger score over time ({bin_seconds}s/bar, {n_bars} bars, "
          f"{total_min:.1f} min total, 0-100 scale, '{_SPARK_CHARS[-1]}'=100):")
    print("  " + sparkline(per_second.to_numpy()))
    end_label = f"{int(total_min):02d}:{int(total_secs % 60):02d}"
    print(f"  ^00:00{' ' * max(0, n_bars - 11)}{end_label}^")

    # ── Boundary exclusion setup (shared by all three categories' reporting) ─
    min_sep_frames = round(PEAK_MIN_SEPARATION_SECONDS * fps)
    boundary_frames = round(BOUNDARY_EXCLUSION_SECONDS * fps)
    if 2 * boundary_frames >= n:
        # Window too short for a 20s-each-side exclusion to leave anything
        # (e.g. a 30s clip) — disable rather than silently reporting zero
        # peaks for the whole window.
        print(f"\nNOTE: window is only {n / fps:.1f}s, too short for a "
              f"{BOUNDARY_EXCLUSION_SECONDS}s-per-side boundary exclusion "
              f"({2 * BOUNDARY_EXCLUSION_SECONDS}s total) — boundary exclusion disabled "
              f"for this run.")
        boundary_frames = 0
    in_boundary = lambda i: i < boundary_frames or i >= n - boundary_frames

    # ── UNCERTAIN: long ball-detection gaps, checked BEFORE classification
    # so they pre-empt every other category (see UNCERTAIN_GAP_MAX_FRAMES) ──
    detect_every = SOURCES[source]['detect_every']
    uncertain_gaps = find_uncertain_gaps(tracks, n, start_frame, detect_every)
    uncertain_mask = np.zeros(n, dtype=bool)
    for g_start, g_end in uncertain_gaps:
        uncertain_mask[g_start:g_end + 1] = True

    def _fmt_hms(t):
        m, s = divmod(t, 60)
        h, m = divmod(int(m), 60)
        return f"{h:02d}:{m:02d}:{s:05.2f}"

    if notes_path:
        with open(notes_path, 'w', encoding='utf-8') as f:
            f.write(f"UNCERTAIN ball-tracking windows — source='{source}' "
                     f"window=[{start_frame}:{end_frame})\n")
            f.write(f"(>{UNCERTAIN_GAP_MAX_FRAMES} consecutive frames / "
                     f"{UNCERTAIN_GAP_MAX_FRAMES/fps:.2f}s with no raw, non-interpolated "
                     f"ball detection. NOT shown in the video overlay — logged here instead.)\n")
            f.write(f"{len(uncertain_gaps)} window(s) found.\n\n")
            for idx, (g_start, g_end) in enumerate(uncertain_gaps, start=1):
                t_start = (start_frame + g_start) / fps
                t_end = (start_frame + g_end) / fps
                dur = t_end - t_start + (1.0 / fps)
                f.write(f"  #{idx}  {_fmt_hms(t_start)} - {_fmt_hms(t_end)}  "
                        f"({dur:.2f}s, {g_end - g_start + 1} frames)   "
                        f"frames {start_frame + g_start}-{start_frame + g_end}\n")
        print(f"\nWrote {len(uncertain_gaps)} UNCERTAIN window(s) to {notes_path}")

    # ── Four-category classification: every frame gets exactly one of
    # UNCERTAIN / SHOT / SET_PIECE / OPEN_PLAY / NONE (see classify_frames
    # docstring for the precedence order — UNCERTAIN pre-empts everything,
    # SET_PIECE frames never become OPEN_PLAY). UNCERTAIN is internal/notes
    # only: it must render as blank/neutral in the video overlay, never as
    # an on-screen label — see classify_frames' NOTE. ─────────────────────
    counts = classify_frames(rows, ball_speeds, fps, uncertain_mask=uncertain_mask)
    print(f"\nFrame classification (full window, {n} frames):")
    for cat in ('UNCERTAIN', 'SHOT', 'SET_PIECE', 'OPEN_PLAY', 'NONE'):
        print(f"  {cat:10s}: {counts[cat]:6d} frames  ({100.0 * counts[cat] / n:5.1f}%)")

    # ── Retrospective: how many of the OLD single-formula "high danger"
    # moments (composite score alone, exactly the previous design's
    # criterion) turn out to be SET_PIECE or SHOT under the new
    # classification — quantifies how much dead-ball congestion was
    # inflating the old blended ranking ──────────────────────────────────
    old_style_candidates = sorted(range(n), key=lambda i: -rows[i]['danger_smoothed'])
    old_peaks = []
    n_boundary_skipped = 0
    for i in old_style_candidates:
        if rows[i]['danger_smoothed'] < DANGER_PEAK_THRESHOLD:
            break
        if in_boundary(i):
            n_boundary_skipped += 1
            continue
        if clean_only and rows[i]['count_anomaly']:
            continue
        if all(abs(i - p) >= min_sep_frames for p in old_peaks):
            old_peaks.append(i)

    if old_peaks:
        old_now_shot = sum(1 for i in old_peaks if rows[i]['category'] == 'SHOT')
        old_now_setpiece = sum(1 for i in old_peaks if rows[i]['category'] == 'SET_PIECE')
        old_now_openplay = sum(1 for i in old_peaks if rows[i]['category'] == 'OPEN_PLAY')
        old_now_uncertain = sum(1 for i in old_peaks if rows[i]['category'] == 'UNCERTAIN')
        print(f"\nRetrospective on the OLD single-formula peak list "
              f"({len(old_peaks)} moments that would have all been reported as "
              f"undifferentiated 'high danger' before this redesign):")
        print(f"  still OPEN_PLAY : {old_now_openplay}/{len(old_peaks)} "
              f"({100.0*old_now_openplay/len(old_peaks):.0f}%)")
        print(f"  reclassified SET_PIECE: {old_now_setpiece}/{len(old_peaks)} "
              f"({100.0*old_now_setpiece/len(old_peaks):.0f}%) — "
              f"these were scored as organized attacking danger but weren't")
        print(f"  reclassified SHOT     : {old_now_shot}/{len(old_peaks)} "
              f"({100.0*old_now_shot/len(old_peaks):.0f}%) — "
              f"these are real strikes that happened to also clear the open-play bar")
        print(f"  reclassified UNCERTAIN: {old_now_uncertain}/{len(old_peaks)} "
              f"({100.0*old_now_uncertain/len(old_peaks):.0f}%) — "
              f"these scored 'high danger' on fabricated interpolated ball data")

    def _print_moment(rank, i, extra_line=None):
        r = rows[i]
        mm, ss = divmod(r['t_sec'], 60)
        print(f"\n  #{rank}  frame {r['frame']}  t={int(mm):02d}:{ss:05.2f}")
        if extra_line:
            print(extra_line)
        if r['count_anomaly']:
            print(f"      ** WARNING: {max(r['n_attackers'], r['n_defenders'])} raw detections "
                  f"for one team this frame — physically impossible (max 11). **")

    # ── OPEN PLAY: unchanged composite formula/smoothing, ranked by
    # danger_smoothed exactly as the original design did ───────────────────
    top_open_play = select_top_events(rows, 'OPEN_PLAY', 'danger_smoothed', fps, top_n,
                                       clean_only=clean_only, boundary_frames=boundary_frames)
    print(f"\n{'='*70}\nOPEN PLAY — top {len(top_open_play)} "
          f"(of {counts['OPEN_PLAY']} qualifying frames)\n{'='*70}")
    for rank, i in enumerate(top_open_play, start=1):
        r = rows[i]
        _print_moment(rank, i,
                      f"      danger={r['danger_smoothed']:.1f}  "
                      f"(attacking team {r['attacking_team']}, "
                      f"{r['n_attackers']}v{r['n_defenders']} overall, "
                      f"{r['n_att_zone']}v{r['n_def_zone']} in the final third)")
        contribs = r['contributions_smoothed']
        ranked_signals = sorted(contribs.items(), key=lambda kv: -kv[1])
        for sig, contrib in ranked_signals:
            pct_of_total = 100.0 * contrib / r['danger_smoothed'] if r['danger_smoothed'] else 0.0
            print(f"      {sig:22s} contributed {contrib:5.1f} pts "
                  f"({pct_of_total:4.1f}% of this moment's score)  "
                  f"[raw signal: {r['scores'][sig]:5.1f}/100]")

    # ── SET PIECE: own confidence metric, NOT the open-play score ──────────
    top_set_piece = select_top_events(rows, 'SET_PIECE', 'set_piece_confidence', fps, top_n,
                                       clean_only=clean_only, boundary_frames=boundary_frames)
    print(f"\n{'='*70}\nSET PIECE — top {len(top_set_piece)} "
          f"(of {counts['SET_PIECE']} qualifying frames)\n{'='*70}")
    for rank, i in enumerate(top_set_piece, start=1):
        r = rows[i]
        ev = r['set_piece_evidence']
        _print_moment(rank, i,
                      f"      confidence={r['set_piece_confidence']:.1f}/100  "
                      f"ball static {ev['ball_static_fraction']*100:.0f}% of prior "
                      f"{SET_PIECE_LOOKBACK_SECONDS}s (median {ev['median_ball_speed_kmh']:.1f}km/h), "
                      f"hull-area CoV={ev['hull_area_cov']:.2f}, "
                      f"dist_to_goal={r['ball_dist_to_goal_m']:.1f}m")

    # ── SHOT: own velocity+proximity score, independent of danger_smoothed ─
    top_shot = select_top_events(rows, 'SHOT', 'shot_score', fps, top_n,
                                  clean_only=False, boundary_frames=boundary_frames)
    print(f"\n{'='*70}\nSHOT — top {len(top_shot)} "
          f"(of {counts['SHOT']} qualifying frames)\n{'='*70}")
    for rank, i in enumerate(top_shot, start=1):
        r = rows[i]
        _print_moment(rank, i,
                      f"      shot_score={r['shot_score']:.1f}/100  "
                      f"closing_speed(smoothed,{SHOT_SMOOTH_WINDOW_FRAMES}f)={r['shot_signal_kmh']:.1f}km/h  "
                      f"[raw this frame: {r['closing_speed_kmh']:.1f}km/h]  "
                      f"dist_to_goal={r['ball_dist_to_goal_m']:.1f}m")

    if out_csv:
        df = pd.DataFrame([{
            'frame': r['frame'], 't_sec': r['t_sec'],
            'danger_raw': r['danger'], 'danger_smoothed': r['danger_smoothed'],
            'category': r['category'],
            'shot_score': r['shot_score'],
            'shot_signal_kmh': r['shot_signal_kmh'],
            'set_piece_confidence': r['set_piece_confidence'],
            'attacking_team': r['attacking_team'],
            'n_attackers': r['n_attackers'], 'n_defenders': r['n_defenders'],
            'count_anomaly': r['count_anomaly'],
            'ball_dist_to_goal_m': r['ball_dist_to_goal_m'],
            'closing_speed_kmh': r['closing_speed_kmh'],
            'ball_speed_kmh': r['ball_speed_kmh'],
            'hull_area_m2': r['hull_area_m2'],
            **{f'score_{k}': r['scores'][k] for k in WEIGHTS},
        } for r in rows])
        df.to_csv(out_csv, index=False)
        print(f"\nFull per-frame log written to {out_csv}")

    print()
    return {
        'rows': rows,
        'category_counts': counts,
        'open_play_peaks': [rows[i] for i in top_open_play],
        'set_piece_events': [rows[i] for i in top_set_piece],
        'shot_events': [rows[i] for i in top_shot],
        'uncertain_gaps': [{
            'start_frame': start_frame + g_start, 'end_frame': start_frame + g_end,
            't_start': (start_frame + g_start) / fps, 't_end': (start_frame + g_end) / fps,
            'n_frames': g_end - g_start + 1,
        } for g_start, g_end in uncertain_gaps],
        'fps': fps,
        'video_path': video_path,
        'start_frame': start_frame,
        'end_frame': end_frame,
    }


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Per-frame danger score (0-100) from spatial data.")
    p.add_argument('start_frame', type=int, nargs='?', default=0)
    p.add_argument('end_frame', type=int, nargs='?', default=10**9)
    p.add_argument('--source', choices=sorted(SOURCES), default=DEFAULT_SOURCE)
    p.add_argument('--csv', default=None, help="Path to write the full per-frame CSV log.")
    p.add_argument('--diagnose', type=int, default=None, metavar='FRAME',
                    help="Print the raw-pid/team/position diagnostic for one frame "
                         "(pre/post within-frame dedup) and exit, instead of running "
                         "the full analysis.")
    p.add_argument('--top-n', type=int, default=5, help="How many peak moments to print in detail.")
    p.add_argument('--include-flagged', action='store_true',
                    help="Allow count_anomaly frames into the peak ranking (default: skip them).")
    p.add_argument('--notes', default=None,
                    help="Path to write the UNCERTAIN ball-tracking-gap notes file.")
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    if args.diagnose is not None:
        tracks, video_path, fps, start_frame, end_frame = load_window_tracks(
            args.start_frame, args.end_frame, source=args.source)
        assign_teams_for_window(tracks, video_path, start_frame, end_frame)
        diagnose_frame(tracks, args.diagnose - start_frame)
    else:
        out_csv = args.csv or f"danger_score_{args.source}.csv"
        notes_path = args.notes or f"danger_score_notes_{args.source}.txt"
        run(source=args.source, start_frame=args.start_frame, end_frame=args.end_frame,
            out_csv=out_csv, top_n=args.top_n, clean_only=not args.include_flagged,
            notes_path=notes_path)
