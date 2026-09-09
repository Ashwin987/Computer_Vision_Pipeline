"""
corner_kicks.py — manual corner-kick marking, team-shape visualization, and
real-position-based metrics.

Scope, firm (see the original feature prompt for the full reasoning):
- Manual marking only. No automatic corner detection.
- NO player-role inference, ever — no near-post/far-post, no zonal/man-
  marking assignment. Every output here is either a real tracked position,
  a real distance between two real positions, or a purely geometric shape
  (a convex hull) computed from those positions. Nothing here claims to
  know what a player's job was.
- A mark's team-shape/metrics data requires that corner's window to have
  already been run through the CV pipeline AND through
  reconstruct_positions.py (see that script's docstring for why a second,
  offline step is needed - the pipeline's own tracks.pkl stub predates
  position-transformation/team-assignment). This module never launches CV
  processing itself; a mark with no player_positions.json yet is shown
  with an honest "not processed yet" message, never a broken render.

Storage: marks are written to CACHE_DIR (the writable system-temp location
- see app.py's CACHE_DIR comment for why: Streamlit Community Cloud mounts
the repo read-only), same ephemeral-across-restarts tradeoff already
accepted for training plans and curated-match renames. Never bundle.json.
"""
import colorsys
import json
import math
import os
from collections import defaultdict, deque
from pathlib import Path

import re

import cv2
import numpy as np

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


def safe_filename_part(s):
    """A mark id/timestamp label (e.g. '3:06-3:15') is a fine JSON key or
    UI label as-is, but is NOT a safe filename component - confirmed live:
    a colon in a Windows path made cv2.VideoWriter fail to open silently
    (isOpened() False, no exception, zero bytes written, no error surfaced
    until a later, unrelated-looking "couldn't prepare for playback"
    message). Anything derived from a mark id that becomes part of a
    filesystem path must go through this first."""
    return re.sub(r'[^A-Za-z0-9_-]', '_', str(s))


# ==========================================
# MARK STORAGE (per match, writable)
# ==========================================

def _marks_path(cache_dir, match_key):
    return Path(cache_dir) / "corner_kicks" / f"{match_key}.json"


def _seed_marks_path(curated_matches_dir, match_key):
    return Path(curated_matches_dir) / match_key / "corner_kicks_seed.json"


def load_marks(cache_dir, match_key, curated_matches_dir=None):
    """Writable CACHE_DIR wins once anything's been saved there. Until then,
    falls back to a committed seed file under curated_matches/<key>/ if one
    exists - same "committed default, writable override" shape
    training_plan.py already uses for the same CACHE_DIR-resets-on-
    redeploy problem. Needed because 3 real corners were already CV-
    processed for the liverpool_psg curated match, but a mark recorded only
    in CACHE_DIR would vanish on Streamlit Community Cloud's next container
    restart, silently orphaning that already-committed CV data."""
    path = _marks_path(cache_dir, match_key)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    if curated_matches_dir:
        seed_path = _seed_marks_path(curated_matches_dir, match_key)
        if seed_path.exists():
            try:
                with open(seed_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
    return []


def save_marks(cache_dir, match_key, marks):
    path = _marks_path(cache_dir, match_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(marks, f, indent=2)
    os.replace(tmp, path)


def upsert_mark(cache_dir, match_key, mark_id, timestamp_label, attacking_team, defending_team, cv_output_dir=None):
    """attacking_team/defending_team are 'team_a'/'team_b' tokens (the same
    stable convention cv_team_mapping already uses), never a raw color or a
    literal team name - a rename must never orphan an existing mark.
    cv_output_dir is the folder name under cv_pipeline/output_videos/ once
    that window has actually been processed - None until then."""
    marks = [m for m in load_marks(cache_dir, match_key) if m["id"] != mark_id]
    marks.append({
        "id": mark_id,
        "timestamp_label": timestamp_label,
        "attacking_team": attacking_team,
        "defending_team": defending_team,
        "cv_output_dir": cv_output_dir,
    })
    marks.sort(key=lambda m: m["id"])
    save_marks(cache_dir, match_key, marks)
    return marks


def delete_mark(cache_dir, match_key, mark_id):
    marks = [m for m in load_marks(cache_dir, match_key) if m["id"] != mark_id]
    save_marks(cache_dir, match_key, marks)
    return marks


# ==========================================
# POSITION DATA + TEAM MAPPING
# ==========================================

def load_player_positions(cv_pipeline_dir, cv_output_dir):
    """None if this mark's window hasn't been through
    reconstruct_positions.py yet - a real, expected state (a freshly-marked
    corner with no CV run behind it), not an error."""
    if not cv_output_dir:
        return None
    path = Path(cv_pipeline_dir) / "output_videos" / cv_output_dir / "player_positions.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def resolve_corner_team_mapping(positions_data, reference_team_colors_bgr):
    """Which of THIS corner's own team-1/team-2 jersey-color clusters is
    team_a vs team_b. Each corner is its own independent CV run with its
    own from-scratch K-means color clustering, so team1/team2 are NOT
    guaranteed to line up with the match's overall assignment - confirmed
    directly: of the 3 real corners processed for this feature, one had
    team1/team2 inverted relative to the other two and to the main match
    (cost ratio of ~20x between the two possible mappings - not a close
    call, genuinely swapped). Resolved by nearest-color match against the
    match's own already-confirmed reference colors (its main analyzed
    window's team_resolution.team_colors_bgr + cv_team_mapping), never
    assumed to be index-stable across separate runs."""
    def dist(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    c1 = positions_data["team_colors_bgr"]["1"]
    c2 = positions_data["team_colors_bgr"]["2"]
    ref_a = reference_team_colors_bgr["team1"]
    ref_b = reference_team_colors_bgr["team2"]
    cost_a = dist(c1, ref_a) + dist(c2, ref_b)
    cost_b = dist(c1, ref_b) + dist(c2, ref_a)
    return {"1": "team_a", "2": "team_b"} if cost_a <= cost_b else {"1": "team_b", "2": "team_a"}


def _team_num_for(team_mapping, token):
    return next(int(k) for k, v in team_mapping.items() if v == token)


# ==========================================
# METRICS — real positions/distances and honest geometry only
# ==========================================

# Same validation rule the CV pipeline itself already establishes and
# relies on elsewhere (pitch_calibrator.py's homography sanity check;
# stated explicitly in the technical report, Section 2.4.2: "any computed
# position falling outside [0, 105] x [0, 68] is flagged as the product of
# an unreliable homography for that frame and is discarded before it can
# corrupt a downstream metric"). Found necessary here by direct inspection:
# corner3's reconstructed positions include frames with wildly out-of-
# bounds values (x as extreme as -1459 or +718, once observed) - almost
# certainly the exact failure mode the report's Section 2.4.7 already
# documents (calibration degrades on frames where visible pitch markings
# are confined to a small area, which a tight corner-kick camera angle is
# close to by definition) rather than a new failure. Filtering here is
# applying the pipeline's own existing rule to a place it hadn't been
# applied yet, not inventing a new one.
_BOUNDS_MARGIN_M = 2.0  # small slack for legitimate near-touchline/goal-line positions


def _in_bounds(x, y):
    return (-_BOUNDS_MARGIN_M <= x <= PITCH_LENGTH_M + _BOUNDS_MARGIN_M
            and -_BOUNDS_MARGIN_M <= y <= PITCH_WIDTH_M + _BOUNDS_MARGIN_M)


def _outfield(frame, team_num):
    return [(p["x"], p["y"]) for p in frame
            if p["team"] == team_num and not p["is_goalkeeper"] and _in_bounds(p["x"], p["y"])]


def _outfield_with_ids(frame, team_num):
    return [(p["player_id"], (p["x"], p["y"])) for p in frame
            if p["team"] == team_num and not p["is_goalkeeper"] and _in_bounds(p["x"], p["y"])]


def _compactness(positions):
    """Mean pairwise distance between players - a simple, honest spread
    metric (chosen over an enclosed-hull-area calculation specifically
    because it's simpler to get right from the same position data, per the
    feature spec's own explicit either/or)."""
    if len(positions) < 2:
        return None
    dists = [math.dist(positions[i], positions[j])
              for i in range(len(positions)) for j in range(i + 1, len(positions))]
    return sum(dists) / len(dists)


def _goal_line_x(positions):
    """Which end (x=0 or x=105) this team is defending - inferred purely
    from where THEIR OWN tracked players are clustered on the real
    105x68m pitch, not an assumption about match direction."""
    mean_x = sum(x for x, _ in positions) / len(positions)
    return 0.0 if mean_x < PITCH_LENGTH_M / 2 else PITCH_LENGTH_M


# Real penalty-box dimensions, same numbers _pitch_base() already draws the
# box outline with (cv2.rectangle(pt(0, 13.84), pt(16.5, 54.16), ...)) -
# not a separate/invented definition.
_BOX_DEPTH_M = 16.5
_BOX_Y_MIN_M = 13.84
_BOX_Y_MAX_M = 54.16
# Found necessary by direct inspection: without any zone filter,
# last-defender-distance and compactness were being averaged across EVERY
# tracked outfield player on a team, including ones nowhere near the
# corner setup (e.g. an out-ball outlet left near halfway) - producing
# nonsensical numbers like "PSG's last defender is 37.41m from goal" for a
# corner, where every genuine defender is clustered near their own goal.
# This buffer is deliberately modest: wide enough to keep a real edge-of-
# box defender or near-post zonal marker who is standing just outside the
# box line, not wide enough to pull in a player who is clearly out of the
# play (e.g. 30-50m away near the halfway line).
_BOX_BUFFER_M = 5.0


def _in_box_zone(x, y, goal_x):
    """goal_x is the end (0.0 or 105.0) the corner is being played at -
    the SAME end for both teams, since a corner-kick setup only exists
    relative to one goal. Both teams' players are scoped to this same
    zone: the attacking team also only has some players actually up for
    the corner, with others potentially held back for a counter-attack."""
    if goal_x <= PITCH_LENGTH_M / 2:
        x_ok = x <= _BOX_DEPTH_M + _BOX_BUFFER_M
    else:
        x_ok = x >= PITCH_LENGTH_M - _BOX_DEPTH_M - _BOX_BUFFER_M
    return x_ok and (_BOX_Y_MIN_M - _BOX_BUFFER_M) <= y <= (_BOX_Y_MAX_M + _BOX_BUFFER_M)


def compute_corner_metrics(positions_data, team_mapping, attacking_team_token, defending_team_token):
    """Every value is averaged across every frame with resolved data for
    both the relevant team(s), rather than picked from one 'moment of
    delivery' frame - more robust to single-frame tracking noise, and
    honestly labeled as a window average rather than asserting a specific
    instant this data can't reliably pin down on its own.

    All three metrics (last-defender distance, compactness, marking
    distance) are scoped to players in/near the penalty box for that
    frame (_in_box_zone) BEFORE any distance is computed - a team's full
    outfield complement, wherever they happen to be on the pitch, is not
    what "how deep is the defense" or "how tight is this team's shape at
    the corner" is supposed to mean."""
    attacking_num = _team_num_for(team_mapping, attacking_team_token)
    defending_num = _team_num_for(team_mapping, defending_team_token)

    last_defender_vals, att_compact_vals, def_compact_vals = [], [], []
    marking_by_pid = {}

    for frame in positions_data["frames"]:
        att_full = _outfield(frame, attacking_num)
        def_full_with_ids = _outfield_with_ids(frame, defending_num)
        def_full = [pos for _, pos in def_full_with_ids]
        if not def_full:
            continue  # no defenders tracked this frame - can't tell which goal is in play

        goal_x = _goal_line_x(def_full)
        att = [pos for pos in att_full if _in_box_zone(pos[0], pos[1], goal_x)]
        def_with_ids = [(pid, pos) for pid, pos in def_full_with_ids if _in_box_zone(pos[0], pos[1], goal_x)]
        defn = [pos for _, pos in def_with_ids]

        if defn:
            last_defender_vals.append(min(abs(x - goal_x) for x, _ in defn))
        if len(att) >= 2:
            att_compact_vals.append(_compactness(att))
        if len(defn) >= 2:
            def_compact_vals.append(_compactness(defn))
        if defn and att:
            for pid, pos in def_with_ids:
                marking_by_pid.setdefault(pid, []).append(min(math.dist(pos, a) for a in att))

    def _avg(vals):
        return round(sum(vals) / len(vals), 2) if vals else None

    marking_distances = sorted(
        [{"player_id": pid, "distance_m": round(sum(v) / len(v), 2)} for pid, v in marking_by_pid.items()],
        key=lambda d: d["distance_m"],
    )

    return {
        "last_defender_distance_m": _avg(last_defender_vals),
        "attacking_compactness_m": _avg(att_compact_vals),
        "defending_compactness_m": _avg(def_compact_vals),
        "marking_distances": marking_distances,
        "n_frames_with_data": len(last_defender_vals),
        "n_frames_total": len(positions_data["frames"]),
    }


# ==========================================
# TEAM-SHAPE VIDEO — top-down pitch diagram, real positions, convex-hull
# outline (a purely geometric "connect the outermost players" shape - never
# a tactical-role assignment).
# ==========================================

def _pitch_base(px_per_m=10):
    w, h = int(PITCH_LENGTH_M * px_per_m), int(PITCH_WIDTH_M * px_per_m)
    img = np.full((h, w, 3), (34, 139, 34), dtype=np.uint8)  # grass green, BGR
    line = (255, 255, 255)

    def pt(x, y):
        return (int(x * px_per_m), int(y * px_per_m))

    cv2.rectangle(img, pt(0, 0), pt(PITCH_LENGTH_M, PITCH_WIDTH_M), line, 2)
    cv2.line(img, pt(52.5, 0), pt(52.5, PITCH_WIDTH_M), line, 2)
    cv2.circle(img, pt(52.5, 34), int(9.15 * px_per_m), line, 2)
    cv2.rectangle(img, pt(0, 13.84), pt(16.5, 54.16), line, 2)
    cv2.rectangle(img, pt(0, 24.84), pt(5.5, 43.16), line, 2)
    cv2.rectangle(img, pt(88.5, 13.84), pt(105, 54.16), line, 2)
    cv2.rectangle(img, pt(99.5, 24.84), pt(105, 43.16), line, 2)
    return img, px_per_m


def render_team_shape_video(positions_data, team_mapping, attacking_team_token, defending_team_token,
                             attacking_label, defending_label, out_path, fps=25.0):
    """out_path should end in .avi - written XVID, same convention
    run_cv_analysis.py's save_video() already uses for every other
    rendered output in this app. NOT written directly as a browser-
    playable mp4: confirmed live that cv2.VideoWriter's mp4v output isn't
    reliably playable by Streamlit's st.video() (MediaFileStorageError).
    The caller must run this through app.py's existing
    _ensure_browser_playable_video (the same moviepy-based transcode
    every other CV-rendered video already goes through) rather than
    displaying it directly - reusing that fix, not re-solving it here."""
    attacking_num = _team_num_for(team_mapping, attacking_team_token)
    defending_num = _team_num_for(team_mapping, defending_team_token)
    attacking_color = (0, 140, 255)   # orange, BGR
    defending_color = (60, 60, 230)   # red, BGR

    base_img, px_per_m = _pitch_base()
    h, w = base_img.shape[:2]

    def pt(x, y):
        return (int(x * px_per_m), int(y * px_per_m))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    try:
        for frame in positions_data["frames"]:
            img = base_img.copy()
            att = _outfield(frame, attacking_num)
            defn = _outfield(frame, defending_num)

            for positions, color in ((att, attacking_color), (defn, defending_color)):
                pts = [pt(x, y) for x, y in positions]
                for p in pts:
                    cv2.circle(img, p, 6, color, -1)
                if len(pts) >= 3:
                    hull = cv2.convexHull(np.array(pts, dtype=np.int32))
                    cv2.polylines(img, [hull], isClosed=True, color=color, thickness=2)
                elif len(pts) == 2:
                    cv2.line(img, pts[0], pts[1], color, 2)

            cv2.putText(img, f"{attacking_label} (attacking)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, attacking_color, 2)
            cv2.putText(img, f"{defending_label} (defending)", (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, defending_color, 2)
            writer.write(img)
    finally:
        writer.release()


# ==========================================
# TRAILS-ONLY VIDEO — clean-pitch movement trails, no player markers, no
# footage underneath. Same real position_transformed data the team-shape
# video already uses (not the CV pipeline's own render_output6.py, which
# operates on raw camera-pixel coordinates with its own camera-movement
# compensation - unnecessary here since position_transformed is already in
# real, camera-motion-independent pitch meters). Same fading-trail concept
# (short history, fades oldest-to-newest) as render_output6, re-expressed
# against this feature's own already-available data instead of importing
# cv_pipeline code the dashboard environment (no ultralytics/torch) can't
# run anyway.
# ==========================================

_TRAIL_SECONDS = 3.0   # matches render_output6.py's own trail-history window
_VARIANT_BUCKETS = 7   # small per-player hue/value jitter so two teammates
                        # whose trails cross don't read as one indistinct line
# Found necessary by direct visual inspection of a real rendered clip: without
# this, a tracker-id gap or reassignment (the same fragmentation issue
# render_output6.py's own RECORD_JUMP_PX/TRAIL_EXPIRE_FRAMES guard against)
# produced dead-straight lines connecting two totally unrelated pitch
# locations, since a raw tracker id can reappear later at a real player's
# CURRENT position with nothing recorded in between. Well above any plausible
# single real movement at 25fps (sprinting is ~0.4m/frame) but far below a
# near-pitch-length artifact.
_MAX_FRAME_GAP = 25     # ~1 real second - a longer absence means "different sighting"
_MAX_JUMP_M = 5.0


def _variant_color(base_bgr, seed):
    b, g, r = base_bgr
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    hue_bucket = int(seed) % _VARIANT_BUCKETS
    val_bucket = (int(seed) * 3) % _VARIANT_BUCKETS
    half = (_VARIANT_BUCKETS - 1) / 2.0
    h2 = (h + ((hue_bucket - half) / half) * 0.05) % 1.0
    v2 = min(1.0, max(0.55, v + ((val_bucket - half) / half) * 0.18))
    r2, g2, b2 = colorsys.hsv_to_rgb(h2, s, v2)
    return (int(round(b2 * 255)), int(round(g2 * 255)), int(round(r2 * 255)))


def render_trails_only_video(positions_data, team_mapping, attacking_team_token, defending_team_token,
                              attacking_label, defending_label, out_path, fps=25.0):
    """Same out_path/XVID/_ensure_browser_playable_video convention as
    render_team_shape_video - see that function's docstring."""
    attacking_num = _team_num_for(team_mapping, attacking_team_token)
    defending_num = _team_num_for(team_mapping, defending_team_token)
    attacking_color = (0, 140, 255)   # orange, BGR - same convention as team-shape
    defending_color = (60, 60, 230)   # red, BGR

    base_img, px_per_m = _pitch_base()
    h, w = base_img.shape[:2]

    def pt(x, y):
        return (int(x * px_per_m), int(y * px_per_m))

    trail_len = max(2, int(round(fps * _TRAIL_SECONDS)))
    trails = defaultdict(lambda: deque(maxlen=trail_len))  # (team_num, player_id) -> deque[(x,y)]
    last_point = {}  # (team_num, player_id) -> (frame_num, x, y), for gap/jump detection

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    try:
        for frame_num, frame in enumerate(positions_data["frames"]):
            img = base_img.copy()
            for team_num, color in ((attacking_num, attacking_color), (defending_num, defending_color)):
                for pid, (x, y) in _outfield_with_ids(frame, team_num):
                    key = (team_num, pid)
                    prev = last_point.get(key)
                    if prev is not None:
                        prev_fn, px, py = prev
                        if (frame_num - prev_fn > _MAX_FRAME_GAP
                                or math.hypot(x - px, y - py) > _MAX_JUMP_M):
                            trails[key].clear()   # different sighting - start this trail fresh
                    trails[key].append((x, y))
                    last_point[key] = (frame_num, x, y)

            for (team_num, pid), history in trails.items():
                if len(history) < 2:
                    continue
                color = attacking_color if team_num == attacking_num else defending_color
                variant = _variant_color(color, pid)
                pts = [pt(x, y) for x, y in history]
                n = len(pts)
                for i in range(1, n):
                    t = i / n
                    alpha = 0.20 + 0.80 * t   # oldest 20% opacity -> newest 100%, same fade as render_output6
                    seg_color = tuple(int(c * alpha) for c in variant)
                    cv2.line(img, pts[i - 1], pts[i], seg_color, 2, cv2.LINE_AA)

            cv2.putText(img, f"{attacking_label} (attacking)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, attacking_color, 2)
            cv2.putText(img, f"{defending_label} (defending)", (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, defending_color, 2)
            writer.write(img)
    finally:
        writer.release()
