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
import json
import math
import os
from pathlib import Path

import re

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


def upsert_mark(cache_dir, match_key, mark_id, timestamp_label, attacking_team, defending_team, cv_output_dir=None,
                 reference_team1_is_team_a=True):
    """attacking_team/defending_team are 'team_a'/'team_b' tokens (the same
    stable convention cv_team_mapping already uses), never a raw color or a
    literal team name - a rename must never orphan an existing mark.
    cv_output_dir is the folder name under cv_pipeline/output_videos/ once
    that window has actually been processed - None until then.

    reference_team1_is_team_a: whether THIS match's own main-window
    team_resolution.team_colors_bgr["team1"] is real team_a - see
    resolve_corner_team_mapping's docstring for why this can't be assumed
    (True only holds by coincidence, not by rule) and must be resolved
    once, per match, and carried on the mark rather than re-guessed from
    color at render time."""
    marks = [m for m in load_marks(cache_dir, match_key) if m["id"] != mark_id]
    marks.append({
        "id": mark_id,
        "timestamp_label": timestamp_label,
        "attacking_team": attacking_team,
        "defending_team": defending_team,
        "cv_output_dir": cv_output_dir,
        "reference_team1_is_team_a": reference_team1_is_team_a,
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


def load_calibration_status(cv_pipeline_dir, cv_output_dir):
    """Whether this corner's own calibration has been verified reliable, per
    the real-coordinate back-projection spot-check (see KNOWN_ISSUES.md) -
    NOT something derivable from the pipeline's own reported confidence
    number, which is exactly what that investigation found doesn't
    reliably predict correct homography (a wrong-but-internally-consistent
    RANSAC fit can score 0.92-1.00 while being severely wrong). This is a
    human-verified determination, written once per processed segment
    alongside player_positions.json - not a formula computed here.

    Defaults to reliable=True when no status file exists yet (e.g. a
    freshly-marked corner nobody has checked) - the frame-count check in
    compute_corner_metrics's caller is a real, independent signal on its
    own and still applies regardless of this flag; this flag exists to
    catch the specific failure mode (bad calibration that still produces
    plenty of in-bounds-looking positions) that a frame-count alone cannot
    detect, not to replace that check."""
    if not cv_output_dir:
        return {"reliable": True, "note": None}
    path = Path(cv_pipeline_dir) / "output_videos" / cv_output_dir / "calibration_status.json"
    if not path.exists():
        return {"reliable": True, "note": None}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {"reliable": bool(data.get("reliable", True)), "note": data.get("note")}
    except (json.JSONDecodeError, OSError):
        return {"reliable": True, "note": None}


def resolve_corner_team_mapping(positions_data, reference_team_colors_bgr, reference_team1_is_team_a=True):
    """Which of THIS corner's own team-1/team-2 jersey-color clusters is
    team_a vs team_b. Each corner is its own independent CV run with its
    own from-scratch K-means color clustering, so team1/team2 are NOT
    guaranteed to line up with the match's overall assignment - confirmed
    directly: of the 3 real corners processed for this feature, one had
    team1/team2 inverted relative to the other two and to the main match
    (cost ratio of ~20x between the two possible mappings - not a close
    call, genuinely swapped). Resolved by nearest-color match against the
    match's own already-confirmed reference colors (its main analyzed
    window's team_resolution.team_colors_bgr), never assumed to be
    index-stable across separate runs.

    reference_team1_is_team_a: whether the REFERENCE match's own "team1"
    (not this corner's team1) is real team_a. Defaults to True (the
    original assumption) - correct for liverpool_psg by coincidence, since
    its own cv_team_mapping happens to be {"1":"team_a","2":"team_b"} - but
    genuinely FALSE for other matches (found on barca_madrid_pt1: team1's
    reference color is dark/Maroon, i.e. team_b, not team_a's White -
    confirmed via jersey brightness and on-screen team labels, not
    assumed). This must never be re-derived from color alone here (that's
    exactly the ambiguous judgment call the app's own one-time swatch
    confirmation exists to make for the main dashboard) - it's a per-mark,
    already-resolved fact the caller passes in."""
    def dist(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    c1 = positions_data["team_colors_bgr"]["1"]
    c2 = positions_data["team_colors_bgr"]["2"]
    if reference_team1_is_team_a:
        ref_a, ref_b = reference_team_colors_bgr["team1"], reference_team_colors_bgr["team2"]
    else:
        ref_a, ref_b = reference_team_colors_bgr["team2"], reference_team_colors_bgr["team1"]
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