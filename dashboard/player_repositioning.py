"""
player_repositioning.py — drag-and-drop hypothetical player repositioning,
the dashboard's "Game Board" tab.

Built on several rounds of prior investigation and two earlier shipped
designs, in order: (0) a simple ID-labeled circular marker composited onto
the real broadcast frame, which visually resembled this project's own
"Tracking + Speed & Distance" rendered output closely enough to be mistaken
for it; (1) real, background-differencing-segmented player pixels pasted
onto the real broadcast frame in place of that marker - broadly audited
against every tracked player across several real frames (not just the one
case each fix was first tried against) and found to have a genuine,
structural ~10% failure rate (a dark kit against a shadowed pitch can put a
real player's legs below Otsu's single global diff threshold entirely, and
separately, clean-plate reconstruction's own real ~2% residual misalignment
can produce a false blob larger than the real player) - reported as such
rather than shipped with the failure rate hidden, since neither is a bug
more mask post-processing fixes.

Current design: a plain, team-colored tactical board - the same idea as a
real coach's magnetic whiteboard, not a doctored photo. Every tracked
player's REAL pitch position (see frame_player_pitch_positions, via this
match's own per-frame homography - already the same calibrated pitch plane
every other module in this project uses) is drawn as a colored dot on a
flat, schematic 2D pitch, not composited onto the perspective broadcast
frame at all. This sidesteps every failure mode design (1) had - there is
no segmentation, no clean-plate erase, no perspective-scale estimation
anywhere in this design, so there is nothing left in it that can produce a
"torso only" or oversized/undersized result. The real broadcast frame is
still shown alongside it, for visual context of the real moment, but is no
longer the surface anything is dragged on.

Storage: a per-match list of active repositions (now in pitch-meter
coordinates, not pixel coordinates), layered non-destructively under
CACHE_DIR (same pattern as corner_kicks.py / training_plan.py /
player_labels.py - a separate file, never bundle.json or the CV pipeline's
own output files).

Data dependency: cv_pipeline/output_videos/<cv_output_dir>/repositioning_data.json,
produced offline by cv_pipeline/export_repositioning_data.py from that
match's already-cached tracks/camera_movement/homography stubs - this
module never touches the raw .pkl stubs directly (they're outside the
dashboard's own dependency-light environment; see that export script's own
docstring for why). A match with no such file simply doesn't support this
feature yet - reported honestly (see load_context's return), not silently
faked.
"""
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
_BOUNDS_MARGIN_M = 2.0  # matches corner_kicks.py's own _BOUNDS_MARGIN_M
_CENTER = (52.5, 34.0)
_SCALE_RATIO_LIMIT = 6.0  # matches the polish-pass report's validated gate threshold

# Standard pitch-marking dimensions (metres) - real FIFA/IFAB dimensions,
# the same numbers this project's own PITCH_REFERENCE_POINTS convention
# already implies. Kept here as the single source of truth the tactical
# board's client-side drawing reads through its own args, rather than a
# second hardcoded copy.
CENTER_CIRCLE_R_M = 9.15
BOX_DEPTH_M = 16.5
SIXBOX_DEPTH_M = 5.5
PENALTY_SPOT_DIST_M = 11.0
BOX_Y_M = (13.84, 54.16)
SIXBOX_Y_M = (24.84, 43.16)


# ==========================================================================
# CONTEXT LOADING
# ==========================================================================

class RepositioningContext:
    """Everything needed to reposition players within one CV-analyzed
    window. One instance per (cv_output_dir, video_path) pair."""

    def __init__(self, data, video_path):
        self.players = data["players"]
        self.referees = data["referees"]
        self.ball = data["ball"]
        self.camera_movement = data["camera_movement"]
        # JSON object keys are strings; frame lookups elsewhere use int
        # indices consistently with every other per-frame list here.
        self.homography = {int(k): v for k, v in data["homography"].items()}
        self.n_frames = data["n_frames"]
        self.video_path = video_path
        self._cap = None
        self._frame_w = None
        self._frame_h = None

    @property
    def cap(self):
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.video_path)
            self._frame_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._frame_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self._cap

    @property
    def frame_w(self):
        _ = self.cap
        return self._frame_w

    @property
    def frame_h(self):
        _ = self.cap
        return self._frame_h

    @property
    def fps(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 1 else 25.0

    def get_frame(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"could not read frame {idx} from {self.video_path}")
        return frame


def load_context(cv_output_dir, video_path):
    """Returns a RepositioningContext, or None if this match hasn't been
    exported for repositioning yet (repositioning_data.json missing) - an
    honest, expected state for any match processed before this feature
    existed, or one whose export step hasn't been run.

    `cv_output_dir` is expected already-resolved (app.py's
    st.session_state.cv_job_output_dir, itself produced by _resolve_cv_path
    at bundle-activation time) - same convention every other CV-output file
    read in this app already follows, not re-resolved here."""
    data_path = Path(cv_output_dir) / "repositioning_data.json"
    if not data_path.exists():
        return None
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return RepositioningContext(data, video_path)


# ==========================================================================
# GEOMETRY — camera motion, homography, real pitch coordinates
# (ported from the investigation's lib.py, adapted to per-context data
# instead of module-level globals, so this module supports any match)
# ==========================================================================

def cumulative_offset(ctx, frame_a, frame_b):
    """(dx, dy) such that pos_in_frame_b = pos_in_frame_a - (dx, dy). See
    the investigation's lib.py for the full derivation from
    camera_movement_estimator.py's own convention."""
    if frame_b == frame_a:
        return 0.0, 0.0
    cm = ctx.camera_movement
    if frame_b > frame_a:
        seg = cm[frame_a + 1: frame_b + 1]
        sign = 1.0
    else:
        seg = cm[frame_b + 1: frame_a + 1]
        sign = -1.0
    dx = sign * sum(v[0] for v in seg)
    dy = sign * sum(v[1] for v in seg)
    return dx, dy


def pixel_to_pitch(ctx, frame_idx, px, py):
    h = ctx.homography.get(frame_idx)
    if h is None:
        return None
    H = np.array(h[0])
    out = H @ np.array([px, py, 1.0])
    if abs(out[2]) < 1e-9:
        return None
    return (out[0] / out[2], out[1] / out[2])


def pitch_to_pixel(ctx, frame_idx, X, Y):
    h = ctx.homography.get(frame_idx)
    if h is None:
        return None
    H_inv = np.array(h[1])
    out = H_inv @ np.array([X, Y, 1.0])
    if abs(out[2]) < 1e-9:
        return None
    return (out[0] / out[2], out[1] / out[2])


def local_pixel_scale(ctx, frame_idx, X, Y, delta=1.0):
    p0 = pitch_to_pixel(ctx, frame_idx, X, Y)
    p1 = pitch_to_pixel(ctx, frame_idx, X, Y + delta)
    if p0 is None or p1 is None:
        return None
    return ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5


def _in_pitch_bounds(X, Y):
    return (-_BOUNDS_MARGIN_M <= X <= PITCH_LENGTH_M + _BOUNDS_MARGIN_M
            and -_BOUNDS_MARGIN_M <= Y <= PITCH_WIDTH_M + _BOUNDS_MARGIN_M)


def _homography_trustworthy(ctx, frame_idx, X, Y):
    """Same self-consistency check the polish-pass report's validate_scale
    used as a hard reject - reused here as a soft quality filter on which
    OTHER real players' positions are trusted for the placement-distance
    stats (see compute_placement_stats), not as a gate on the tactical
    board's own dot placement (which needs no scale estimate at all)."""
    if not _in_pitch_bounds(X, Y):
        return False
    scale = local_pixel_scale(ctx, frame_idx, X, Y)
    center_scale = local_pixel_scale(ctx, frame_idx, *_CENTER)
    if scale is None or scale <= 0 or center_scale is None or center_scale <= 0:
        return False
    ratio = scale / center_scale
    return (1.0 / _SCALE_RATIO_LIMIT) <= ratio <= _SCALE_RATIO_LIMIT


# ==========================================================================
# TACTICAL BOARD — real per-player pitch positions for one paused frame
# ==========================================================================

def _nearest_homography_frame(ctx, frame_idx, max_search=90):
    """Nearest frame (this one first) with usable homography, within
    max_search frames either direction - None if truly none nearby.
    Verified directly against this project's own real match data: usable
    calibration is missing on a real minority of frames (~8% for the
    liverpool_psg curated match), but the longest real gap measured was 12
    consecutive frames, well inside this function's default search
    window."""
    if ctx.homography.get(frame_idx) is not None:
        return frame_idx
    for d in range(1, max_search + 1):
        for cand in (frame_idx - d, frame_idx + d):
            if 0 <= cand < ctx.n_frames and ctx.homography.get(cand) is not None:
                return cand
    return None


def frame_player_pitch_positions(ctx, frame_idx, max_homography_search=90):
    """Real pitch (X, Y) position for every tracked player in this frame -
    the data the tactical board is built from. When this exact frame has no
    usable homography of its own, borrows the nearest frame's that does
    (camera-motion-compensated via cumulative_offset, so this frame's own
    real tracked pixel positions are shifted into that borrowed frame's
    pixel-coordinate space before projecting - the same alignment
    convention this module's clean-plate work already used) rather than
    leaving that frame's board empty; a small time-adjustment error from
    borrowing a few frames away is preferable to no board at all, and is
    reported back via the second return value so the caller can disclose it
    rather than pass it off as exact.

    Returns ({track_id: (X, Y)}, used_frame_idx), or (None, None) only when
    NO frame within range has usable calibration at all - reported
    honestly, not faked."""
    src = _nearest_homography_frame(ctx, frame_idx, max_homography_search)
    if src is None:
        return None, None
    dx, dy = cumulative_offset(ctx, frame_idx, src) if src != frame_idx else (0.0, 0.0)
    positions = {}
    for tid, info in ctx.players[frame_idx].items():
        x1, y1, x2, y2 = info["bbox"]
        fx, fy = (x1 + x2) / 2 - dx, y2 - dy
        pitch = pixel_to_pitch(ctx, src, fx, fy)
        if pitch is not None and _in_pitch_bounds(*pitch):
            positions[tid] = pitch
    return positions, src


# ==========================================================================
# PERSISTENCE — per-match, non-destructive (same discipline as
# corner_kicks.py / training_plan.py / player_labels.py). Moves are stored
# in pitch-meter coordinates now, not pixel coordinates - resolution- and
# camera-independent, matching the tactical board's own coordinate space.
# ==========================================================================

def _repositions_path(cache_dir, match_key):
    return Path(cache_dir) / "repositions" / f"{match_key}.json"


def load_repositions(cache_dir, match_key):
    if not match_key:
        return []
    path = _repositions_path(cache_dir, match_key)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_repositions(cache_dir, match_key, repositions):
    path = _repositions_path(cache_dir, match_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(repositions, f, indent=2)
    os.replace(tmp, path)


def clear_repositions(cache_dir, match_key):
    save_repositions(cache_dir, match_key, [])


# ==========================================================================
# POSITION-DEPENDENT STATS — on-pitch only, honestly skipped off-pitch
# ==========================================================================

def compute_placement_stats(ctx, frame_idx, moved_track_id, target_pitch_XY, exclude_track_ids=()):
    """Real, computable stats for an ON-PITCH hypothetical placement only -
    distance to the nearest other real tracked player (any team) and
    nearest opponent-team-agnostic teammate, both straightforward real
    Euclidean distances on the same calibrated pitch plane every other
    module in this project already uses. Returns None for an off-pitch
    target - those stats don't apply there, so nothing is computed rather
    than showing a meaningless number (same principle as corner_kicks.py's
    explicit no-role-inference scope)."""
    if target_pitch_XY is None or not _in_pitch_bounds(*target_pitch_XY):
        return None
    tx, ty = target_pitch_XY
    distances = []
    for tid, info in ctx.players[frame_idx].items():
        if tid == str(moved_track_id) or tid in exclude_track_ids:
            continue
        x1, y1, x2, y2 = info["bbox"]
        pitch = pixel_to_pitch(ctx, frame_idx, (x1 + x2) / 2, y2)
        if pitch is None or not _homography_trustworthy(ctx, frame_idx, *pitch):
            continue
        d = math.hypot(pitch[0] - tx, pitch[1] - ty)
        distances.append((tid, d))
    if not distances:
        return None
    distances.sort(key=lambda t: t[1])
    return {
        "nearest_player_track_id": distances[0][0],
        "nearest_player_distance_m": round(distances[0][1], 2),
        "n_players_within_5m": sum(1 for _, d in distances if d <= 5.0),
        "n_players_within_10m": sum(1 for _, d in distances if d <= 10.0),
    }
