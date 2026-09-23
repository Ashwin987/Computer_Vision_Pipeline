"""
player_repositioning.py — drag-and-drop hypothetical player repositioning,
the dashboard's "Game Board" tab.

Built on several rounds of prior investigation and design changes, in
order: (0) a simple ID-labeled circular marker composited onto the real
broadcast frame, which visually resembled this project's own "Tracking +
Speed & Distance" rendered output closely enough to be mistaken for it;
(1) real, background-differencing-segmented player pixels pasted onto the
real broadcast frame in place of that marker - broadly audited and found
to have a genuine, structural ~10% failure rate from background-
differencing segmentation itself (dark kit vs. shadowed pitch; clean-plate
residual misalignment), reported honestly rather than shipped hidden;
(2) a plain team-colored marker REPROJECTED through this match's per-frame
homography onto a flat, schematic 2D pitch diagram - eliminated
segmentation entirely, but a real, direct root-cause check (round-trip
pixel<->pitch self-consistency: ~0.02px error, ruling out a coding bug;
same-frame local-scale self-consistency ratio: most real players in a
single frame measured 5-11x vs. this module's own already-documented
_SCALE_RATIO_LIMIT=6.0 threshold) confirmed the visible dot/player
misalignment users saw was genuine homography/calibration inaccuracy, not
a bug in this module's use of it - the same "smoothly wrong" limitation
this project's own calibration work has documented since the original
investigation, now visibly exposed by putting a diagram next to a photo of
the same moment.

Current design: no reprojection at all for placement. A player's dot is
drawn at their REAL TRACKED PIXEL position for that exact frame (the same
bbox the tracker already produced - no homography, no world-coordinate
round trip), on top of that same real frame with every player erased via
the same clean-plate reconstruction technique already proven for the
original drag feature (find_clean_patch/composite_patch: search nearby,
camera-motion-aligned frames for an unoccluded patch, blend when a pitch
line crosses it). Dot position and cleaned background come from the same
frame in the same pixel coordinate system, so they are aligned by
construction - there is no calibration step left in this path that could
misalign them. Homography is still used, but only as a secondary, openly-
approximate real-world-distance ESTIMATE for the optional nearest-player
stat (compute_placement_stats) - never for where anything is actually
drawn.

Storage: a per-match list of active repositions, in pixel coordinates
(this frame's own, not world meters), layered non-destructively under
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
# used only by line_crosses_patch's blending-quality check below (whether
# to erase-blend one player's patch, not for any dot placement).
CENTER_CIRCLE_R_M = 9.15
BOX_DEPTH_M = 16.5
SIXBOX_DEPTH_M = 5.5
PENALTY_SPOT_DIST_M = 11.0
BOX_Y_M = (13.84, 54.16)
SIXBOX_Y_M = (24.84, 43.16)
_PENALTY_SPOT_L = (PENALTY_SPOT_DIST_M, _CENTER[1])
_PENALTY_SPOT_R = (PITCH_LENGTH_M - PENALTY_SPOT_DIST_M, _CENTER[1])


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
# GEOMETRY — camera motion, homography (used only for the optional
# nearest-player distance stat now, never for dot placement)
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


def shift_rect(rect, dx, dy):
    x1, y1, x2, y2 = rect
    return (x1 - dx, y1 - dy, x2 - dx, y2 - dy)


def rect_overlap_area(r1, r2):
    ax1, ay1, ax2, ay2 = r1
    bx1, by1, bx2, by2 = r2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


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
    stat (see compute_placement_stats). Direct, current-data confirmation
    of why this matters: for frame 340 of the liverpool_psg curated match,
    10 real on-pitch players' own local homography scale ranged from 1.9x
    to 11.2x the frame-center scale - most fail even this generous 6x
    self-consistency check. Never used to gate dot placement itself, which
    no longer depends on the homography at all."""
    if not _in_pitch_bounds(X, Y):
        return False
    scale = local_pixel_scale(ctx, frame_idx, X, Y)
    center_scale = local_pixel_scale(ctx, frame_idx, *_CENTER)
    if scale is None or scale <= 0 or center_scale is None or center_scale <= 0:
        return False
    ratio = scale / center_scale
    return (1.0 / _SCALE_RATIO_LIMIT) <= ratio <= _SCALE_RATIO_LIMIT


# ==========================================================================
# OCCLUSION + CLEAN-PLATE (ported, per-context) — identical technique the
# original player-repositioning drag feature already proved out; reused
# here to erase EVERY real player from a frame, not just one being dragged.
# ==========================================================================

def is_clean(ctx, frame_idx, rect, overlap_frac_thresh=0.02):
    rx1, ry1, rx2, ry2 = rect
    rect_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    if rect_area <= 0:
        return False
    for tracks in (ctx.players[frame_idx], ctx.referees[frame_idx], ctx.ball[frame_idx]):
        for tid, info in tracks.items():
            if rect_overlap_area(rect, info["bbox"]) / rect_area > overlap_frac_thresh:
                return False
    return True


def _pitch_marking_points(step=0.25):
    pts = []

    def seg(x0, y0, x1, y1):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step))
        for t in np.linspace(0, 1, n):
            pts.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))

    L, W = PITCH_LENGTH_M, PITCH_WIDTH_M
    seg(0, 0, L, 0); seg(0, W, L, W); seg(0, 0, 0, W); seg(L, 0, L, W)
    seg(L / 2, 0, L / 2, W)
    by0, by1 = BOX_Y_M
    seg(0, by0, BOX_DEPTH_M, by0); seg(0, by1, BOX_DEPTH_M, by1); seg(BOX_DEPTH_M, by0, BOX_DEPTH_M, by1)
    seg(L, by0, L - BOX_DEPTH_M, by0); seg(L, by1, L - BOX_DEPTH_M, by1)
    seg(L - BOX_DEPTH_M, by0, L - BOX_DEPTH_M, by1)
    sy0, sy1 = SIXBOX_Y_M
    seg(0, sy0, SIXBOX_DEPTH_M, sy0); seg(0, sy1, SIXBOX_DEPTH_M, sy1)
    seg(SIXBOX_DEPTH_M, sy0, SIXBOX_DEPTH_M, sy1)
    seg(L, sy0, L - SIXBOX_DEPTH_M, sy0); seg(L, sy1, L - SIXBOX_DEPTH_M, sy1)
    seg(L - SIXBOX_DEPTH_M, sy0, L - SIXBOX_DEPTH_M, sy1)
    n = int(2 * math.pi * CENTER_CIRCLE_R_M / step)
    for a in np.linspace(0, 2 * math.pi, n):
        pts.append((_CENTER[0] + CENTER_CIRCLE_R_M * math.cos(a), _CENTER[1] + CENTER_CIRCLE_R_M * math.sin(a)))
    for a in np.linspace(0, 2 * math.pi, n):
        x = _PENALTY_SPOT_L[0] + CENTER_CIRCLE_R_M * math.cos(a)
        y = _PENALTY_SPOT_L[1] + CENTER_CIRCLE_R_M * math.sin(a)
        if x > BOX_DEPTH_M:
            pts.append((x, y))
        xr = _PENALTY_SPOT_R[0] + CENTER_CIRCLE_R_M * math.cos(a)
        yr = _PENALTY_SPOT_R[1] + CENTER_CIRCLE_R_M * math.sin(a)
        if xr < L - BOX_DEPTH_M:
            pts.append((xr, yr))
    return pts


_PITCH_MARKING_ARRAY = np.array([(X, Y, 1.0) for X, Y in _pitch_marking_points()], dtype=np.float64)


def line_crosses_patch(ctx, frame_idx, rect, margin_px=60):
    """See the polish-pass investigation for why margin_px=60, not a tight
    few pixels: tested directly against real line-crossing cases, this
    pipeline's own homography position error ran 50-70px at the halfway
    line and ~25m of real pitch distance at the center circle - a tight
    margin missed both. (Only decides whether a patch needs multi-frame
    blending quality - unrelated to, and unaffected by, that same real
    calibration inaccuracy no longer being used for dot placement.)"""
    h = ctx.homography.get(frame_idx)
    if h is None:
        return False
    H_inv = np.array(h[1])
    proj = (H_inv @ _PITCH_MARKING_ARRAY.T).T
    w = proj[:, 2]
    valid = np.abs(w) > 1e-9
    if not valid.any():
        return False
    px = proj[valid, 0] / w[valid]
    py = proj[valid, 1] / w[valid]
    x1, y1, x2, y2 = rect
    x1, y1, x2, y2 = x1 - margin_px, y1 - margin_px, x2 + margin_px, y2 + margin_px
    return bool(np.any((px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)))


def extract_patch(frame_img, rect):
    x1, y1, x2, y2 = [int(round(v)) for v in rect]
    return frame_img[y1:y2, x1:x2].copy()


def find_clean_patch(ctx, target_frame, rect, search_radius=100, max_blend_frames=3):
    """Identical algorithm to the investigation/polish-pass lib.py - see
    that module for the full docstring. Works unchanged for stands/crowd
    background: is_clean only checks tracked player/referee/ball bboxes,
    which the tracker only ever produces on the pitch, so any rect
    entirely in the stands is trivially "clean" the first time it's
    checked (verified directly - see the report's stands test)."""
    needs_blend = line_crosses_patch(ctx, target_frame, rect)
    limit = max_blend_frames if needs_blend else 1
    found = []
    for offset in range(1, search_radius + 1):
        for cand in (target_frame - offset, target_frame + offset):
            if cand < 0 or cand >= ctx.n_frames:
                continue
            dx, dy = cumulative_offset(ctx, target_frame, cand)
            aligned = shift_rect(rect, dx, dy)
            ax1, ay1, ax2, ay2 = aligned
            if ax1 < 0 or ay1 < 0 or ax2 > ctx.frame_w or ay2 > ctx.frame_h:
                continue
            if is_clean(ctx, cand, aligned):
                found.append((cand, offset, aligned))
                if len(found) >= limit:
                    break
        if len(found) >= limit:
            break
    if not found:
        return None

    rx1, ry1, rx2, ry2 = [int(round(v)) for v in rect]
    target_w, target_h = rx2 - rx1, ry2 - ry1
    patches = []
    for cand, offset, aligned in found:
        img = ctx.get_frame(cand)
        patch = extract_patch(img, aligned)
        if patch.shape[1] != target_w or patch.shape[0] != target_h:
            patch = cv2.resize(patch, (target_w, target_h))
        patches.append(patch.astype(np.float32))

    if len(patches) == 1:
        blended = patches[0]
    else:
        weights = np.array([1.0 / (offset + 1) for _, offset, _ in found], dtype=np.float32)
        weights = weights / weights.sum()
        blended = np.zeros_like(patches[0])
        for wgt, p in zip(weights, patches):
            blended += wgt * p

    return {
        "patch": np.clip(blended, 0, 255).astype(np.uint8),
        "sources": [(c, o) for c, o, _ in found],
        "blended": len(patches) > 1,
    }


def composite_patch(dest_img, patch_img, rect, feather_px=10):
    """Identical to the polish-pass lib.py - see that module for the full
    docstring (Poisson blend via cv2.seamlessClone, feathered-alpha
    fallback for a patch flush against the frame edge).

    Clamps `rect` to dest_img's real bounds before pasting - a real rect
    can validly extend past the frame edge (a player standing right at the
    boundary plus erase margin; confirmed directly - tid 245 in a real
    frame had rect x2=1926 against a 1920px-wide frame). Unclamped, numpy
    slicing there silently returns a NARROWER region than patch_img's own
    nominal size instead of erroring, desyncing the two and crashing the
    blend below. This was a real, latent bug in this exact code from the
    original investigation - it just never surfaced before because only
    one actively-dragged player was ever erased per call; erasing every
    player in a frame for the game board makes an edge-adjacent player
    common enough to hit it directly."""
    rx1, ry1, rx2, ry2 = [int(round(v)) for v in rect]
    dh, dw = dest_img.shape[:2]
    crx1, cry1 = max(0, rx1), max(0, ry1)
    crx2, cry2 = min(dw, rx2), min(dh, ry2)
    w, h = crx2 - crx1, cry2 - cry1
    if w < 3 or h < 3:
        return dest_img.copy(), "skip-degenerate"
    px1, py1 = crx1 - rx1, cry1 - ry1
    patch_img = patch_img[py1:py1 + h, px1:px1 + w]
    rx1, ry1, rx2, ry2 = crx1, cry1, crx2, cry2

    if 0 < rx1 and rx2 < dest_img.shape[1] and 0 < ry1 and ry2 < dest_img.shape[0]:
        try:
            mask = np.full((h, w), 255, dtype=np.uint8)
            center = (rx1 + w // 2, ry1 + h // 2)
            cloned = cv2.seamlessClone(patch_img, dest_img, mask, center, cv2.NORMAL_CLONE)
            return cloned, "seamless"
        except cv2.error:
            pass

    inner = np.zeros((h, w), dtype=np.uint8)
    if h > 2 and w > 2:
        inner[1:-1, 1:-1] = 255
        dist = cv2.distanceTransform(inner, cv2.DIST_L2, 5)
        alpha = np.clip(dist / max(1, feather_px), 0, 1)[..., None]
    else:
        alpha = np.ones((h, w, 1), dtype=np.float32)
    result = dest_img.copy()
    dest_region = result[ry1:ry2, rx1:rx2].astype(np.float32)
    blended = patch_img.astype(np.float32) * alpha + dest_region * (1 - alpha)
    result[ry1:ry2, rx1:rx2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result, "feather"


# ==========================================================================
# GAME BOARD — real pixel-space player positions on a cleaned real frame
# ==========================================================================

def frame_player_pixel_positions(ctx, frame_idx):
    """Real (x, y) foot-point pixel position for every tracked player in
    this frame, straight from the tracker's own bbox - no homography, no
    world-coordinate round trip. This is the position the game board's
    dots are drawn at, so a dot and the real player it represents are
    aligned by construction: both come from the exact same tracked bbox in
    the exact same frame's pixel space."""
    positions = {}
    for tid, info in ctx.players[frame_idx].items():
        x1, y1, x2, y2 = info["bbox"]
        if (x2 - x1) < 10 or (y2 - y1) < 20:
            continue
        positions[tid] = ((x1 + x2) / 2.0, y2)
    return positions


def clean_frame_no_players(ctx, frame_idx, margin=6):
    """Erases every real tracked player AND referee from this frame via
    the exact same clean-plate reconstruction already proven for the
    original drag feature (find_clean_patch + composite_patch) - leaving
    just the pitch, crowd, and background, so a dot drawn at a player's
    real pixel position afterward sits on a background that's genuinely
    clean there, not a duplicate of the player it represents.

    A rare missing-clean-patch case (measured under 2% in the original
    investigation) is skipped for just that one person rather than failing
    the whole frame - their real pixels stay visible underneath, and
    everything else about the board still works; reported back via the
    second return value rather than hidden.

    Returns (cleaned_img, n_failed)."""
    composite = ctx.get_frame(frame_idx)
    n_failed = 0
    all_tracks = list(ctx.players[frame_idx].items()) + list(ctx.referees[frame_idx].items())
    for tid, info in all_tracks:
        x1, y1, x2, y2 = info["bbox"]
        rect = (x1 - margin, y1 - margin, x2 + margin, y2 + margin)
        found = find_clean_patch(ctx, frame_idx, rect, search_radius=150, max_blend_frames=3)
        if found is None:
            n_failed += 1
            continue
        composite, _ = composite_patch(composite, found["patch"], rect)
    return composite, n_failed


# ==========================================================================
# PERSISTENCE — per-match, non-destructive (same discipline as
# corner_kicks.py / training_plan.py / player_labels.py). Moves are stored
# in this frame's own pixel coordinates, matching the board's own
# coordinate space (no world-meter conversion for placement).
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
# POSITION-DEPENDENT STATS — on-pitch only, honestly skipped off-pitch.
# Still uses the homography, deliberately - this is an openly-approximate
# real-world-distance ESTIMATE for a caption, not where anything is drawn.
# ==========================================================================

def compute_placement_stats(ctx, frame_idx, moved_track_id, target_pitch_XY, exclude_track_ids=()):
    """Real, computable stats for an ON-PITCH hypothetical placement only -
    distance to the nearest other real tracked player (any team), a
    straightforward Euclidean distance on the same calibrated pitch plane
    every other module in this project already uses. Returns None for an
    off-pitch (or un-resolvable) target - those stats don't apply there,
    so nothing is computed rather than showing a meaningless number (same
    principle as corner_kicks.py's explicit no-role-inference scope)."""
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
