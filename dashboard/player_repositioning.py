"""
player_repositioning.py — drag-and-drop hypothetical player repositioning,
CV Deep Analysis tab.

Built on three rounds of prior investigation against this exact match's real
data (clean-plate reconstruction, background-differencing segmentation, and
seamless-clone edge blending all verified working; a naive scale-gate first
built then honestly discarded when tested against real players; a
temporal-consistency check investigated and also discarded when it failed to
separate known-good from known-bad placements). This module is the first
time any of that becomes a real, shipped feature rather than a scratch
investigation script - see PROPOSAL_advanced_tactical_signals.md and this
project's own session history for the full record of what was tried and
rejected before landing on the design below.

Core design decision this module makes differently from every earlier round:
no position is ever refused. The investigation's scale-gate (reject a
placement if implied scale looks physically absurd) is replaced here by a
bounded, continuous scale function (see bounded_scale_for_target) - the goal
changed from "know when NOT to trust the homography" to "never need to fully
trust it in the first place": on-pitch positions where the homography's
local scale passes the same self-consistency check the gate used to enforce
as a hard reject are used directly; every other position (including
anywhere off the pitch - stands, crowd, behind the goal) falls back to a
log-linear perspective trend fitted from this frame's own REAL tracked
players, clamped to a plausible pixel-size range also derived from those
same real players. A placement can therefore always be composited; the
worst case is a size that's merely approximate, never one that's absurd.

Real player, no synthetic overlay: this module operates on, and only ever
displays, the raw broadcast frame (the same clean, unannotated source the
clean-plate reconstruction work has used as its input from round one) - the
UI it feeds draws real segmented player pixels as the draggable element,
never an ID-labeled marker icon standing in for one, and the moved result
carries no badge, label, or other baked-in text distinguishing it from a
real, untouched player. A first version of this feature's UI used simple
circular ID markers instead, which - even though the underlying frame was
already the correct raw one - visually resembled this project's own
"Tracking + Speed & Distance" rendered output (which does bake in ID-
labeled circles) closely enough to be mistaken for it. Corrected directly.

Storage: a per-match list of active repositions, layered non-destructively
under CACHE_DIR (same pattern as corner_kicks.py / training_plan.py /
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
        self._trend_cache = {}

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
# GEOMETRY — camera motion, homography, pitch markings
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


# HSV grass-green range - same spirit as this project's own existing
# jersey-color sampling (technical report Section 2.3: team classification
# filters OUT grass-green hues to isolate jersey color); used here in the
# opposite direction, to directly DETECT grass. Deliberately not homography-
# based: confirmed directly, dragging to a pixel visually in the crowd/
# advertising-board area at the top of a real frame, the homography's own
# [0,105]x[0,68] bounds check classified it as "on the pitch" anyway (ratio-
# consistency passed too) - the same "smoothly wrong" failure pattern this
# project's calibration work keeps finding, here affecting the on/off-pitch
# classification that gates whether a stat is even meaningful to show. A
# direct, real pixel-color read of what's actually there doesn't depend on
# trusting an extrapolated homography at all.
_GRASS_HUE_RANGE = (30, 95)   # OpenCV H in [0,179]
_GRASS_MIN_SAT = 40
_GRASS_MIN_VAL = 30


def _sample_is_grass(frame_img, x, y, radius=8, min_frac=0.5):
    h, w = frame_img.shape[:2]
    x1, y1 = max(0, int(x - radius)), max(0, int(y - radius))
    x2, y2 = min(w, int(x + radius)), min(h, int(y + radius))
    if x2 <= x1 or y2 <= y1:
        return False
    patch = frame_img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass_mask = ((h_ch >= _GRASS_HUE_RANGE[0]) & (h_ch <= _GRASS_HUE_RANGE[1])
                  & (s_ch >= _GRASS_MIN_SAT) & (v_ch >= _GRASS_MIN_VAL))
    return float(grass_mask.mean()) >= min_frac


def _is_really_on_pitch(ctx, frame_idx, target_pixel_xy, target_pitch_XY):
    """Both signals required, deliberately - the homography-bounds check
    alone was confirmed to misclassify a real crowd-area pixel as on-pitch
    (see module docstring), and grass color alone could misfire on a
    grass-colored ad board or pitch-adjacent warm-up area, so requiring
    agreement is more conservative than either check on its own."""
    if target_pitch_XY is None or not _in_pitch_bounds(*target_pitch_XY):
        return False
    frame_img = ctx.get_frame(frame_idx)
    return _sample_is_grass(frame_img, *target_pixel_xy)


def _homography_trustworthy(ctx, frame_idx, X, Y):
    """Same self-consistency check the polish-pass report's validate_scale
    used as a hard reject - reused here as a soft signal (trust raw
    homography scale, or fall back to the trend function) instead."""
    if not _in_pitch_bounds(X, Y):
        return False
    scale = local_pixel_scale(ctx, frame_idx, X, Y)
    center_scale = local_pixel_scale(ctx, frame_idx, *_CENTER)
    if scale is None or scale <= 0 or center_scale is None or center_scale <= 0:
        return False
    ratio = scale / center_scale
    return (1.0 / _SCALE_RATIO_LIMIT) <= ratio <= _SCALE_RATIO_LIMIT


# ==========================================================================
# OCCLUSION + CLEAN-PLATE (ported, per-context)
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


_CENTER_CIRCLE_R = 9.15
_PENALTY_SPOT_L = (11.0, 34.0)
_PENALTY_SPOT_R = (94.0, 34.0)
_BOX_Y = (13.84, 54.16)
_SIXBOX_Y = (24.84, 43.16)


def _pitch_marking_points(step=0.25):
    pts = []

    def seg(x0, y0, x1, y1):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step))
        for t in np.linspace(0, 1, n):
            pts.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))

    seg(0, 0, 105, 0); seg(0, 68, 105, 68); seg(0, 0, 0, 68); seg(105, 0, 105, 68)
    seg(52.5, 0, 52.5, 68)
    seg(0, _BOX_Y[0], 16.5, _BOX_Y[0]); seg(0, _BOX_Y[1], 16.5, _BOX_Y[1]); seg(16.5, _BOX_Y[0], 16.5, _BOX_Y[1])
    seg(105, _BOX_Y[0], 88.5, _BOX_Y[0]); seg(105, _BOX_Y[1], 88.5, _BOX_Y[1]); seg(88.5, _BOX_Y[0], 88.5, _BOX_Y[1])
    seg(0, _SIXBOX_Y[0], 5.5, _SIXBOX_Y[0]); seg(0, _SIXBOX_Y[1], 5.5, _SIXBOX_Y[1]); seg(5.5, _SIXBOX_Y[0], 5.5, _SIXBOX_Y[1])
    seg(105, _SIXBOX_Y[0], 99.5, _SIXBOX_Y[0]); seg(105, _SIXBOX_Y[1], 99.5, _SIXBOX_Y[1]); seg(99.5, _SIXBOX_Y[0], 99.5, _SIXBOX_Y[1])
    n = int(2 * math.pi * _CENTER_CIRCLE_R / step)
    for a in np.linspace(0, 2 * math.pi, n):
        pts.append((_CENTER[0] + _CENTER_CIRCLE_R * math.cos(a), _CENTER[1] + _CENTER_CIRCLE_R * math.sin(a)))
    for a in np.linspace(0, 2 * math.pi, n):
        x = _PENALTY_SPOT_L[0] + _CENTER_CIRCLE_R * math.cos(a)
        y = _PENALTY_SPOT_L[1] + _CENTER_CIRCLE_R * math.sin(a)
        if x > 16.5:
            pts.append((x, y))
        xr = _PENALTY_SPOT_R[0] + _CENTER_CIRCLE_R * math.cos(a)
        yr = _PENALTY_SPOT_R[1] + _CENTER_CIRCLE_R * math.sin(a)
        if xr < 88.5:
            pts.append((xr, yr))
    return pts


_PITCH_MARKING_ARRAY = np.array([(X, Y, 1.0) for X, Y in _pitch_marking_points()], dtype=np.float64)


def line_crosses_patch(ctx, frame_idx, rect, margin_px=60):
    """See the polish-pass investigation for why margin_px=60, not a tight
    few pixels: tested directly against real line-crossing cases, this
    pipeline's own homography position error ran 50-70px at the halfway
    line and ~25m of real pitch distance at the center circle - a tight
    margin missed both."""
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
    fallback for a patch flush against the frame edge)."""
    rx1, ry1, rx2, ry2 = [int(round(v)) for v in rect]
    w, h = rx2 - rx1, ry2 - ry1
    if w < 3 or h < 3:
        result = dest_img.copy()
        result[ry1:ry2, rx1:rx2] = patch_img
        return result, "hard-replace"

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


def segment_player(ctx, frame_idx, rect):
    """Background-differencing segmentation, identical method to the
    investigation's Step 2 - clean plate for `rect`, diff against the real
    frame, Otsu threshold, keep only the largest connected component."""
    found = find_clean_patch(ctx, frame_idx, rect, search_radius=150, max_blend_frames=3)
    if found is None:
        return None
    real = extract_patch(ctx.get_frame(frame_idx), rect)
    bg = found["patch"]
    if bg.shape[:2] != real.shape[:2]:
        bg = cv2.resize(bg, (real.shape[1], real.shape[0]))
    diff = cv2.absdiff(real, bg)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(diff_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    return {"real": real, "mask": mask, "background_patch": found}


def encode_cutout_png(real_bgr, mask):
    """A real player's segmented pixels as a base64 PNG with the
    segmentation mask as its alpha channel - transparent everywhere except
    the real, visible silhouette. This, not an ID-labeled marker icon, is
    what the drag UI should render and drag: a real player, cropped to
    their real outline, nothing synthetic added. Returns None if the crop
    is degenerate (can happen right at a frame edge)."""
    if real_bgr is None or real_bgr.size == 0:
        return None
    b, g, r = cv2.split(real_bgr)
    bgra = cv2.merge([b, g, r, mask])
    ok, buf = cv2.imencode('.png', bgra)
    if not ok:
        return None
    import base64
    return base64.b64encode(buf).decode('ascii')


# ==========================================================================
# BOUNDED SCALE — replaces the earlier gate. Never refuses; always returns
# a plausible pixel size, derived from real players already in this frame.
# ==========================================================================

def _frame_scale_trend(ctx, frame_idx):
    """Fits log(local_scale) ~= a + b*pixel_y from every REAL tracked
    player in this frame whose own position passes the same
    self-consistency check the earlier gate used as a hard reject (kept
    here as a soft filter on which samples to trust for the fit, not as a
    rejector of the fit's own output). Log-linear, not linear: perspective
    compresses multiplicatively with distance, and a log fit can never
    predict a negative or zero scale for extreme extrapolation the way a
    plain linear fit could.

    Also returns [min_height_px, max_height_px]: the plausible on-screen
    size bound, derived directly from every real player's own tracked bbox
    height in this frame (expanded by a margin) - literally the sizes
    already seen in this frame, not a meters conversion (see the
    polish-pass report for why a meters-based height check was tried and
    discarded).

    Cached per frame_idx on the context (this frame's real players don't
    change between multiple repositioning calls in the same session)."""
    if frame_idx in ctx._trend_cache:
        return ctx._trend_cache[frame_idx]

    ys, log_scales, heights = [], [], []
    for tid, info in ctx.players[frame_idx].items():
        x1, y1, x2, y2 = info["bbox"]
        h_px = y2 - y1
        if h_px < 15:
            continue
        heights.append(h_px)
        pitch = pixel_to_pitch(ctx, frame_idx, (x1 + x2) / 2, y2)
        if pitch is None or not _homography_trustworthy(ctx, frame_idx, *pitch):
            continue
        scale = local_pixel_scale(ctx, frame_idx, *pitch)
        if scale is None or scale <= 0:
            continue
        ys.append(y2)
        log_scales.append(math.log(scale))

    if len(ys) >= 2 and (max(ys) - min(ys)) > 5:
        b, a = np.polyfit(ys, log_scales, 1)
    elif len(ys) >= 1:
        # Not enough spread to fit a real trend - flat fallback at the one
        # (or few, identical-y) sample(s) available, still real data, just
        # not enough of it to say anything about how scale changes with
        # depth in this particular frame.
        a, b = float(np.mean(log_scales)), 0.0
    else:
        # No trustworthy on-pitch sample at all in this frame - fall back
        # to this frame's own pitch-center scale so there's still a real,
        # frame-specific anchor rather than an arbitrary constant.
        center_scale = local_pixel_scale(ctx, frame_idx, *_CENTER)
        a = math.log(center_scale) if center_scale and center_scale > 0 else math.log(15.0)
        b = 0.0

    if heights:
        min_h = max(8.0, 0.4 * min(heights))
        max_h = min(0.75 * ctx.frame_h, 2.5 * max(heights))
        if max_h < min_h:
            max_h = min_h * 1.5
    else:
        min_h, max_h = 15.0, 0.6 * ctx.frame_h

    result = {"a": float(a), "b": float(b), "min_height_px": min_h, "max_height_px": max_h,
              "n_samples": len(ys)}
    ctx._trend_cache[frame_idx] = result
    return result


def bounded_scale_for_target(ctx, frame_idx, source_bbox, target_pixel_xy):
    """The core replacement for the old scale-gate. Never returns a
    rejection - always a usable (new_width_px, new_height_px) for the
    resized cutout, plus metadata about how it was derived.

    Tries the real homography first (only when the target position both
    resolves to a pitch coordinate AND passes the self-consistency check);
    otherwise falls back to this frame's own log-linear perspective trend,
    fitted from real tracked players (see _frame_scale_trend). Either way,
    the final size is clamped to that same frame's real-player-derived
    plausible range, so the result can be approximate but is never absurd.
    """
    x1, y1, x2, y2 = source_bbox
    source_w, source_h = x2 - x1, y2 - y1
    tx, ty = target_pixel_xy

    trend = _frame_scale_trend(ctx, frame_idx)
    target_pitch = pixel_to_pitch(ctx, frame_idx, tx, ty)
    on_pitch = _is_really_on_pitch(ctx, frame_idx, target_pixel_xy, target_pitch)

    used_homography = False
    if target_pitch is not None and _homography_trustworthy(ctx, frame_idx, *target_pitch):
        target_scale = local_pixel_scale(ctx, frame_idx, *target_pitch)
        source_pitch = pixel_to_pitch(ctx, frame_idx, (x1 + x2) / 2, y2)
        if (target_scale is not None and target_scale > 0 and source_pitch is not None
                and _homography_trustworthy(ctx, frame_idx, *source_pitch)):
            source_scale = local_pixel_scale(ctx, frame_idx, *source_pitch)
            if source_scale and source_scale > 0:
                ratio = target_scale / source_scale
                used_homography = True

    if not used_homography:
        predicted_scale_at_target = math.exp(trend["a"] + trend["b"] * ty)
        source_scale_from_trend = math.exp(trend["a"] + trend["b"] * y2)
        ratio = predicted_scale_at_target / source_scale_from_trend if source_scale_from_trend > 0 else 1.0

    new_h = source_h * ratio
    new_h_clamped = max(trend["min_height_px"], min(trend["max_height_px"], new_h))
    clamped = abs(new_h_clamped - new_h) > 0.5
    new_w_clamped = source_w * (new_h_clamped / source_h) if source_h > 0 else source_w

    return {
        "new_width_px": new_w_clamped,
        "new_height_px": new_h_clamped,
        "ratio": new_h_clamped / source_h if source_h > 0 else 1.0,
        "used_homography": used_homography,
        "clamped": clamped,
        "on_pitch": on_pitch,
        "target_pitch": target_pitch if on_pitch else None,
        "trend_n_samples": trend["n_samples"],
    }


# ==========================================================================
# FULL PIPELINE
# ==========================================================================

def propose_reposition(ctx, frame_idx, source_track_id, target_pixel_xy, erase_margin=6, cutout_margin=15, base_img=None):
    """Full pipeline for one drag-and-drop move: erase the player from
    their original position (clean-plate), segment them out, resize per
    bounded_scale_for_target, and composite at the new position. Always
    succeeds - see module docstring for why - the only failure path is
    genuinely missing input data (no clean patch findable at all for the
    erase step, which the investigation measured at under 2% even with a
    tight 3s search window, effectively never with this function's wider
    default window).

    `base_img`, when given, is composited onto INSTEAD of a fresh read of
    the pristine frame - required for applying more than one move to the
    same frame (each subsequent move must build on the previous one's
    result, not silently discard it by re-reading the original frame).
    Every clean-plate SEARCH step (finding a background patch to erase
    into, or to segment a cutout against) still always reads real,
    untouched video frames regardless of base_img - only the two final
    "paint onto the canvas" steps (erasing the original position, pasting
    the resized cutout) use base_img as their destination. This is safe
    precisely because the search steps never look at the composite itself.

    Returns dict(composite_img, mask_debug, placement_info) or
    dict(error=...) for the rare missing-clean-patch case."""
    bbox = ctx.players[frame_idx].get(str(source_track_id), {}).get("bbox")
    if bbox is None:
        return {"error": f"player {source_track_id} has no tracked position in frame {frame_idx}"}

    # Clamp the target into the visible canvas rather than refusing - a
    # real drag interaction is already constrained to the image element by
    # the browser, so this only matters for a target a few pixels past the
    # edge (rounding, a drop right at the boundary); "no position should
    # ever be refused" applies here too, not just to calibration distrust.
    tx_raw, ty_raw = target_pixel_xy
    target_pixel_xy = (
        max(0.0, min(ctx.frame_w - 1.0, tx_raw)),
        max(0.0, min(ctx.frame_h - 1.0, ty_raw)),
    )

    x1, y1, x2, y2 = bbox
    erase_rect = (x1 - erase_margin, y1 - erase_margin, x2 + erase_margin, y2 + erase_margin)
    cutout_rect = (x1 - cutout_margin, y1 - cutout_margin, x2 + cutout_margin, y2 + cutout_margin)

    seg = segment_player(ctx, frame_idx, cutout_rect)
    if seg is None:
        return {"error": "no clean background patch available near this player - can't erase or cut them out"}

    erase_patch_result = find_clean_patch(ctx, frame_idx, erase_rect, search_radius=150, max_blend_frames=3)
    if erase_patch_result is None:
        return {"error": "no clean background patch available at the player's original position"}

    frame_img = base_img if base_img is not None else ctx.get_frame(frame_idx)
    composite, erase_method = composite_patch(frame_img, erase_patch_result["patch"], erase_rect)

    scale_info = bounded_scale_for_target(ctx, frame_idx, bbox, target_pixel_xy)
    ratio = scale_info["ratio"]
    new_w = max(1, int(round(seg["real"].shape[1] * ratio)))
    new_h = max(1, int(round(seg["real"].shape[0] * ratio)))
    resized_rgb = cv2.resize(seg["real"], (new_w, new_h))
    resized_mask = cv2.resize(seg["mask"], (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    foot_x_in_crop = (x1 + x2) / 2 - cutout_rect[0]
    foot_y_in_crop = y2 - cutout_rect[1]
    tx, ty = target_pixel_xy
    paste_x = int(round(tx - foot_x_in_crop * ratio))
    paste_y = int(round(ty - foot_y_in_crop * ratio))

    px1, py1 = max(0, paste_x), max(0, paste_y)
    px2, py2 = min(ctx.frame_w, paste_x + new_w), min(ctx.frame_h, paste_y + new_h)
    if px2 <= px1 or py2 <= py1:
        return {"error": "target position is entirely outside the visible frame"}

    sx1, sy1 = px1 - paste_x, py1 - paste_y
    sx2, sy2 = sx1 + (px2 - px1), sy1 + (py2 - py1)
    region_mask = resized_mask[sy1:sy2, sx1:sx2] > 0

    # mask-based paste (not seamlessClone) for the player cutout itself -
    # the cutout's own silhouette edge, not a rectangle, is the true
    # boundary here, and seamlessClone's rectangular mask assumption (used
    # for the ERASE step above, where the whole rect is real background)
    # doesn't fit a non-rectangular subject the same way; matches exactly
    # what the investigation's Step 3 already verified renders plausibly.
    dest = composite[py1:py2, px1:px2]
    dest[region_mask] = resized_rgb[sy1:sy2, sx1:sx2][region_mask]
    composite[py1:py2, px1:px2] = dest
    # No badge, no label, no marker baked in here by design - the moved
    # player must look exactly like any other real player in the frame,
    # nothing added to distinguish them (see module docstring's "Real
    # player, no synthetic overlay" note).

    return {
        "composite_img": composite,
        "erase_method": erase_method,
        "erase_sources": erase_patch_result["sources"],
        "cutout_sources": seg["background_patch"]["sources"],
        "paste_rect": (px1, py1, px2, py2),
        "scale_info": scale_info,
        "source_bbox": bbox,
        "error": None,
    }


# ==========================================================================
# PERSISTENCE — per-match, non-destructive (same discipline as
# corner_kicks.py / training_plan.py / player_labels.py)
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
