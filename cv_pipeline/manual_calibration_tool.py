#!/usr/bin/env python3
"""
manual_calibration_tool.py — Interactive per-frame calibration override.

Item 8's manual/semi-manual correction path: for one specific frame (or a
short, contiguous frame range) where the automatic 48-keypoint homography is
known-bad (see find_worst_frames-style self-consistency checks, or
KNOWN_ISSUES.md), a human operator re-pins a handful of real pitch landmarks
by eye and this tool solves + saves a corrected homography for just that
frame/range — the same shape of fix `label_keypoints.py` already does for the
global keypoint table, but per-frame instead of global, and starting from a
blank slate (the automatic detection is wrong, so nothing is pre-populated).

LEFT panel : the real frame, with the CURRENT (automatic) homography's pitch
             outline overlaid in RED so the operator can see exactly how it's
             wrong, plus any correspondences already clicked (yellow dots).
RIGHT panel: a schematic FIFA pitch diagram to click the matching landmark on.

Workflow
--------
1. Click a real, recognizable landmark in the LEFT frame (e.g. a penalty-box
   corner, the centre spot, a goal post) — a numbered yellow dot appears.
2. Click the SAME landmark's position on the RIGHT pitch diagram.
3. Repeat for >= 4 landmarks (more is better; RANSAC-free exact/least-squares
   fit is used since every point here is human-verified, not auto-detected).
4. Press P to preview: solves a homography from the correspondences so far
   and overlays it in GREEN on the LEFT panel, next to the RED original, for
   a direct before/after visual check.
5. Press S to save: writes the previewed homography into this segment's
   calibration_status.json under "frame_overrides", applied to every frame
   in --apply-to (default: just this one frame).
6. Press U to undo the last correspondence, Q/Esc to quit without saving.

This module's non-interactive functions (solve_homography_from_correspondences,
reproject_pitch_lines, save_frame_override, load_current_homography_for_frame)
are the same functions the interactive loop calls — they can be driven
directly by a script for automated/simulated verification without a GUI.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# ── Pitch geometry (metres) — same convention as pitch_calibrator.py ───────
PM_W = 105.0   # x: 0 = left goal line, 105 = right goal line
PM_H = 68.0    # y: 0 = far touchline,   68 = near touchline

# ── Window layout ───────────────────────────────────────────────────────────
VID_W, VID_H = 960, 540
PITCH_DRAW_W = 500
PITCH_DRAW_H = int(PITCH_DRAW_W * PM_H / PM_W)
MARGIN = 40
STATUS_H = 60
PITCH_PANEL_W = PITCH_DRAW_W + 2 * MARGIN
PITCH_PANEL_H = STATUS_H + PITCH_DRAW_H + MARGIN
WIN_H = max(VID_H, PITCH_PANEL_H)
WIN_W = VID_W + PITCH_PANEL_W
PITCH_OX = MARGIN
PITCH_OY = STATUS_H

C_LINE = (255, 255, 255)
C_PITCH_BG = (34, 139, 34)
C_PENDING = (0, 220, 255)   # yellow — clicked on LEFT, not yet assigned on RIGHT
C_ASSIGNED = (0, 200, 0)    # green — a completed correspondence
C_CURRENT_BAD = (0, 0, 255)     # red — the current (automatic) homography's overlay
C_CANDIDATE_GOOD = (0, 255, 0)  # green — the previewed corrected homography's overlay


# ==========================================================================
# CORE, GUI-FREE FUNCTIONS — the same ones the interactive loop calls, so a
# script can drive this tool's real logic for automated/simulated
# verification without a display or mouse.
# ==========================================================================

def solve_homography_from_correspondences(pix_pts, world_pts):
    """pix_pts, world_pts: lists of (x, y), same length, >= 4.

    Every correspondence here is human-verified (the operator identified a
    real landmark by eye), not auto-detected — so no RANSAC outlier
    rejection is needed or wanted; a plain least-squares fit (method=0) is
    used instead, same as label_keypoints.py's own diagnostic scripts use
    for hand-picked seed points (diag_minimal_h.py/h2.py/h3.py). Fit
    world->pixel then invert, matching pitch_calibrator.py's own
    world->pixel-then-invert convention exactly, for consistency with the
    rest of this pipeline.

    Returns (H, H_inv) as plain nested lists (JSON-serializable), or None if
    fewer than 4 points or the fit is degenerate.
    """
    if len(pix_pts) < 4 or len(pix_pts) != len(world_pts):
        return None
    pix = np.array(pix_pts, dtype=np.float32)
    world = np.array(world_pts, dtype=np.float32)
    H_inv, _ = cv2.findHomography(world, pix, 0)
    if H_inv is None:
        return None
    try:
        H = np.linalg.inv(H_inv)
    except np.linalg.LinAlgError:
        return None
    return H.tolist(), H_inv.tolist()


def reproject_pitch_lines(H_inv):
    """Returns {'boundary': [(px,py),...], 'halfway': [...], 'centre_circle': [...]}
    - the same handful of pitch markings label_keypoints.py's diagram draws,
    reprojected into pixel space via H_inv, for overlaying on a real frame."""
    H_inv = np.array(H_inv, dtype=np.float64)

    def to_px(wx, wy):
        v = H_inv @ np.array([wx, wy, 1.0])
        if abs(v[2]) < 1e-9:
            return None
        return (v[0] / v[2], v[1] / v[2])

    boundary = [to_px(x, y) for x, y in [(0, 0), (PM_W, 0), (PM_W, PM_H), (0, PM_H), (0, 0)]]
    halfway = [to_px(52.5, 0), to_px(52.5, PM_H)]
    centre_circle = []
    for deg in range(0, 361, 10):
        rad = np.radians(deg)
        wx = 52.5 + 9.15 * np.cos(rad)
        wy = 34.0 + 9.15 * np.sin(rad)
        centre_circle.append(to_px(wx, wy))
    left_box = [to_px(0, 13.84), to_px(16.5, 13.84), to_px(16.5, 54.16), to_px(0, 54.16)]
    right_box = [to_px(105, 13.84), to_px(88.5, 13.84), to_px(88.5, 54.16), to_px(105, 54.16)]
    return {
        "boundary": boundary, "halfway": halfway, "centre_circle": centre_circle,
        "left_box": left_box, "right_box": right_box,
    }


def draw_pitch_overlay(frame, H_inv, color, thickness=2):
    """Draws the reprojected pitch lines from reproject_pitch_lines onto a
    copy of frame, skipping any point that failed to project (out of the
    frame, or a degenerate divide) rather than crashing on it."""
    out = frame.copy()
    lines = reproject_pitch_lines(H_inv)

    def draw_poly(pts, closed=False):
        pts = [p for p in pts if p is not None]
        if len(pts) < 2:
            return
        pts_int = np.array([[int(x), int(y)] for x, y in pts], dtype=np.int32)
        cv2.polylines(out, [pts_int], closed, color, thickness, cv2.LINE_AA)

    draw_poly(lines["boundary"], closed=True)
    draw_poly(lines["halfway"])
    draw_poly(lines["centre_circle"], closed=True)
    draw_poly(lines["left_box"], closed=True)
    draw_poly(lines["right_box"], closed=True)
    return out


def load_current_homography_for_frame(cv_output_dir, frame_idx):
    """Reads repositioning_data.json (the same file player_repositioning.py's
    Game Board reads) for this frame's CURRENT (automatic) homography, so the
    tool can show the operator exactly what's wrong before they fix it.
    Returns (H, H_inv) or None if unavailable."""
    data_path = Path(cv_output_dir) / "repositioning_data.json"
    if not data_path.exists():
        return None
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    homography = data.get("homography", {})
    h = homography.get(str(frame_idx))
    return tuple(h) if h else None


def _calibration_status_path(cv_output_dir):
    return Path(cv_output_dir) / "calibration_status.json"


def load_calibration_status_raw(cv_output_dir):
    path = _calibration_status_path(cv_output_dir)
    if not path.exists():
        return {"reliable": True, "note": None, "frame_overrides": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"reliable": True, "note": None, "frame_overrides": {}}
    data.setdefault("frame_overrides", {})
    return data


def save_frame_override(cv_output_dir, frame_indices, H, H_inv, correspondences):
    """Writes a manual per-frame correction into calibration_status.json's
    frame_overrides — extends the existing per-segment reliable/note schema
    (see corner_kicks.py's load_calibration_status and KNOWN_ISSUES.md)
    rather than inventing a new file, per the same "extend, don't invent a
    parallel mechanism" discipline this project already follows for
    training plans / corner kicks / player labels. Applying the SAME H/H_inv
    to every frame_idx in `frame_indices` keeps every consumer's read side a
    plain per-frame dict lookup - no range logic needed downstream."""
    path = _calibration_status_path(cv_output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = load_calibration_status_raw(cv_output_dir)
    entry = {
        "H": H, "H_inv": H_inv,
        "correspondences": correspondences,
        "corrected_at": datetime.now(timezone.utc).isoformat(),
        "corrected_by": "manual_calibration_tool",
    }
    for f in frame_indices:
        data["frame_overrides"][str(f)] = entry
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)
    return path


# ==========================================================================
# Pitch diagram (right panel) — same drawing style as label_keypoints.py
# ==========================================================================

def w2p(wx, wy):
    px = PITCH_OX + int(wx / PM_W * PITCH_DRAW_W)
    py = PITCH_OY + int(wy / PM_H * PITCH_DRAW_H)
    return px, py


def p2w(px, py):
    wx = (px - PITCH_OX) / PITCH_DRAW_W * PM_W
    wy = (py - PITCH_OY) / PITCH_DRAW_H * PM_H
    return max(0.0, min(PM_W, wx)), max(0.0, min(PM_H, wy))


def make_pitch_base():
    panel = np.full((PITCH_PANEL_H, PITCH_PANEL_W, 3), C_PITCH_BG, dtype=np.uint8)
    lw = 2
    cv2.rectangle(panel, w2p(0, 0), w2p(PM_W, PM_H), C_LINE, lw)
    cv2.line(panel, w2p(52.5, 0), w2p(52.5, PM_H), C_LINE, lw)
    r_px = int(9.15 / PM_W * PITCH_DRAW_W)
    cv2.circle(panel, w2p(52.5, 34.0), r_px, C_LINE, lw)
    cv2.circle(panel, w2p(52.5, 34.0), 4, C_LINE, -1)
    cv2.rectangle(panel, w2p(0, 13.84), w2p(16.5, 54.16), C_LINE, lw)
    cv2.rectangle(panel, w2p(0, 24.84), w2p(5.5, 43.16), C_LINE, lw)
    cv2.circle(panel, w2p(11.0, 34.0), 4, C_LINE, -1)
    cv2.rectangle(panel, w2p(88.5, 13.84), w2p(PM_W, 54.16), C_LINE, lw)
    cv2.rectangle(panel, w2p(99.5, 24.84), w2p(PM_W, 43.16), C_LINE, lw)
    cv2.circle(panel, w2p(94.0, 34.0), 4, C_LINE, -1)
    for xm in [0, 16.5, 52.5, 88.5, 105]:
        ppx, _ = w2p(xm, 0)
        cv2.putText(panel, f'{xm:.0f}', (ppx - 10, PITCH_OY - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_LINE, 1)
    for ym in [0, 13.84, 34, 54.16, 68]:
        _, ppy = w2p(0, ym)
        cv2.putText(panel, f'{ym:.0f}', (2, ppy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_LINE, 1)
    return panel


# ==========================================================================
# Interactive main
# ==========================================================================

def _parse_range(spec, default_frame):
    if not spec:
        return [default_frame]
    if ":" in spec:
        lo, hi = spec.split(":")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv-output-dir", required=True, help="e.g. output_videos/liverpool_psg_verified")
    ap.add_argument("--video", required=True, help="path to this match's clip")
    ap.add_argument("--frame", type=int, required=True, help="the frame index to view/correct")
    ap.add_argument("--apply-to", default=None, help="'start:end' or 'a,b,c' - frames the saved correction applies to (default: just --frame)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise IOError(f"could not read frame {args.frame} from {args.video}")

    sx, sy = VID_W / frame.shape[1], VID_H / frame.shape[0]
    base_frame = cv2.resize(frame, (VID_W, VID_H))
    base_pitch = make_pitch_base()

    current_h = load_current_homography_for_frame(args.cv_output_dir, args.frame)
    if current_h is not None:
        H_scaled_inv = np.array(current_h[1]) @ np.diag([1 / sx, 1 / sy, 1])
        base_frame_with_bad = draw_pitch_overlay(base_frame, H_scaled_inv.tolist(), C_CURRENT_BAD)
    else:
        base_frame_with_bad = base_frame.copy()
        print("No existing homography found for this frame - nothing to compare against.")

    clicks = []       # list of {"px":(x,y) in FULL-res frame coords, "world": (wx,wy) or None}
    preview_H = None
    preview_H_inv = None

    WIN = f"Manual Calibration - frame {args.frame}  [click frame, then pitch]  P=preview S=save U=undo Q=quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, WIN_W, WIN_H)

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if x < VID_W:
            full_px = (x / sx, y / sy)
            clicks.append({"px": full_px, "px_display": (x, y), "world": None})
            print(f"  point {len(clicks)}: clicked frame at display ({x},{y}) -> full-res {full_px}")
        else:
            pending = next((c for c in clicks if c["world"] is None), None)
            if pending is None:
                return
            rx, ry = x - VID_W, y
            if PITCH_OX <= rx <= PITCH_OX + PITCH_DRAW_W and PITCH_OY <= ry <= PITCH_OY + PITCH_DRAW_H:
                wx, wy = p2w(rx, ry)
                pending["world"] = (round(wx, 2), round(wy, 2))
                print(f"  -> assigned world ({wx:.2f}, {wy:.2f})")

    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        left = base_frame_with_bad.copy()
        right = base_pitch.copy()
        for i, c in enumerate(clicks):
            color = C_ASSIGNED if c["world"] is not None else C_PENDING
            cv2.circle(left, c["px_display"], 6, color, -1)
            cv2.putText(left, str(i + 1), (c["px_display"][0] + 8, c["px_display"][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            if c["world"] is not None:
                ppx, ppy = w2p(*c["world"])
                cv2.circle(right, (ppx, ppy), 5, color, -1)
                cv2.putText(right, str(i + 1), (ppx + 6, ppy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
        if preview_H_inv is not None:
            H_scaled_inv = np.array(preview_H_inv) @ np.diag([1 / sx, 1 / sy, 1])
            left = draw_pitch_overlay(left, H_scaled_inv.tolist(), C_CANDIDATE_GOOD)

        n_ready = sum(1 for c in clicks if c["world"] is not None)
        status = f"{n_ready} correspondences ready (need >=4)  RED=current(bad)  GREEN=candidate(preview)"
        cv2.putText(right, status, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1)
        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        canvas[:VID_H, :VID_W] = left
        canvas[:PITCH_PANEL_H, VID_W:VID_W + PITCH_PANEL_W] = right
        cv2.line(canvas, (VID_W, 0), (VID_W, WIN_H), (180, 180, 180), 2)
        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            print("Quit without saving.")
            break
        elif key in (ord('u'), ord('U')):
            if clicks:
                clicks.pop()
                preview_H = preview_H_inv = None
        elif key in (ord('p'), ord('P')):
            ready = [c for c in clicks if c["world"] is not None]
            if len(ready) < 4:
                print(f"Need >=4 correspondences, have {len(ready)}.")
                continue
            result = solve_homography_from_correspondences(
                [c["px"] for c in ready], [c["world"] for c in ready])
            if result is None:
                print("Homography solve failed - check for collinear/duplicate points.")
                continue
            preview_H, preview_H_inv = result
            print("Preview solved - GREEN overlay updated.")
        elif key in (ord('s'), ord('S')):
            if preview_H is None:
                print("Press P to preview/solve first.")
                continue
            frames = _parse_range(args.apply_to, args.frame)
            ready = [c for c in clicks if c["world"] is not None]
            saved_path = save_frame_override(
                args.cv_output_dir, frames, preview_H, preview_H_inv,
                [[c["px"][0], c["px"][1], c["world"][0], c["world"][1]] for c in ready])
            print(f"Saved override for frames {frames} -> {saved_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
