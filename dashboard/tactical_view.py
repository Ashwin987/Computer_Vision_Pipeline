"""
tactical_view.py — the "Tactical Map" tab: a 2D top-down schematic pitch
view (not the real broadcast frame) showing every player's real calibrated
position as a team-colored dot, a per-team convex-hull shape outline, the
ball's position, and each player's current speed as a label near their dot.

This is a genuinely different feature from Game Board (player_repositioning.py):
Game Board deliberately draws on the real video frame at real tracked PIXEL
positions with no homography involved, specifically because reprojecting
through the homography onto a flat diagram was found to visibly misalign
(see that module's own docstring for the full investigation). This feature
is the opposite case on purpose — a flat schematic diagram is exactly what
a top-down tactical map is supposed to be, so it embraces the homography
reprojection Game Board avoids. That's fine here because placement inaccuracy
on an abstract diagram (no real photo to visibly clash against) doesn't
have the same "obviously wrong" failure mode Game Board's investigation
found — the two features made different, and both correct, calls for their
own context.

Data path: every position here goes through pixel_to_pitch on
player_repositioning.py's own RepositioningContext, whose ctx.homography
already has this match's manual per-frame calibration corrections
(calibration_status.json's frame_overrides) merged on top of the automatic
homography — see player_repositioning.load_context's own docstring. This
is the single thing that distinguishes "current" from "stale" position data
in this codebase; this module never reads a pickled homography_per_frame
directly.

History note: an inventory of this project's four (really: three, plus one
red herring) prior partial attempts at a top-down view found none of them
wired to that corrected path — cv_pipeline/tactical_map/tactical_map_generator.py
and cv_pipeline/tactical_output.py are both complete, axis-correct
(105x68m FIFA convention), but fully orphaned batch-video renderers with
no calibration_status/frame_overrides awareness at all; the small,
genuinely on-point convex-hull-per-team drawing code lived in
dashboard/corner_kicks.py's now-deleted render_team_shape_video (recovered
here from git commit 6124202) but read from the stale, override-unaware
player_positions.json/reconstruct_positions.py path. This module borrows
that drawing style (grass-green cv2 pitch diagram, per-team dot+hull loop)
but rebuilds the data path on the current one, and adds the ball dot and
per-player speed labels that render_team_shape_video never had.
"""
import numpy as np
import cv2

import player_repositioning as pr

PITCH_LENGTH_M = pr.PITCH_LENGTH_M   # 105.0
PITCH_WIDTH_M = pr.PITCH_WIDTH_M     # 68.0
_BOUNDS_MARGIN_M = pr._BOUNDS_MARGIN_M  # 2.0 — same off-pitch tolerance as Game Board's stat gate

PX_PER_M = 10
PAD_PX = 26
CANVAS_W = int(PITCH_LENGTH_M * PX_PER_M) + 2 * PAD_PX
CANVAS_H = int(PITCH_WIDTH_M * PX_PER_M) + 2 * PAD_PX

# Speed smoothing window, in frames each side of the current one — matches
# this project's own established convention for instantaneous-speed
# smoothing (speed_and_distance_estimator.py's BUFFER_SIZE=3 rolling window,
# and cv_pipeline/tactical_map/tactical_map_generator.py's own VEL_SMOOTH=3
# for the same reason: a 1-frame delta is too noisy against tracker jitter).
SPEED_WINDOW_FRAMES = 3

# Same implausible-speed reject threshold cv_pipeline/speed_and_distance_estimator.py
# already uses (its own MAX_SPEED_KMH) — found necessary here the same way:
# real data check on frame 594 (inside group 98's 3-frame span, right next to
# a probable scene-cut/shot-boundary at 592->593 and 595->596) showed track
# ids reporting 189-242 km/h, traced to a ~40-55m single-frame position jump
# for the same track id — a tracking-ID discontinuity across the cut, not
# real movement or a calibration bug. Rather than trying to detect cuts
# directly, this reuses the project's own established physical-plausibility
# gate: an impossible speed is reported as unavailable, never displayed.
MAX_PLAUSIBLE_SPEED_KMH = 36.0

BALL_COLOR_BGR = (255, 255, 255)
HULL_FILL_ALPHA = 0.16


def _contrast_text_color(bgr):
    """Black or white, whichever reads better on this dot color — found
    necessary on real data: barca_madrid_pt1's actual curated team color for
    one team is literal white (its bundle sets color_a="White"), which made
    that team's white player-ID text over a white dot fill completely
    unreadable, and its white hull outline visually merge with this pitch
    diagram's own white line markings. Standard relative-luminance threshold,
    not a special case for white specifically — any light team color hits
    the same problem, so this fixes the general case."""
    b, g, r = bgr
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (20, 20, 20) if luminance > 150 else (255, 255, 255)


def _draw_outlined_polyline(canvas, pts_arr, color, thickness=2):
    """Team-colored line with a black halo underneath, so it stays visible
    against both the green pitch and this diagram's own white markings —
    same reasoning as _contrast_text_color."""
    cv2.polylines(canvas, [pts_arr], isClosed=True, color=(0, 0, 0),
                  thickness=thickness + 2, lineType=cv2.LINE_AA)
    cv2.polylines(canvas, [pts_arr], isClosed=True, color=color,
                  thickness=thickness, lineType=cv2.LINE_AA)


def _pt(x, y):
    """Pitch metres (x=length 0-105, y=width 0-68) -> canvas pixel."""
    return (int(round(PAD_PX + x * PX_PER_M)), int(round(PAD_PX + y * PX_PER_M)))


def _build_pitch_background():
    img = np.full((CANVAS_H, CANVAS_W, 3), (34, 139, 34), dtype=np.uint8)  # grass green, BGR
    line = (235, 235, 235)
    L, W = PITCH_LENGTH_M, PITCH_WIDTH_M

    cv2.rectangle(img, _pt(0, 0), _pt(L, W), line, 2)
    cv2.line(img, _pt(L / 2, 0), _pt(L / 2, W), line, 2)
    cv2.circle(img, _pt(L / 2, W / 2), int(round(9.15 * PX_PER_M)), line, 2)
    cv2.circle(img, _pt(L / 2, W / 2), 3, line, -1)

    # Penalty boxes, six-yard boxes, penalty spots — real FIFA dimensions,
    # same values player_repositioning.py's own _pitch_marking_points uses.
    cv2.rectangle(img, _pt(0, 13.84), _pt(16.5, 54.16), line, 2)
    cv2.rectangle(img, _pt(0, 24.84), _pt(5.5, 43.16), line, 2)
    cv2.rectangle(img, _pt(L - 16.5, 13.84), _pt(L, 54.16), line, 2)
    cv2.rectangle(img, _pt(L - 5.5, 24.84), _pt(L, 43.16), line, 2)
    cv2.circle(img, _pt(11.0, W / 2), 2, line, -1)
    cv2.circle(img, _pt(L - 11.0, W / 2), 2, line, -1)

    # Goals — small rectangles poking out past each goal line, purely
    # decorative (not a real correspondence landmark used anywhere else).
    goal_depth_m, goal_y = 1.8, (30.34, 37.66)
    cv2.rectangle(img, _pt(-goal_depth_m, goal_y[0]), _pt(0, goal_y[1]), line, 2)
    cv2.rectangle(img, _pt(L, goal_y[0]), _pt(L + goal_depth_m, goal_y[1]), line, 2)

    return img


_PITCH_BG = _build_pitch_background()


def _in_pitch_bounds(X, Y):
    return (-_BOUNDS_MARGIN_M <= X <= PITCH_LENGTH_M + _BOUNDS_MARGIN_M
            and -_BOUNDS_MARGIN_M <= Y <= PITCH_WIDTH_M + _BOUNDS_MARGIN_M)


def player_pitch_position(ctx, frame_idx, tid):
    """This player's real calibrated pitch position (metres) at this exact
    frame, or None if they're not tracked in this frame, off-pitch, or this
    frame has no usable homography (including a frame past ctx.n_frames)."""
    if frame_idx < 0 or frame_idx >= ctx.n_frames:
        return None
    info = ctx.players[frame_idx].get(str(tid)) or ctx.players[frame_idx].get(tid)
    if info is None:
        return None
    x1, y1, x2, y2 = info["bbox"]
    pos = pr.pixel_to_pitch(ctx, frame_idx, (x1 + x2) / 2.0, y2)
    if pos is None or not _in_pitch_bounds(*pos):
        return None
    return pos


def frame_player_positions(ctx, frame_idx):
    """{track_id (str): (X, Y) metres} for every on-pitch player this frame.
    A player tracked but currently off-pitch-bounds (or with no usable
    homography) is left out rather than drawn at a meaningless position —
    same honesty convention compute_placement_stats already follows."""
    positions = {}
    for tid, info in ctx.players[frame_idx].items():
        x1, y1, x2, y2 = info["bbox"]
        pos = pr.pixel_to_pitch(ctx, frame_idx, (x1 + x2) / 2.0, y2)
        if pos is not None and _in_pitch_bounds(*pos):
            positions[tid] = pos
    return positions


def ball_pitch_position(ctx, frame_idx):
    """The ball's real calibrated pitch position this frame, or None if not
    detected/interpolated for this frame, off-pitch, or no usable homography."""
    ball_tracks = ctx.ball[frame_idx] if 0 <= frame_idx < len(ctx.ball) else {}
    if not ball_tracks:
        return None
    info = next(iter(ball_tracks.values()))
    x1, y1, x2, y2 = info["bbox"]
    pos = pr.pixel_to_pitch(ctx, frame_idx, (x1 + x2) / 2.0, (y1 + y2) / 2.0)
    if pos is None or not _in_pitch_bounds(*pos):
        return None
    return pos


def player_speed_kmh(ctx, frame_idx, tid, window=SPEED_WINDOW_FRAMES):
    """Instantaneous speed (km/h), smoothed over a small frame window either
    side of frame_idx — same reasoning as this project's own
    speed_and_distance_estimator.py (BUFFER_SIZE=3) and
    tactical_map_generator.py (VEL_SMOOTH=3): a single-frame delta is mostly
    tracker/homography jitter, not real movement. Returns None (never a
    fabricated number) if the player isn't resolvable at both window edges,
    e.g. they just entered/left frame or those frames' homography failed."""
    f0 = max(0, frame_idx - window)
    f1 = min(ctx.n_frames - 1, frame_idx + window)
    if f0 == f1:
        return None
    p0 = player_pitch_position(ctx, f0, tid)
    p1 = player_pitch_position(ctx, f1, tid)
    if p0 is None or p1 is None:
        return None
    dt = (f1 - f0) / ctx.fps
    if dt <= 0:
        return None
    dist_m = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
    speed_kmh = dist_m / dt * 3.6
    if speed_kmh > MAX_PLAUSIBLE_SPEED_KMH:
        return None
    return speed_kmh


def team_hull_points(positions, player_team, team_num, exclude_ids=()):
    """cv2.convexHull of one team's on-pitch player positions this frame (as
    canvas pixels, ready to draw), or None with fewer than 3 usable points —
    a hull needs 3+ points to mean anything; 2 points still gets a plain
    connecting line, 0-1 gets nothing, both handled by the caller rather
    than forcing a degenerate shape."""
    pts = [
        _pt(*pos) for tid, pos in positions.items()
        if player_team.get(str(tid)) == team_num and tid not in exclude_ids
    ]
    if len(pts) < 2:
        return None, pts
    arr = np.array(pts, dtype=np.int32)
    if len(pts) == 2:
        return None, pts
    hull = cv2.convexHull(arr)
    return hull, pts


def render_topdown_frame(ctx, frame_idx, player_team, team_bgr,
                          goalkeeper_ids=(), exclude_gk_from_hull=True,
                          show_speed_labels=True):
    """Renders one frame of the top-down tactical map. Returns (image, meta)
    where image is a BGR canvas (ready for st.image(..., channels='BGR'))
    and meta reports what was actually drawable this frame — same honesty
    convention as clean_frame_no_players' (cleaned_img, n_failed) — so a
    frame with, say, only 8 of 22 players on-pitch-bounds is visible in the
    return value, not silently hidden.

    goalkeeper_ids: track ids known to be a goalkeeper (from repositioning_data.json's
    is_goalkeeper flag) — still drawn as a dot+speed label, just excluded
    from their team's hull by default (a GK's usual position badly distorts
    a team-shape hull; this is a real, disclosed choice, not a bug)."""
    canvas = _PITCH_BG.copy()
    positions = frame_player_positions(ctx, frame_idx)
    n_tracked = len(ctx.players[frame_idx])
    n_on_pitch = len(positions)

    gk_ids = set(str(g) for g in goalkeeper_ids)
    hull_meta = {}
    for team_num in (1, 2):
        color = team_bgr.get(team_num, (150, 150, 150))
        exclude = gk_ids if exclude_gk_from_hull else set()
        hull, pts = team_hull_points(positions, player_team, team_num, exclude_ids=exclude)
        if hull is not None:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [hull], color)
            cv2.addWeighted(overlay, HULL_FILL_ALPHA, canvas, 1 - HULL_FILL_ALPHA, 0, dst=canvas)
            _draw_outlined_polyline(canvas, hull, color, thickness=2)
        elif len(pts) == 2:
            cv2.line(canvas, pts[0], pts[1], (0, 0, 0), 4, cv2.LINE_AA)
            cv2.line(canvas, pts[0], pts[1], color, 2, cv2.LINE_AA)
        hull_meta[team_num] = len(pts)

    for tid, pos in positions.items():
        team_num = player_team.get(str(tid))
        color = team_bgr.get(team_num, (150, 150, 150))
        text_color = _contrast_text_color(color)
        cx, cy = _pt(*pos)
        cv2.circle(canvas, (cx, cy), 8, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(tid), (cx - 8, cy + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, text_color, 1, cv2.LINE_AA)
        if show_speed_labels:
            speed = player_speed_kmh(ctx, frame_idx, tid)
            if speed is not None:
                cv2.putText(canvas, f"{speed:.1f}", (cx - 10, cy - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (20, 20, 20), 2, cv2.LINE_AA)
                cv2.putText(canvas, f"{speed:.1f}", (cx - 10, cy - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 230, 120), 1, cv2.LINE_AA)

    ball_pos = ball_pitch_position(ctx, frame_idx)
    if ball_pos is not None:
        bx, by = _pt(*ball_pos)
        cv2.circle(canvas, (bx, by), 5, BALL_COLOR_BGR, -1, cv2.LINE_AA)
        cv2.circle(canvas, (bx, by), 5, (30, 30, 30), 1, cv2.LINE_AA)

    meta = {
        "n_tracked": n_tracked,
        "n_on_pitch": n_on_pitch,
        "n_off_pitch": n_tracked - n_on_pitch,
        "team1_hull_pts": hull_meta.get(1, 0),
        "team2_hull_pts": hull_meta.get(2, 0),
        "ball_visible": ball_pos is not None,
    }
    return canvas, meta
