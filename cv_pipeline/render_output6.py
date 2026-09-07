"""
render_output6.py — Off-ball movement trails with camera compensation.

Trail history (fps*3 frames) stored in stabilised pixel coordinates
(position_adjusted). Before drawing each historical segment the cumulative
camera movement from that recording frame to the current frame is subtracted
so the trail appears pitch-relative on a panning camera.

Camera compensation uses a median-filtered version of the raw
camera_movement signal (window=5).  Single-frame optical-flow spikes
(up to ~72 px observed) are removed before cumulative prefix sums are
computed, so trails compensate smoothly through camera panning without
sudden jumps at spike frames.

Trail lifecycle (identity-linked, same proximity matching as the stamina
panel in render_output2.py):
  - Trails are keyed by a stable id (sid), not the raw tracker pid. When
    the tracker drops a pid and a new pid appears within PROXIMITY_PX of
    the lost player's last position (within LOST_BUF_FRAMES) AND on the
    SAME team, the new pid is mapped to the same sid, so its trail history
    continues instead of resetting. The team check (matching
    render_output3's identity-linker) exists because proximity alone is
    not enough to tell "same player, reassigned tracker id" apart from
    "different player who happened to pass nearby" — confirmed by the
    pid4/pid194 tracker-split case, where the two are on different teams;
    without a team check they were being silently glued into one trail,
    splicing one physical player's history onto the other's and drawing
    it in whichever team's color was current.
  - A trail is only drawn for a player with an active detection in the
    CURRENT frame — no orphaned/floating trails for players not in frame.
  - TRAIL_EXPIRE_FRAMES == LOST_BUF_FRAMES: a sid's trail buffer stays
    alive exactly as long as the sid itself stays eligible for re-linking.
    If the trail expired sooner, a player occluded just long enough to
    still be correctly re-identified would still find its trail already
    wiped, forcing a visible restart-from-empty for no reason.
  - A reappearing/newly-tracked player shows a trail immediately from
    whatever history exists (as few as MIN_TRAIL_PTS points) rather than
    waiting for the full 3-second buffer to fill.
  - RECORD_JUMP_PX rejection compares each new position against
    last_seen_pos (the previous frame's actual position), which always
    advances regardless of whether that point made it into the trail.
    A single spike frame is skipped without being drawn, but it never
    becomes a stale anchor — the next frame is judged against the real
    position, not the rejected one, so recording can't lock up forever.

Team 1/2 trail base color: read dynamically from each player's own
  'team_color' field (tracks['players'][...]['team_color'], the actual
  per-clip fitted centroid — the same field tracker.draw_annotations()
  reads), NOT a hardcoded red/green. On the original clip this happens to
  BE red/green, so no visible change there; T1_COLOR/T2_COLOR below are
  now only a fallback for the (normally unreachable) case where a frame's
  team_color is missing. A prior version hardcoded red/green outright,
  which rendered the wrong color for any clip whose kits aren't literally
  red/green (confirmed broken on a red-vs-navy clip during a scalability
  audit).
Team 3 (goalkeeper) trails: cyan/sky-blue BGR(255, 220, 0) — same color
  team=3 uses everywhere else in the pipeline (team_assigner.team_colors,
  render_output2's _bar_color/_dot_color). Chosen over the earlier
  yellow-green because that shade sat too close in hue to team 2's actual
  fitted green to read as visually distinct at render size — cyan is far
  from red, green, AND the referee's yellow all at once.
Each player's trail is drawn in a small, deterministic hue/value jitter of
  their team's base color (see _team_variant_color), so two teammates
  standing close together — same base hue, only fade differs otherwise —
  still read as individually traceable trails instead of one indistinct
  tangle (confirmed on the pid 6/194/7 cluster around frame 650).
Fade: 20% opacity (oldest) → 100% (newest). A small filled circle
  (HEAD_DOT_RADIUS) is always drawn at the trail's newest point at full
  brightness, so a near-stationary player's trail — which can collapse to
  a scribble too tiny to read as a line — still shows a visible marker
  rather than looking like no trail at all.
Segments with camera-compensated displacement > 50 px are skipped.

Saves: output_videos/output6.avi
"""

import os
import colorsys
import cv2
import numpy as np
from collections import defaultdict, deque
from tqdm import tqdm

T1_COLOR            = (  0,   0, 220)   # red   (BGR)
T2_COLOR            = (  0, 220,   0)   # green (BGR)
T3_COLOR            = (255, 220,   0)   # goalkeeper — cyan/sky-blue (BGR), matches
                                         # team_assigner.team_colors[3] elsewhere
MIN_TRAIL_PTS       = 2    # min points to draw a segment — show trails ASAP
JUMP_PX             = 50   # max camera-compensated px between consecutive trail pts
RECORD_JUMP_PX      = 25   # Fix 1: max position_adjusted jump allowed at recording time
OUTLIER_NEIGHBOR_PX = 35   # Fix 2: max draw-space distance from neighbor avg before rejection
HEAD_DOT_RADIUS     = 3    # small marker at each trail's newest point — keeps a
                           # near-stationary player's trail visible even when its
                           # path is too short/tight to read as a line at video scale

# Legibility fix: two teammates standing close together draw identical-hue
# trails that differ only by fade opacity, so in a tight cluster it's
# impossible to tell whose trail is whose (confirmed on the pid 6/194/7
# cluster around frame 650). Each sid gets a small, deterministic hue/value
# jitter within its team's base color — the trail still reads as "this
# team" (hue stays close to the base), but individual players get a
# distinguishable shade that stays constant for their whole trail history.
_VARIANT_BUCKETS = 7


def _team_variant_color(base_bgr, sid):
    b, g, r = base_bgr
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    hue_bucket = int(sid) % _VARIANT_BUCKETS
    val_bucket = (int(sid) * 3) % _VARIANT_BUCKETS   # decorrelated from hue_bucket
    half = (_VARIANT_BUCKETS - 1) / 2.0
    hue_offset = ((hue_bucket - half) / half) * 0.05   # up to ~+-18 degrees
    val_offset = ((val_bucket - half) / half) * 0.18   # up to ~+-18% brightness
    h2 = (h + hue_offset) % 1.0
    v2 = min(1.0, max(0.55, v + val_offset))
    r2, g2, b2 = colorsys.hsv_to_rgb(h2, s, v2)
    return (int(round(b2 * 255)), int(round(g2 * 255)), int(round(r2 * 255)))

PROXIMITY_PX        = 80   # identity-link: max px between lost and new detection
LOST_BUF_FRAMES     = 30   # identity-link: how long a lost sid stays eligible for re-matching
# Trail buffer must survive at least as long as a sid stays eligible for
# re-linking (LOST_BUF_FRAMES) — if it expired sooner, a player occluded
# 16-30 frames would get its identity correctly re-linked but find its
# whole trail history already wiped, forcing a visible restart-from-empty
# even though nothing about its identity actually reset.
TRAIL_EXPIRE_FRAMES = LOST_BUF_FRAMES


def render_output6(video_frames, tracks, team_ball_control, view_transformer,
                   fps, annotated_frames, homography_per_frame=None,
                   camera_movement_per_frame=None):
    if not annotated_frames:
        print("render_output6: no frames, skipping.")
        return

    orig_h, orig_w = annotated_frames[0].shape[:2]
    trail_len = int(round(fps * 3))   # 3 seconds of history — fps may be a
                                       # float (real video fps, e.g. 25.0);
                                       # deque(maxlen=...) requires an int.

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out    = cv2.VideoWriter('output_videos/output6.avi', fourcc, fps,
                             (orig_w, orig_h))

    n = len(annotated_frames)

    # ── Prefix sums with median-filtered camera movement ──────────────────────
    # Raw camera_movement_per_frame can contain single-frame optical-flow spikes
    # (up to ~72 px at frame 327 in 121364).  Applying a median filter with
    # window=5 before accumulating prefix sums removes isolated spikes while
    # preserving genuine sustained panning, so old trail points don't jump
    # in draw space when compensation passes through a spike frame.
    prefix_x = np.zeros(n, dtype=np.float64)
    prefix_y = np.zeros(n, dtype=np.float64)
    if camera_movement_per_frame is not None:
        raw_dx = np.zeros(n, dtype=np.float64)
        raw_dy = np.zeros(n, dtype=np.float64)
        for fn in range(n):
            try:
                move = camera_movement_per_frame[fn]
                raw_dx[fn] = float(move[0])
                raw_dy[fn] = float(move[1])
            except (IndexError, KeyError, TypeError):
                pass

        # Rolling median: stack K shifted views, take median across them.
        # 'edge' padding avoids zero-padding artefacts at start/end.
        K    = 5
        half = K // 2
        pdx  = np.pad(raw_dx, half, mode='edge')
        pdy  = np.pad(raw_dy, half, mode='edge')
        smooth_dx = np.median(
            np.stack([pdx[i : i + n] for i in range(K)], axis=0), axis=0
        )
        smooth_dy = np.median(
            np.stack([pdy[i : i + n] for i in range(K)], axis=0), axis=0
        )

        for fn in range(1, n):
            prefix_x[fn] = prefix_x[fn - 1] + smooth_dx[fn]
            prefix_y[fn] = prefix_y[fn - 1] + smooth_dy[fn]

    # ── Trail history, keyed by stable identity (sid) ──────────────────────────
    # Each entry: (frame_num, x_adj, y_adj) — position_adjusted at recording time
    trails         = defaultdict(lambda: deque(maxlen=trail_len))
    last_seen      = {}     # sid → last frame_num it had a detection
    last_seen_pos  = {}     # sid → (x, y) of the last valid position_adjusted seen,
                            # updated every frame regardless of trail acceptance —
                            # the jump check always compares against the real
                            # previous position, never a stale/rejected anchor.
    sid_last_center = {}    # sid → (cx, cy) from most recent bbox
    sid_team       = {}     # sid → team, from the most recent detection
    tracker_to_sid = {}     # current tracker pid → stable id
    recently_lost  = {}     # sid → (cx, cy, frame_lost, team)
    next_sid       = 1
    variant_color_cache = {}   # (team, sid) → jittered BGR, computed once per sid

    for frame_num in tqdm(range(n), desc="Rendering output6 (movement trails)"):
        player_data = (tracks['players'][frame_num]
                       if frame_num < len(tracks['players']) else {})

        frame = annotated_frames[frame_num].copy()

        # ── Identity linking (same proximity logic as render_output2) ─────────
        current_pids = set(player_data.keys())
        prev_pids    = set(tracker_to_sid.keys())
        appeared     = current_pids - prev_pids
        disappeared  = prev_pids    - current_pids

        for pid in disappeared:
            sid = tracker_to_sid.pop(pid, None)
            if sid is None:
                continue
            recently_lost[sid] = (*sid_last_center.get(sid, (0.0, 0.0)), frame_num,
                                  sid_team.get(sid, 0))

        stale = [s for s, (_, _, fn, _) in recently_lost.items()
                 if frame_num - fn > LOST_BUF_FRAMES]
        for s in stale:
            del recently_lost[s]

        matched_sids = set()
        for pid in sorted(appeared):
            bbox = player_data[pid].get('bbox')
            if bbox is None:
                sid = next_sid; next_sid += 1
                tracker_to_sid[pid] = sid
                continue

            ncx = (bbox[0] + bbox[2]) / 2.0
            ncy = (bbox[1] + bbox[3]) / 2.0
            team = player_data[pid].get('team', 0)

            # Only re-link to a lost sid of the SAME team — otherwise two
            # unrelated players passing close together (e.g. a duel, or a
            # tracker ID reassignment like the confirmed pid4/pid194 case)
            # get silently glued into one trail, splicing one physical
            # player's history onto another's and drawing it in whichever
            # team's color happens to be current.
            best_sid, best_d = None, PROXIMITY_PX + 1
            for sid, (cx, cy, _, lost_team) in recently_lost.items():
                if sid in matched_sids or lost_team != team:
                    continue
                d = np.sqrt((ncx - cx) ** 2 + (ncy - cy) ** 2)
                if d < best_d:
                    best_d, best_sid = d, sid

            if best_sid is not None:
                tracker_to_sid[pid] = best_sid
                matched_sids.add(best_sid)
                del recently_lost[best_sid]
            else:
                sid = next_sid; next_sid += 1
                tracker_to_sid[pid] = sid

        for pid in current_pids:
            if pid not in tracker_to_sid:
                sid = next_sid; next_sid += 1
                tracker_to_sid[pid] = sid

        # ── Record valid positions (keyed by sid, so identity carries over) ────
        for pid, info in player_data.items():
            sid = tracker_to_sid[pid]
            last_seen[sid] = frame_num
            sid_team[sid]  = info.get('team', 0)

            bbox = info.get('bbox')
            if bbox:
                sid_last_center[sid] = ((bbox[0] + bbox[2]) / 2.0,
                                        (bbox[1] + bbox[3]) / 2.0)

            pos = info.get('position_adjusted')
            if pos is None:
                continue
            try:
                x, y = float(pos[0]), float(pos[1])
            except Exception:
                continue
            if x == 0.0 and y == 0.0:
                continue   # (0,0) is an undetected-position placeholder

            # Fix 1: skip if position_adjusted jumped too far from the last
            # SEEN position — rejects camera-spike frames before they enter
            # the trail and create stray segments at draw time. last_seen_pos
            # always advances to this frame's real position below, whether or
            # not the point clears the jump check, so a single rejected spike
            # never becomes a permanently stale anchor that locks the trail up.
            last_pos = last_seen_pos.get(sid)
            if last_pos is None or (x - last_pos[0]) ** 2 + (y - last_pos[1]) ** 2 <= RECORD_JUMP_PX ** 2:
                trails[sid].append((frame_num, x, y))
            last_seen_pos[sid] = (x, y)

        # ── Expire stale trail history for sids not seen recently ──────────────
        for sid, seen_fn in last_seen.items():
            if frame_num - seen_fn > TRAIL_EXPIRE_FRAMES and trails[sid]:
                trails[sid].clear()

        # ── Draw trails for off-ball players actively visible this frame ──────
        ball_holders = {pid for pid, info in player_data.items()
                        if info.get('has_ball')}

        for pid, info in player_data.items():
            if pid in ball_holders:
                continue
            team = info.get('team', 0)
            if team not in (1, 2, 3):
                continue

            sid   = tracker_to_sid[pid]
            trail = list(trails[sid])
            if len(trail) < MIN_TRAIL_PTS:
                continue

            team_color = info.get('team_color')
            if team_color is not None and team in (1, 2):
                base_color = tuple(int(c) for c in team_color)
            else:
                base_color = T1_COLOR if team == 1 else T2_COLOR if team == 2 else T3_COLOR
            variant_color = variant_color_cache.get(sid)
            if variant_color is None:
                variant_color = _team_variant_color(base_color, sid)
                variant_color_cache[sid] = variant_color

            # Convert each historical position to current screen coordinates.
            # A point recorded at frame T appeared at screen (x_adj, y_adj) in
            # the stabilised frame.  Between T and frame_num, the camera has
            # drifted by (prefix[frame_num] - prefix[T]).  Subtracting that
            # drift compensates for camera panning so the trail stays
            # pitch-relative on the current video frame.
            cam_pts = []
            for (fn, x_adj, y_adj) in trail:
                cum_dx = prefix_x[frame_num] - prefix_x[fn]
                cum_dy = prefix_y[frame_num] - prefix_y[fn]
                draw_x = int(round(x_adj - cum_dx))
                draw_y = int(round(y_adj - cum_dy))
                cam_pts.append((x_adj, y_adj, draw_x, draw_y))

            # Fix 2: draw-time outlier rejection — remove any single point
            # whose draw position is more than OUTLIER_NEIGHBOR_PX from the
            # average of its immediate neighbors.  Handles residual spikes
            # that slipped through the recording-time filter.
            if len(cam_pts) >= 3:
                clean = [cam_pts[0]]
                for i in range(1, len(cam_pts) - 1):
                    _, _, cx, cy = cam_pts[i]
                    _, _, px, py = cam_pts[i - 1]
                    _, _, nx, ny = cam_pts[i + 1]
                    avg_x = (px + nx) * 0.5
                    avg_y = (py + ny) * 0.5
                    if (cx - avg_x) ** 2 + (cy - avg_y) ** 2 <= OUTLIER_NEIGHBOR_PX ** 2:
                        clean.append(cam_pts[i])
                clean.append(cam_pts[-1])
                cam_pts = clean

            # Draw segments with fade and jump filter
            for i in range(1, len(cam_pts)):
                xp, yp, dx1, dy1 = cam_pts[i - 1]
                xc, yc, dx2, dy2 = cam_pts[i]

                # Skip origin placeholders
                if (xp == 0.0 and yp == 0.0) or (xc == 0.0 and yc == 0.0):
                    continue

                # Skip if camera-compensated jump exceeds threshold
                disp = np.sqrt((dx2 - dx1) ** 2 + (dy2 - dy1) ** 2)
                if disp > JUMP_PX:
                    continue

                # Clip to frame bounds
                if not (0 <= dx1 < orig_w and 0 <= dy1 < orig_h and
                        0 <= dx2 < orig_w and 0 <= dy2 < orig_h):
                    continue

                # Fade: 20% opacity at oldest, 100% at newest
                t     = i / len(cam_pts)
                alpha = 0.20 + 0.80 * t
                col   = tuple(int(c * alpha) for c in variant_color)
                cv2.line(frame, (dx1, dy1), (dx2, dy2), col, 2, cv2.LINE_AA)

            # Head marker at the trail's newest point, full brightness. A
            # near-stationary player's trail can collapse to a scribble too
            # small to read as a line at video scale (confirmed: pid 6 at
            # 1.5km/h) — this keeps a real-but-tiny trail visibly present
            # instead of looking like no trail at all, and doubles as an
            # anchor a viewer can trace a crowded cluster's trails back to.
            _, _, hx, hy = cam_pts[-1]
            if 0 <= hx < orig_w and 0 <= hy < orig_h:
                cv2.circle(frame, (hx, hy), HEAD_DOT_RADIUS, variant_color, -1, cv2.LINE_AA)

        # Header label
        cv2.putText(frame, "OFF-BALL MOVEMENT", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

        out.write(frame)

    out.release()
    path    = 'output_videos/output6.avi'
    size_mb = os.path.getsize(path) / 1e6
    print(f"output6.avi saved ({n} frames, {size_mb:.1f} MB).")
