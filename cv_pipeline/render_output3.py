"""
render_output3.py — Per-player spatial control, grid-based.

Uses the same pixel-space grid approach as render_output4 (pitch control),
but classifies each cell by the nearest individual player rather than the
nearest team.  No per-player polygon math, no Voronoi library dependency,
no flickering.

Method:
  1. Build a pitch mask from PITCH_CLIP (fillPoly, pixel space).
  2. Build a coarse grid: every STEP-th pixel in both axes.
  3. Per frame: for each grid cell inside the pitch, find the nearest
     player by Euclidean distance in position_adjusted space.
  4. Colour the cell with that player's pastel colour.
  5. Upsample grid to full frame (INTER_NEAREST → solid blocks).
  6. GaussianBlur(61) smooths zone boundaries.
  7. Apply pitch mask, blend 35% over source frame.

Outfield player colours (team 1/2) are deterministic (identity-based) and
team-INDEPENDENT: every outfield player gets a colour from a curated,
hand-picked palette of maximally perceptually-distinct colours (red, blue,
green, yellow, magenta, cyan, orange, purple, lime, pink, brown, teal,
navy, maroon, etc.), assigned in merged-identity order. Evenly spacing
hues around the wheel only put neighbours ~13° apart, which still read as
similar; a curated distinct-colour list keeps every pair of players
visually separable. Players beyond the palette size get saturation/value-
varied repeats of the same base hues so the palette never runs out.

Goalkeepers (team 3) do NOT draw from this palette — there's only one per
side, so nothing needs telling apart, and using a fixed colour (GK_COLOR)
instead of an identity-derived one means their colour can't change if
their track id fragments and identity-linking fails to re-merge it back
to the same merged identity.

Base layer is the RAW source video (not output_video.avi), since the
latter already has opaque team-colour ellipses baked in by
tracker.draw_annotations() which would visually dominate over the
per-player zone colours. Player markers (small filled circle + ID
label) are drawn here directly, in the same per-player palette colour
as the zone overlay, so marker and zone always match.

The raw frames are passed in by the caller (video_frames — the same
already-decoded, pre-annotation frames main.py reads once via
read_video() and reuses everywhere else) rather than this module
re-reading the source video from disk itself. Previously this read a
hardcoded path (Match_videos/121364_0.mp4) unconditionally, regardless
of which clip was actually being processed — harmless as long as
main.py was only ever pointed at that one clip, but silently wrong
(showing a different match's footage entirely) the first time it was
pointed at anything else.

Tracker ID fragmentation (a real player losing tracking and coming back
under a new pid) is resolved with the same proximity-based identity
linking used by render_output2.py's stamina panel: a new pid appearing
within PROXIMITY_PX of a same-team pid lost within the last
LOST_BUF_FRAMES frames is treated as the same real player and merged
into that player's existing stable identity (sid). Colour is assigned
per merged identity, not per raw tracker pid, so a player keeps one
colour through ID switches. Any identity — merged or not — that still
totals fewer than MIN_APPEARANCE_FRAMES frames is treated as leftover
tracker noise and excluded from colour assignment and zone computation.

Saves:  output_videos/output3.avi
"""

import os
import colorsys
import cv2
import numpy as np
from tqdm import tqdm

from tactical_events.space_control import (
    build_pitch_mask, build_sampling_grid, nearest_player_per_cell,
    compute_pitch_verts_from_tracks,
)

BLEND_A = 0.35
BLUR_K  = 61
BLUR_S  = 20.0
STEP    = 8

MIN_APPEARANCE_FRAMES = 30   # merged identities seen fewer times than this are noise
MARKER_RADIUS        = 8
MARKER_TEXT_COLOR    = (255, 255, 255)

PROXIMITY_PX     = 80   # identity-link: max px between a lost id and a new detection
LOST_BUF_FRAMES  = 30   # identity-link: how long a lost id stays eligible for re-matching

# ── Player colour: hand-picked maximally-distinct palette, team-independent ──
# Curated for pairwise perceptual separation (hue, saturation AND value all
# vary between entries — not just evenly-spaced hue) so no two players read
# as "similar", in contrast to a pure evenly-spaced hue wheel.
_DISTINCT_PALETTE_RGB = [
    (230,  25,  75),  # red
    ( 60, 180,  75),  # green
    (255, 225,  25),  # yellow
    (  0, 130, 200),  # blue
    (245, 130,  48),  # orange
    (145,  30, 180),  # purple
    ( 70, 240, 240),  # cyan
    (240,  50, 230),  # magenta
    (210, 245,  60),  # lime
    (250, 190, 212),  # pink
    (  0, 128, 128),  # teal
    (220, 190, 255),  # lavender
    (170, 110,  40),  # brown
    (128,   0,   0),  # maroon
    (170, 255, 195),  # mint
    (128, 128,   0),  # olive
    (255, 215, 180),  # apricot
    (  0,   0, 128),  # navy
    ( 64, 224, 208),  # turquoise
    (255, 200,   0),  # gold
    ( 75,   0, 130),  # indigo
    (250, 128, 114),  # salmon
    (135, 206, 235),  # sky blue
    (220,  20,  60),  # crimson
    (127, 255,   0),  # chartreuse
    (218, 112, 214),  # orchid
    ( 70, 130, 180),  # steel blue
    (255,  99,  71),  # tomato
]
_DISTINCT_PALETTE_BGR = [(b, g, r) for (r, g, b) in _DISTINCT_PALETTE_RGB]

# Fallback pitch boundary, in pixel space (CCW winding), for the ORIGINAL
# tuned clip (121364_0.mp4) — used only when compute_pitch_verts_from_tracks
# can't derive a shape confidently (too few player detections). The real,
# per-run polygon is now DERIVED from each clip's own observed player
# positions (see compute_pitch_verts_from_tracks in
# tactical_events.space_control) rather than this fixed constant: a
# hardcoded polygon tuned to one camera's framing silently clipped a
# DIFFERENT clip's visible pitch (confirmed: 16.2% of real player
# detections, bbox tops as high as y=81, fell above this polygon's y=260
# top edge on a wider/higher broadcast angle — not a calibration
# limitation, since homography was comparably reliable there).
_FALLBACK_PITCH_VERTS = np.array([
    [-140, 1035],
    [  15,  275],
    [1830,  260],
    [1980, 1035],
], dtype=np.float32)

# Goalkeeper colour: fixed and team-3-dedicated, NOT drawn from the
# per-player distinct palette. Unlike outfield players (where several
# players share a team and need to be told apart), there is only ever one
# goalkeeper per side on the pitch, so a single fixed colour is sufficient
# to identify them — and, critically, it is independent of track/merged-
# identity id. If a goalkeeper's track fragments and proximity-linking
# fails to re-merge it into the same sid, a palette-based colour would
# change; this fixed colour can't, since it never depends on sid at all.
# Same cyan/sky-blue BGR(255, 220, 0) team=3 uses everywhere else in the
# pipeline (team_assigner, render_output2, render_output6).
GK_COLOR = (255, 220, 0)

# Player colour cache: pid → BGR tuple — stable across frames, computed once
_color_cache: dict = {}
_colors_assigned = False


def _palette_color(index: int) -> tuple:
    """BGR colour for palette position `index`. The first len(palette)
    indices get the curated hand-picked colours directly. Beyond that,
    the palette repeats with saturation/value varied (darker on odd
    cycles, lighter on even ones, desaturating slightly each cycle) so
    extra players still get a colour, degrading gracefully instead of
    producing an exact duplicate."""
    base_len = len(_DISTINCT_PALETTE_BGR)
    cycle = index // base_len
    slot  = index % base_len
    b, g, r = _DISTINCT_PALETTE_BGR[slot]
    if cycle == 0:
        return (b, g, r)
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    s = max(0.35, s - 0.15 * cycle)
    v = (v - 0.20 * cycle) if cycle % 2 == 1 else min(1.0, v + 0.12 * cycle)
    v = max(0.30, min(1.0, v))
    rr, gg, bb = colorsys.hsv_to_rgb(h, s, v)
    return (int(round(bb * 255)), int(round(gg * 255)), int(round(rr * 255)))


def _assign_player_colors(all_players: dict) -> None:
    """Assign every OUTFIELD player (teams 1/2, sorted by merged identity)
    a colour from the curated maximally-distinct palette, in order.
    Goalkeepers (team 3) are excluded here — they never draw from the
    palette, so they don't consume a slot or shift outfield indices; see
    GK_COLOR / _player_color. Computed once and cached."""
    global _colors_assigned
    ids = sorted(sid for sid, team in all_players.items() if team != 3)
    for idx, pid in enumerate(ids):
        _color_cache[int(pid)] = _palette_color(idx)
    _colors_assigned = True


def _player_color(pid: int, team: int) -> tuple:
    """Stable BGR colour for this player. Goalkeepers (team 3) always get
    the fixed GK_COLOR, independent of pid/sid — so a goalkeeper's colour
    can't change even if its track id fragments and identity-linking fails
    to re-merge it. Outfield players (team 1/2) get their pre-computed
    palette colour from the cache."""
    if team == 3:
        return GK_COLOR
    key = int(pid)
    if key not in _color_cache:
        # Player seen for the first time after the initial assignment pass
        # (shouldn't normally happen since we pre-scan all frames).
        _color_cache[key] = _palette_color(0)
    return _color_cache[key]


def _link_identities(tracks, n):
    """Merge fragmented tracker IDs into stable per-player identities, using
    the same proximity-based re-matching as render_output2.py's stamina
    panel: a new pid appearing within PROXIMITY_PX of a same-team pid lost
    within the last LOST_BUF_FRAMES frames is treated as the same real
    player and mapped to that player's existing merged identity (sid).

    Returns
    -------
    pid_to_sid       : dict  raw tracker pid  -> merged identity id
    sid_team         : dict  merged identity id -> team (1 or 2)
    appearance_count : dict  merged identity id -> total frames it appeared in
    """
    pid_to_sid       = {}
    tracker_to_sid   = {}
    sid_last_center  = {}
    sid_team         = {}
    recently_lost    = {}   # sid -> (cx, cy, team, frame_lost)
    appearance_count = {}
    next_sid = 1

    for frame_num in range(n):
        player_data = (tracks['players'][frame_num]
                       if frame_num < len(tracks['players']) else {})
        current_pids = {pid for pid, info in player_data.items()
                        if info.get('team', 0) in (1, 2, 3)}
        prev_pids    = set(tracker_to_sid.keys())
        appeared     = current_pids - prev_pids
        disappeared  = prev_pids    - current_pids

        for pid in disappeared:
            sid = tracker_to_sid.pop(pid, None)
            if sid is None:
                continue
            cx, cy = sid_last_center.get(sid, (0.0, 0.0))
            recently_lost[sid] = (cx, cy, sid_team.get(sid, 0), frame_num)

        stale = [s for s, (_, _, _, fn) in recently_lost.items()
                 if frame_num - fn > LOST_BUF_FRAMES]
        for s in stale:
            del recently_lost[s]

        matched_sids = set()
        for pid in sorted(appeared):
            info = player_data[pid]
            team = info.get('team', 0)
            bbox = info.get('bbox')
            if bbox is None:
                sid = next_sid; next_sid += 1
                tracker_to_sid[pid] = sid
                sid_team[sid] = team
                pid_to_sid[pid] = sid
                continue

            ncx = (bbox[0] + bbox[2]) / 2.0
            ncy = (bbox[1] + bbox[3]) / 2.0

            best_sid, best_d = None, PROXIMITY_PX + 1
            for sid, (cx, cy, lost_team, _) in recently_lost.items():
                if sid in matched_sids or lost_team != team:
                    continue
                d = np.sqrt((ncx - cx) ** 2 + (ncy - cy) ** 2)
                if d < best_d:
                    best_d, best_sid = d, sid

            if best_sid is not None:
                tracker_to_sid[pid] = best_sid
                matched_sids.add(best_sid)
                del recently_lost[best_sid]
                pid_to_sid[pid] = best_sid
            else:
                sid = next_sid; next_sid += 1
                tracker_to_sid[pid] = sid
                sid_team[sid] = team
                pid_to_sid[pid] = sid

        for pid in current_pids:
            if pid not in tracker_to_sid:
                sid = next_sid; next_sid += 1
                tracker_to_sid[pid] = sid
                sid_team[sid] = player_data[pid].get('team', 0)
                pid_to_sid[pid] = sid

        for pid in current_pids:
            sid = tracker_to_sid[pid]
            appearance_count[sid] = appearance_count.get(sid, 0) + 1
            bbox = player_data[pid].get('bbox')
            if bbox:
                sid_last_center[sid] = ((bbox[0] + bbox[2]) / 2.0,
                                        (bbox[1] + bbox[3]) / 2.0)

    return pid_to_sid, sid_team, appearance_count


def render_output3(video_frames, tracks, team_ball_control, transitions, fps,
                   homography_per_frame=None):
    if not video_frames:
        print("render_output3: no video frames given, skipping.")
        return
    cam_frames = video_frames
    orig_h, orig_w = cam_frames[0].shape[:2]
    print(f"output3: using {len(cam_frames)} raw source frames ({orig_w}x{orig_h})")

    n = min(len(tracks['players']), len(cam_frames))

    # ── Identity linking: merge fragmented tracker IDs into stable players ────
    pid_to_sid, sid_team, sid_appearance_count = _link_identities(tracks, n)

    raw_pid_count         = len(pid_to_sid)
    merged_identity_count = len(sid_team)
    linked_fragment_count = raw_pid_count - merged_identity_count

    valid_sids = {sid for sid, cnt in sid_appearance_count.items()
                 if cnt >= MIN_APPEARANCE_FRAMES}
    dropped = merged_identity_count - len(valid_sids)
    print(f"output3: {raw_pid_count} raw tracker IDs -> {merged_identity_count} merged "
          f"identities ({linked_fragment_count} fragment(s) linked back to an earlier "
          f"player); {len(valid_sids)} kept (>= {MIN_APPEARANCE_FRAMES} frames total), "
          f"{dropped} dropped as unlinked noise")

    all_sids = {sid: team for sid, team in sid_team.items() if sid in valid_sids}
    _assign_player_colors(all_sids)

    # ── Pitch mask + sampling grid (shared with tactical_events.space_control) ─
    pitch_verts = compute_pitch_verts_from_tracks(
        tracks, orig_h, orig_w, fallback_verts=_FALLBACK_PITCH_VERTS)
    pitch_mask = build_pitch_mask(pitch_verts, orig_h, orig_w)
    nz = int(np.count_nonzero(pitch_mask))
    print(f"output3: pitch mask non-zero pixels: {nz} / {orig_h * orig_w}"
          f"  ({'OK' if nz > 0 else 'EMPTY — fillPoly failed!'})")
    mask3 = np.stack([pitch_mask] * 3, axis=-1) > 0

    grid, grid_rows, grid_cols, row_idx, col_idx = build_sampling_grid(pitch_mask, STEP)
    print(f"output3: grid {grid_rows}x{grid_cols}, {len(grid)} cells inside pitch")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out    = cv2.VideoWriter('output_videos/output3.avi', fourcc, fps,
                             (orig_w, orig_h))

    for frame_num in tqdm(range(n), desc="Rendering output3 (per-player control)"):
        player_data = (tracks['players'][frame_num]
                       if frame_num < len(tracks['players']) else {})
        frame = cam_frames[frame_num].copy()

        # ── Collect valid player positions ────────────────────────────────────
        pids  = []
        pos   = []
        teams = {}
        sids  = {}   # pid -> merged identity id, for colour lookup
        for pid, info in player_data.items():
            sid = pid_to_sid.get(pid)
            if sid is None or sid not in valid_sids:
                continue   # unlinked / too-short tracker ID-fragmentation noise
            team = info.get('team', 0)
            if team not in (1, 2, 3):
                continue
            adj = info.get('position_adjusted')
            if adj is None:
                continue
            try:
                x, y = float(adj[0]), float(adj[1])
            except Exception:
                continue
            if x == 0.0 and y == 0.0:
                continue
            pids.append(pid)
            pos.append((x, y))
            teams[pid] = team
            sids[pid]  = sid

        if pids:
            pts = np.array(pos, dtype=np.float32)   # (P, 2)

            # Nearest-player-per-grid-cell — shared algorithm (see module
            # docstring): same function tactical_events.space_control uses
            # for the SPACE event, called here with output3's own merged
            # per-player identities instead of raw tracker pids.
            nearest_idx, _ = nearest_player_per_cell(grid, pts)

            # Build (P, 3) colour array (by merged identity) and index into it
            colors      = np.array(
                [_player_color(sids[pids[i]], teams[pids[i]]) for i in range(len(pids))],
                dtype=np.uint8,
            )
            cell_colors = colors[nearest_idx]   # (M, 3)

            # Fill the small grid image, upsample, blur
            grid_img = np.zeros((grid_rows, grid_cols, 3), dtype=np.uint8)
            grid_img[row_idx, col_idx] = cell_colors

            heatmap = cv2.resize(grid_img, (orig_w, orig_h),
                                 interpolation=cv2.INTER_NEAREST)
            heatmap = cv2.GaussianBlur(heatmap, (BLUR_K, BLUR_K), BLUR_S)
            heatmap[~mask3] = 0

            # Blend 35% over video frame inside pitch area
            blended = frame.astype(np.float32)
            blended[mask3] = np.clip(
                frame[mask3].astype(np.float32) * (1.0 - BLEND_A)
                + heatmap[mask3].astype(np.float32) * BLEND_A,
                0, 255,
            )
            frame = blended.astype(np.uint8)

            # ── Per-player markers: same HSV colour as the zone overlay ────────
            # Coloured by merged identity (sid) so a player keeps one colour
            # through ID switches; the label still shows the raw tracker pid.
            for pid, (x, y) in zip(pids, pos):
                cx, cy = int(round(x)), int(round(y))
                color  = _player_color(sids[pid], teams[pid])
                cv2.circle(frame, (cx, cy), MARKER_RADIUS, color, -1, cv2.LINE_AA)
                cv2.putText(frame, str(int(pid)), (cx + MARKER_RADIUS + 3, cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, MARKER_TEXT_COLOR, 1,
                            cv2.LINE_AA)

        # ── Timestamp label ───────────────────────────────────────────────────
        t_sec = frame_num / max(fps, 1)
        label = f"PLAYER CONTROL  {int(t_sec // 60):02d}:{int(t_sec % 60):02d}"
        for thick, col in ((3, (0, 0, 0)), (1, (220, 220, 220))):
            cv2.putText(frame, label, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, thick)

        if frame_num % 50 == 0:
            print(f"output3: frame {frame_num}/{n}  players={len(pids)}")

        out.write(frame)

    out.release()
    path    = 'output_videos/output3.avi'
    size_mb = os.path.getsize(path) / 1e6
    print(f"output3.avi saved ({n} frames, {size_mb:.1f} MB).")
