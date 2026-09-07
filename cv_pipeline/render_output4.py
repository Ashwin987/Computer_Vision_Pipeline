"""
render_output4.py - Pitch control heatmap (pixel-space only).

Root cause of invisible heatmap (pre-fix):
  A sparse 1-pixel-per-8px grid was blurred with a 61-wide Gaussian.
  Only ~1/64 of pixels were coloured, so the Gaussian average diluted
  each channel value from ~220 down to ~3 (220/64) — completely
  invisible.  Additionally MIN_TEAM_PLAYERS=3 silently skipped frames
  with fewer detections.

Fix:
  1. Build a small (ceil(h/8) x ceil(w/8)) classification grid — each
     cell assigned to the team whose nearest player is closest.
  2. Upsample with INTER_NEAREST to full resolution, giving solid 8x8
     colour blocks at full intensity (no dilution before blur).
  3. GaussianBlur(61, sigma=20) smooths zone boundaries.
  MIN_TEAM_PLAYERS lowered to 1 so frames with few detections render.

Team 1: red  (RGB 230,25,75)  35% blend
Team 2: blue (RGB 0,130,200)  35% blend

Saves: output_videos/output4.avi
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

BLEND_A          = 0.35
BLUR_K           = 61
BLUR_S           = 20.0
STEP             = 8       # classify every Nth pixel in both axes
MIN_TEAM_PLAYERS = 1       # was 3 — caused silent skip on low-detection frames

# Fixed, deliberately high-contrast fill colors — NOT the per-clip fitted
# kit color (tracks['players'][...]['team_color'], the field
# tracker.draw_annotations() uses for the ellipse markers elsewhere).
# This used to prefer the real fitted color when available, on the theory
# that "authentic" colors were nicer — but real kit colors aren't chosen
# for heatmap contrast, and it backfired concretely on a real clip
# (BarcaMadridPT1): Team 1's fitted color was dark maroon (58,81,101 BGR)
# and Team 2's was pale cream (149,195,180 BGR) — blended at 35% opacity
# over a saturated green pitch, BOTH wash toward "a shade of green" and
# become hard to tell apart at a glance, exactly backwards from this
# output's whole purpose (an instantly-readable territorial-control map).
#
# Both fixed colors are chosen with a LOW green channel specifically:
# the pitch itself is high-G (grass), so blending ANY color that still
# carries moderate G at 35% opacity partially dilutes back toward green —
# confirmed concretely on a first attempt at this fix (soft blue,
# BGR 200,130,0) that read as red-vs-green rather than red-vs-blue once
# actually blended, since its G=130 wasn't low enough to resist the
# pitch's own green. Pushing both team colors' G channel near zero avoids
# that regardless of blend ratio: Team 1 (red, dominant R) and Team 2
# (blue, dominant B) are opposite in the one channel each has to spare
# (R vs B) while sharing near-zero G, so both stay clearly distinguishable
# from the pitch AND from each other.
T1_COLOR = ( 30,  30, 220)   # BGR red  (RGB 220,30,30)
T2_COLOR = (220,  30,  30)   # BGR blue (RGB 30,30,220)

# Full-frame pitch boundary — matches ViewTransformer.pixel_vertices
_PIXEL_VERTS_1080 = np.array(
    [[0, 1080], [0, 0], [1920, 0], [1920, 1080]], dtype=np.float32)


def _build_pitch_mask(h, w):
    """uint8 mask (h, w): 255 inside pitch boundary, 0 outside."""
    verts = _PIXEL_VERTS_1080.copy()
    verts[:, 0] *= w / 1920.0
    verts[:, 1] *= h / 1080.0
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [verts.astype(np.int32)], 255)
    return mask


def render_output4(video_frames, tracks, team_ball_control, view_transformer,
                   fps, annotated_frames, homography_per_frame=None):
    if not annotated_frames:
        print("render_output4: no frames, skipping.")
        return

    orig_h, orig_w = annotated_frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out    = cv2.VideoWriter('output_videos/output4.avi', fourcc, fps,
                             (orig_w, orig_h))

    n = len(annotated_frames)

    # ── Pitch mask ────────────────────────────────────────────────────────────
    pitch_mask = _build_pitch_mask(orig_h, orig_w)
    nz_mask = int(np.count_nonzero(pitch_mask))
    print(f"[debug output4] pitch_mask non-zero pixels: {nz_mask} / {orig_h * orig_w}"
          f"  ({'OK' if nz_mask > 0 else 'EMPTY — fillPoly failed!'})")
    mask3 = np.stack([pitch_mask] * 3, axis=-1) > 0   # (h,w,3) bool

    # ── Sampling grid (constant across frames) ────────────────────────────────
    # Classify at every STEP-th pixel.  Store as a small grid_rows x grid_cols
    # image; upsampling to full res avoids the colour-dilution bug.
    grid_rows = (orig_h + STEP - 1) // STEP
    grid_cols = (orig_w + STEP - 1) // STEP

    # Pixel coordinates of each grid cell's top-left corner
    gy_all = (np.arange(grid_rows) * STEP).astype(np.float32)   # (R,)
    gx_all = (np.arange(grid_cols) * STEP).astype(np.float32)   # (C,)
    gx_2d, gy_2d = np.meshgrid(gx_all, gy_all)                  # (R,C)
    gx_flat = gx_2d.ravel()
    gy_flat = gy_2d.ravel()

    # Keep only cells whose pixel falls inside the pitch mask
    yi = np.clip(gy_flat.astype(np.int32), 0, orig_h - 1)
    xi = np.clip(gx_flat.astype(np.int32), 0, orig_w - 1)
    inside  = pitch_mask[yi, xi] > 0
    gx_in   = gx_flat[inside]
    gy_in   = gy_flat[inside]
    row_idx = np.clip((gy_in / STEP).astype(np.int32), 0, grid_rows - 1)
    col_idx = np.clip((gx_in / STEP).astype(np.int32), 0, grid_cols - 1)
    grid    = np.stack([gx_in, gy_in], axis=1)   # (M, 2) float32

    print(f"[debug output4] sampling grid: {grid_rows}x{grid_cols} cells, "
          f"{len(grid)} cells inside pitch mask")

    for frame_num in tqdm(range(n), desc="Rendering output4 (pitch control)"):
        player_data = (tracks['players'][frame_num]
                       if frame_num < len(tracks['players']) else {})

        frame = annotated_frames[frame_num].copy()

        # ── Collect position_adjusted per team ───────────────────────────────
        t1_pos, t2_pos = [], []
        for pid, info in player_data.items():
            pos  = info.get('position_adjusted')
            team = info.get('team', 0)
            if pos is None or team not in (1, 2):
                continue
            try:
                x, y = float(pos[0]), float(pos[1])
            except Exception:
                continue
            (t1_pos if team == 1 else t2_pos).append((x, y))

        if frame_num < 5:
            print(f"[debug output4] frame {frame_num}: "
                  f"T1={len(t1_pos)} players, T2={len(t2_pos)} players "
                  f"with valid position_adjusted")

        t1_pct = t2_pct = 0.0

        if len(t1_pos) >= MIN_TEAM_PLAYERS and len(t2_pos) >= MIN_TEAM_PLAYERS:
            t1 = np.array(t1_pos, dtype=np.float32)   # (n1, 2)
            t2 = np.array(t2_pos, dtype=np.float32)   # (n2, 2)

            # Vectorised Voronoi: min distance per grid cell to each team
            d1 = np.sqrt(
                ((grid[:, None, :] - t1[None, :, :]) ** 2).sum(-1)
            ).min(-1)   # (M,)
            d2 = np.sqrt(
                ((grid[:, None, :] - t2[None, :, :]) ** 2).sum(-1)
            ).min(-1)   # (M,)

            t1_ctrl = d1 <= d2   # (M,) bool

            total  = len(t1_ctrl)
            t1_pct = float(t1_ctrl.sum()) / total * 100.0 if total > 0 else 0.0
            t2_pct = 100.0 - t1_pct

            # ── Build solid-colour grid image ─────────────────────────────────
            # Each cell painted at full colour intensity — no dilution from
            # sparse pixels before blur.
            grid_img = np.zeros((grid_rows, grid_cols, 3), dtype=np.uint8)
            grid_img[row_idx[t1_ctrl],  col_idx[t1_ctrl]]  = T1_COLOR
            grid_img[row_idx[~t1_ctrl], col_idx[~t1_ctrl]] = T2_COLOR

            # ── Upsample to full frame (nearest-neighbour = solid blocks) ─────
            heatmap = cv2.resize(grid_img, (orig_w, orig_h),
                                 interpolation=cv2.INTER_NEAREST)

            if frame_num < 5:
                nz_hm = int(np.count_nonzero(heatmap))
                print(f"[debug output4] frame {frame_num}: "
                      f"heatmap non-zero pixels after upsample (before blur): {nz_hm}")

            # ── Blur zone boundaries ──────────────────────────────────────────
            heatmap = cv2.GaussianBlur(heatmap, (BLUR_K, BLUR_K), BLUR_S)

            # Restrict to pitch area
            heatmap[~mask3] = 0

            # ── Blend 35% over video frame ────────────────────────────────────
            blended      = frame.astype(np.float32)
            blended[mask3] = np.clip(
                frame[mask3].astype(np.float32) * (1.0 - BLEND_A)
                + heatmap[mask3].astype(np.float32) * BLEND_A,
                0, 255
            )
            frame = blended.astype(np.uint8)

            if frame_num < 5:
                print(f"[debug output4] frame {frame_num}: "
                      f"cv2.addWeighted-equivalent blend applied "
                      f"(T1={t1_pct:.1f}%, T2={t2_pct:.1f}%)")

        # ── Labels ────────────────────────────────────────────────────────────
        ctrl_str = f"Team 1: {t1_pct:.0f}% | Team 2: {t2_pct:.0f}%"
        (tw, _), _ = cv2.getTextSize(ctrl_str, cv2.FONT_HERSHEY_SIMPLEX,
                                     0.45, 1)
        bx = orig_w - tw - 12
        by = orig_h - 10
        cv2.rectangle(frame, (bx - 4, by - 18), (orig_w - 6, by + 4),
                      (0, 0, 0), -1)
        cv2.putText(frame, ctrl_str, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Header bar
        cv2.rectangle(frame, (0, 0), (orig_w, 28), (0, 0, 0), -1)
        cv2.putText(frame, "PITCH CONTROL", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
        t_sec = frame_num / max(fps, 1)
        ts    = f"{int(t_sec // 60):02d}:{int(t_sec % 60):02d}"
        ts_w  = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        cv2.putText(frame, ts, (orig_w - ts_w - 8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        if frame_num % 50 == 0:
            print(f"output4: frame {frame_num}/{n}  "
                  f"T1={t1_pct:.0f}%  T2={t2_pct:.0f}%")

        out.write(frame)

    out.release()
    path    = 'output_videos/output4.avi'
    size_mb = os.path.getsize(path) / 1e6
    print(f"output4.avi saved ({n} frames, {size_mb:.1f} MB).")
