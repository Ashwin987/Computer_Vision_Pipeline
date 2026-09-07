"""
tactical_events/space_control.py — shared per-frame nearest-player grid
computation ("lightweight Voronoi"): a coarse pixel-space grid inside the
pitch clip polygon, classified every frame by nearest player.

This is the exact algorithm render_output3.py used to build its zone
heatmap, factored out so it is written and run in ONE place. Two
consumers use it:
  - main.py's one-time tactical-events precompute step
    (compute_space_control_per_frame, keyed by raw tracker pid — every
    player who has a valid position that frame, no identity merging)
  - render_output3.py's zone overlay (build_pitch_mask / build_sampling_grid
    / nearest_player_per_cell directly, keyed by ITS OWN merged per-player
    identity, since that module also has to solve tracker ID-fragmentation
    for stable colouring — a different player-identity space, so it calls
    the same functions with its own player list rather than re-deriving
    the grid/distance math from scratch)
"""

import cv2
import numpy as np


def build_pitch_mask(pitch_verts, h, w, ref_w=1920.0, ref_h=1080.0):
    """uint8 mask (h, w): 255 inside the pitch boundary, 0 outside.
    `pitch_verts` is defined in a (ref_w x ref_h) reference pixel space
    and scaled to the actual (h, w) frame size."""
    verts = pitch_verts.copy()
    verts[:, 0] *= w / ref_w
    verts[:, 1] *= h / ref_h
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [verts.astype(np.int32)], 255)
    return mask


def build_sampling_grid(pitch_mask, step):
    """Build the coarse grid of cell centres inside `pitch_mask`, sampling
    every `step`-th pixel in both axes.

    Returns
    -------
    grid                 : (M, 2) float32 array of (x, y) cell centres
    grid_rows, grid_cols : int — size of the small grid image
    row_idx, col_idx     : (M,) int arrays — where each cell lands in the
                            small (grid_rows, grid_cols) image, for
                            reconstructing/upsampling a per-cell image
    """
    h, w = pitch_mask.shape
    grid_rows = (h + step - 1) // step
    grid_cols = (w + step - 1) // step
    gy_all = (np.arange(grid_rows) * step).astype(np.float32)
    gx_all = (np.arange(grid_cols) * step).astype(np.float32)
    gx_2d, gy_2d = np.meshgrid(gx_all, gy_all)
    gx_flat = gx_2d.ravel()
    gy_flat = gy_2d.ravel()
    yi_g = np.clip(gy_flat.astype(np.int32), 0, h - 1)
    xi_g = np.clip(gx_flat.astype(np.int32), 0, w - 1)
    inside = pitch_mask[yi_g, xi_g] > 0
    gx_in = gx_flat[inside]
    gy_in = gy_flat[inside]
    row_idx = np.clip((gy_in / step).astype(np.int32), 0, grid_rows - 1)
    col_idx = np.clip((gx_in / step).astype(np.int32), 0, grid_cols - 1)
    grid = np.stack([gx_in, gy_in], axis=1)
    return grid, grid_rows, grid_cols, row_idx, col_idx


def nearest_player_per_cell(grid, positions):
    """Vectorised nearest-player-per-cell ("lightweight Voronoi").

    Parameters
    ----------
    grid      : (M, 2) float32 — grid-cell centres
    positions : (P, 2) float32 — player positions, SAME pixel space as grid

    Returns
    -------
    nearest_idx : (M,) int — index into `positions` of the nearest player
                  for each grid cell
    cell_counts : (P,) int — how many grid cells each player controls
    """
    dists = np.sqrt(((grid[:, None, :] - positions[None, :, :]) ** 2).sum(-1))
    nearest_idx = dists.argmin(axis=1)
    cell_counts = np.bincount(nearest_idx, minlength=positions.shape[0])
    return nearest_idx, cell_counts


REF_W, REF_H = 1920.0, 1080.0   # matches build_pitch_mask's own default ref_w/ref_h


def compute_pitch_verts_from_tracks(tracks, frame_h, frame_w,
                                    margin_frac=0.08, min_samples=200,
                                    fallback_verts=None):
    """Derive a perspective-correct pitch-boundary quadrilateral from the
    ACTUAL observed extent of player detections in this clip, instead of a
    fixed pixel polygon tuned to one specific camera framing. A hardcoded
    polygon (the previous approach) silently clips whatever region of the
    pitch a *different* clip's camera happens to frame differently —
    confirmed on a wider/higher broadcast angle where 16.2% of real,
    tracked player detections (bbox tops as high as y=81) fell above a
    polygon whose top edge sat at y=260-275, leaving the top ~20% of the
    visible pitch with no space-control modeling at all, even though
    calibration/homography was comparably reliable there (87.3% in-bounds)
    to the modeled region (90.4%) — so the gap wasn't a calibration
    limitation, it was this fixed shape.

    Splits detections by whether their foot position falls in the frame's
    upper or lower half (the far/near touchline sides of a typical
    broadcast angle) and uses each half's own observed x-extent, so the
    camera's natural perspective narrowing toward the far touchline is
    preserved rather than assumed from one specific camera's numbers.
    Expanded by margin_frac of each half's own observed width/height so
    areas players didn't quite reach in this particular window (exact
    touchlines/corners) still get modeled — modest overshoot past the
    touchline is preferred over leaving visible pitch unmodeled.

    Parameters
    ----------
    tracks       : dict — needs tracks['players'], a list of per-frame
                   {pid: {'bbox': [...], 'team': int, ...}} dicts.
    frame_h, frame_w : int — native size of THIS clip's frames (used only
                   to interpret the raw bbox pixel coordinates while
                   deriving the shape).
    margin_frac  : float — fractional padding added to each observed
                   extent (both halves' x-range and the overall y-range).
                   0.08 is a modest, clip-independent safety buffer, not a
                   value tuned to any one video: any window of play will
                   under-sample the true touchline/corner extent somewhat
                   (players rarely stand exactly on the line), so a small
                   fixed-fraction pad generalizes better than trying to
                   assume the sampled extent already IS the true boundary.
    min_samples  : int — minimum qualifying detections required to derive
                   a shape with any confidence; below this, falls back to
                   fallback_verts (or the whole frame if that's also None)
                   rather than fitting a polygon to too little data.
    fallback_verts : (4,2) array-like or None — used when there's too
                   little data to derive confidently. Expected already in
                   the REF_W x REF_H reference space (same convention as
                   the returned polygon), since that's what a caller's own
                   pre-existing fallback constant would already be in. If
                   None, falls back to the whole frame (over-inclusive,
                   but never excludes real pitch — consistent with
                   preferring overshoot to a blank region).

    Returns
    -------
    (4, 2) float32 ndarray, CCW winding, in the SAME REF_W x REF_H
    (1920x1080) reference pixel space build_pitch_mask expects by
    default: [bottom-left, top-left, top-right, bottom-right]. Callers
    don't need to change how they call build_pitch_mask(verts, h, w) —
    this matches exactly what the old fixed PITCH_VERTS constant provided.
    """
    xs_upper, xs_lower = [], []
    y_min, y_max = frame_h, 0
    mid_y = frame_h / 2.0
    n_samples = 0
    for frame_data in tracks['players']:
        for pid, info in frame_data.items():
            bbox = info.get('bbox')
            if bbox is None or info.get('team', 0) not in (1, 2, 3):
                continue
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            foot_y = y2   # bottom of bbox = feet, the actual pitch-contact point
            n_samples += 1
            y_min = min(y_min, foot_y)
            y_max = max(y_max, foot_y)
            (xs_upper if foot_y < mid_y else xs_lower).append(cx)

    if n_samples < min_samples or not xs_upper or not xs_lower:
        if fallback_verts is not None:
            return np.array(fallback_verts, dtype=np.float32)
        return np.array([[0, REF_H], [0, 0], [REF_W, 0], [REF_W, REF_H]],
                        dtype=np.float32)

    def _expand(vals, lo_bound, hi_bound):
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * margin_frac
        return max(lo_bound, lo - pad), min(hi_bound, hi + pad)

    upper_lo, upper_hi = _expand(xs_upper, 0.0, frame_w)
    lower_lo, lower_hi = _expand(xs_lower, 0.0, frame_w)
    y_pad = (y_max - y_min) * margin_frac
    top_y = max(0.0, y_min - y_pad)
    bottom_y = min(frame_h, y_max + y_pad)

    verts = np.array([
        [lower_lo, bottom_y],
        [upper_lo, top_y],
        [upper_hi, top_y],
        [lower_hi, bottom_y],
    ], dtype=np.float32)
    # Rescale from this clip's native pixel space into the REF_W x REF_H
    # reference space, matching what a fixed PITCH_VERTS constant already
    # represented — so build_pitch_mask(verts, h, w) keeps working
    # unchanged at every call site regardless of which source produced verts.
    verts[:, 0] *= REF_W / frame_w
    verts[:, 1] *= REF_H / frame_h
    return verts


def compute_space_control_per_frame(tracks, grid, top_frac=0.15):
    """Run the shared nearest-player grid computation ONCE for every frame,
    keyed by RAW tracker pid, using position_adjusted (pixel space, the
    same space `grid` was built in). Called once from main.py; both the
    SPACE tactical event and (indirectly, via the shared functions above)
    render_output3.py's zone overlay derive from this same algorithm.

    Returns
    -------
    list, one dict per frame:  pid -> {
        'cells': int   cell count this frame,
        'frac' : float fraction of in-pitch grid cells controlled,
        'rank' : int   1-based rank this frame (1 = most space controlled),
        'top10': bool  True if this player is in the top `top_frac` by rank
    }
    Frames with fewer than 2 valid players get an empty dict (no rank is
    meaningful with 0-1 players on the grid).
    """
    total_cells = len(grid)
    per_frame = []

    for player_data in tracks['players']:
        pids, pos = [], []
        for pid, info in player_data.items():
            if info.get('team', 0) not in (1, 2):
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

        if len(pids) < 2 or total_cells == 0:
            per_frame.append({})
            continue

        positions = np.array(pos, dtype=np.float32)
        _, cell_counts = nearest_player_per_cell(grid, positions)

        order = np.argsort(-cell_counts)   # descending cell count
        n_top = max(1, int(np.ceil(top_frac * len(pids))))

        frame_result = {}
        for rank, idx in enumerate(order, start=1):
            pid = pids[idx]
            frame_result[pid] = {
                'cells': int(cell_counts[idx]),
                'frac':  float(cell_counts[idx]) / total_cells,
                'rank':  rank,
                'top10': rank <= n_top,
            }
        per_frame.append(frame_result)

    return per_frame
