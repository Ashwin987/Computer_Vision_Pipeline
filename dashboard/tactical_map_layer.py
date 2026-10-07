"""
tactical_map_layer.py - the reviewed "map layer" for the Tactical Map.

tactical_view.py draws every tracked box through the automatic per-frame
homography. This module replaces that data path, for matches that have the
reviewed map-layer files in their CV output folder, with:

  * the approved per-second homographies (tactical_map_homographies_v7.json,
    plus any single-frame replacement listed in HOMOGRAPHY_PATCHES), and
  * the reviewed per-box outcomes (tactical_map_players_v12.json, or the
    latest tactical_map_players_v*.json that exists).

Only boxes whose outcome is team1, team2 or goalkeeper are drawn. Every other
outcome (referee, conflict, overflow, duplicate, off-pitch, hidden, anything
this module does not recognise) is left off the map - an allow-list, so a new
outcome name in a later file is never drawn by accident, and there is no grey
"unassigned" dot at all.

The homographies exist only for the sampled seconds, so the map is a
one-picture-per-second sequence: each sampled second is drawn once and held
until the next. Nothing is interpolated between seconds and no homography is
carried from one second to another.

Both files are re-read from disk on every call and nothing is cached in this
module, so swapping in a new players file needs no code change; app.py's
cache key includes layer_content_sig, so the cached video rebuilds itself.

Borrows tactical_view only for the pitch background and its drawing helpers.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import cv2

import tactical_view as tv

HOMOGRAPHY_FILE = "tactical_map_homographies_v7.json"

# Approved single-frame replacements on top of HOMOGRAPHY_FILE:
# {match: {frame: file to take that frame's homography from}}. Barcelona
# t=4s (frame 100) is the one v7 'disagreement' frame re-solved in v8.
HOMOGRAPHY_PATCHES = {
    "barca_madrid_pt1": {100: "tactical_map_homographies_v8.json"},
}

PLAYERS_PREFERRED_FILE = "tactical_map_players_v12.json"
_PLAYERS_FILE_RE = re.compile(r"^tactical_map_players_v(\d+)\.json$")

DRAWN_OUTCOMES = ("team1", "team2", "goalkeeper")
GOALKEEPER_BGR = (0, 212, 255)  # #FFD400


class MapLayerError(Exception):
    """The map-layer files are present but cannot be used as they are."""


def _players_file(cv_output_dir):
    """PLAYERS_PREFERRED_FILE if it exists, else the highest-numbered
    tactical_map_players_v*.json in the folder, else None."""
    folder = Path(cv_output_dir)
    preferred = folder / PLAYERS_PREFERRED_FILE
    if preferred.exists():
        return preferred
    versions = []
    for p in folder.glob("tactical_map_players_v*.json"):
        m = _PLAYERS_FILE_RE.match(p.name)
        if m:
            versions.append((int(m.group(1)), p))
    return max(versions)[1] if versions else None


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _layer_files(cv_output_dir):
    """(homography_path, {frame: patch_path}, players_path) - the exact files
    load_layer reads for this folder. A path is None when that file is
    missing; the patch dict only lists patches that apply to this match."""
    if not cv_output_dir:
        return None, {}, None
    folder = Path(cv_output_dir)
    homography_path = folder / HOMOGRAPHY_FILE
    if not homography_path.exists():
        return None, {}, _players_file(folder)
    try:
        match = _read_json(homography_path).get("match")
    except (OSError, ValueError, AttributeError):
        match = None
    patches = {frame: folder / fname for frame, fname in HOMOGRAPHY_PATCHES.get(match, {}).items()}
    return homography_path, patches, _players_file(folder)


def layer_content_sig(cv_output_dir):
    """Content hash of every map-layer file load_layer reads for this folder,
    or "" when the folder has no map layer at all. Hashes file names as well
    as bytes, so switching to a different players file changes the result
    even if the two files happened to hold the same bytes."""
    homography_path, patches, players_path = _layer_files(cv_output_dir)
    if homography_path is None and players_path is None:
        return ""
    h = hashlib.sha1()
    for p in [homography_path, *[patches[f] for f in sorted(patches)], players_path]:
        if p is not None and p.exists():
            h.update(p.name.encode() + b"\0" + p.read_bytes())
        else:
            h.update(b"MISSING:" + (p.name.encode() if p is not None else b"?"))
    return h.hexdigest()[:12]


def _homography_matrix(entry, source):
    H = (entry or {}).get("H")
    if H is None:
        return None
    H = np.array(H, dtype=float)
    if H.shape != (3, 3):
        raise MapLayerError(f"{source}: homography is not a 3x3 matrix")
    return H


def _boxes_by_frame(data, source):
    """{frame: [box, ...]} from the players file's 'boxes', which may be a
    {frame: [box]} mapping or a flat list of boxes that each carry 'frame'."""
    boxes = data.get("boxes") if isinstance(data, dict) else None
    if isinstance(boxes, dict):
        return {int(k): list(v or []) for k, v in boxes.items()}
    if isinstance(boxes, list):
        by_frame = {}
        for b in boxes:
            by_frame.setdefault(int(b["frame"]), []).append(b)
        return by_frame
    raise MapLayerError(f"{source} has no per-box outcomes (no 'boxes' section)")


def _box_outcome(box):
    """The reviewed outcome for one box - 'outcome', or 'slot' as the v12
    review files name the same field."""
    return box.get("outcome", box.get("slot"))


class MapLayer:
    """One match's reviewed map layer, already projected to pitch metres.

    frames: the sampled frame indices, in order (one per sampled second).
    dots:   {frame: [(outcome, X, Y), ...]} - drawable boxes only.
    stats:  {frame: {"boxes", "drawn", "team1", "team2", "goalkeeper",
            "not_drawn", "off_map", "has_homography"}}."""

    def __init__(self, match, frames, dots, stats, sources):
        self.match = match
        self.frames = frames
        self.dots = dots
        self.stats = stats
        self.sources = sources

    def total(self, key):
        return sum(s[key] for s in self.stats.values())

    def hold_frames(self, fps):
        """How many video frames each sampled frame is held for: up to the
        next sampled frame, and the usual gap (else one second) for the last."""
        gaps = [b - a for a, b in zip(self.frames, self.frames[1:])]
        last = int(np.median(gaps)) if gaps else int(round(fps))
        return [max(1, g) for g in gaps + [last]]


def load_layer(cv_output_dir):
    """This folder's MapLayer; None when it has no map-layer files at all
    (the caller keeps its existing render); MapLayerError when the files are
    there but incomplete or unreadable."""
    homography_path, patches, players_path = _layer_files(cv_output_dir)
    if homography_path is None and players_path is None:
        return None
    if homography_path is None:
        raise MapLayerError(f"{HOMOGRAPHY_FILE} is missing")
    if players_path is None:
        raise MapLayerError("no tactical_map_players_v*.json file found")
    try:
        homography_data = _read_json(homography_path)
        H_by_frame = {
            int(k): _homography_matrix(v, f"{homography_path.name} frame {k}")
            for k, v in homography_data["frames"].items()
        }
        for frame, patch_path in patches.items():
            if not patch_path.exists():
                raise MapLayerError(f"{patch_path.name} is missing (needed for frame {frame})")
            H = _homography_matrix(_read_json(patch_path)["frames"].get(str(frame)),
                                   f"{patch_path.name} frame {frame}")
            if H is None:
                raise MapLayerError(f"{patch_path.name} has no homography for frame {frame}")
            H_by_frame[frame] = H
        boxes_by_frame = _boxes_by_frame(_read_json(players_path), players_path.name)

        frames = sorted(H_by_frame)
        if not frames:
            raise MapLayerError(f"{homography_path.name} has no frames")
        dots, stats = {}, {}
        for frame in frames:
            H = H_by_frame[frame]
            boxes = boxes_by_frame.get(frame, [])
            frame_dots, off_map = [], 0
            for box in boxes:
                outcome = _box_outcome(box)
                if outcome not in DRAWN_OUTCOMES or H is None:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box["bbox"])
                out = H @ np.array([(x1 + x2) / 2.0, y2, 1.0])
                if abs(out[2]) < 1e-9:
                    off_map += 1
                    continue
                X, Y = out[0] / out[2], out[1] / out[2]
                if not tv._in_pitch_bounds(X, Y):
                    off_map += 1
                    continue
                frame_dots.append((outcome, float(X), float(Y)))
            n_reviewed_drawable = sum(1 for b in boxes if _box_outcome(b) in DRAWN_OUTCOMES)
            dots[frame] = frame_dots
            stats[frame] = {
                "boxes": len(boxes),
                "drawn": len(frame_dots),
                "team1": sum(1 for d in frame_dots if d[0] == "team1"),
                "team2": sum(1 for d in frame_dots if d[0] == "team2"),
                "goalkeeper": sum(1 for d in frame_dots if d[0] == "goalkeeper"),
                "not_drawn": len(boxes) - n_reviewed_drawable,
                "off_map": off_map,
                "has_homography": H is not None,
            }
    except MapLayerError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        raise MapLayerError(f"could not read the map-layer files: {type(e).__name__}: {e}")
    sources = [homography_path.name, *[patches[f].name for f in sorted(patches)], players_path.name]
    return MapLayer(homography_data.get("match"), frames, dots, stats, sources)


def render_frame(layer, frame, team_bgr):
    """One sampled frame of the map as a BGR canvas: a filled hull per team
    (outfield dots only - goalkeepers are their own outcome), then one dot
    per drawable box, goalkeepers in yellow."""
    canvas = tv._PITCH_BG.copy()
    frame_dots = layer.dots.get(frame, [])
    team_color = {"team1": team_bgr.get(1, (150, 150, 150)), "team2": team_bgr.get(2, (150, 150, 150))}

    for team in ("team1", "team2"):
        pts = [tv._pt(X, Y) for outcome, X, Y in frame_dots if outcome == team]
        if len(pts) >= 3:
            hull = cv2.convexHull(np.array(pts, dtype=np.int32))
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [hull], team_color[team])
            cv2.addWeighted(overlay, tv.HULL_FILL_ALPHA, canvas, 1 - tv.HULL_FILL_ALPHA, 0, dst=canvas)
            tv._draw_outlined_polyline(canvas, hull, team_color[team], thickness=2)

    for outcome, X, Y in frame_dots:
        cx, cy = tv._pt(X, Y)
        if outcome == "goalkeeper":
            cv2.circle(canvas, (cx, cy), 9, GOALKEEPER_BGR, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 9, (20, 20, 20), 1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (cx, cy), 8, team_color[outcome], -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 8, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def render_video(layer, team_bgr, out_path, fps=25.0):
    """Writes the whole map as an XVID .avi (same render-then-transcode
    convention as tactical_view.render_topdown_video - the caller transcodes
    it for the browser). Each sampled frame is drawn once and held until the
    next one, so the video runs at the clip's own speed."""
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (tv.CANVAS_W, tv.CANVAS_H))
    try:
        for frame, hold in zip(layer.frames, layer.hold_frames(fps)):
            canvas = render_frame(layer, frame, team_bgr)
            for _ in range(hold):
                writer.write(canvas)
    finally:
        writer.release()
    return out_path
