"""
render_output1.py — Tracking video with a precomputed TACTICAL EVENTS
carousel panel.

Events are position/movement-based (SPRINT, BURST, PRESS, RECOVERY,
OVERLAP, SPACE, LATERAL_RUN, DROP, BREAK, ISOLATED), detected once by
tactical_events.tactical_events_detector.TacticalEventsDetector and then
ranked once by tactical_events.event_ranking.rank_events_by_window (base
weight + intensity score, top 10 per 20-second window). This module does
NOT detect, rank, or recompute anything — it only looks up the current
20-second window's already-ranked batch and displays it.

Layout:
  [280px event panel] | [original video frame]

Panel: left side, dark semi-transparent row background, "TACTICAL EVENTS"
header, current window's ranked batch (highest score first), one line
each:
    "[MM:SS] EVENTNAME P<id>"
Event names are colour-coded per type. The whole batch is replaced
("carousel") when playback crosses into the next 20-second window.

Saves: output_videos/output1.avi
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

PANEL_W  = 280
PANEL_BG = (20, 20, 20)
HEADER_H = 40
ROW_H    = 28   # pixels per event row

EVENT_COLORS = {
    'SPRINT':   (0,   0, 230),   # red
    'BURST':    (0, 140, 255),   # orange
    'PRESS':    (0, 220, 220),   # yellow
    'RECOVERY': (230, 220,  0),  # cyan
    'OVERLAP':  (0, 200,   0),   # green
    'SPACE':    (255, 190, 120), # light blue
    'LATERAL_RUN': (200,  0, 160),  # purple
    'DROP':     (150, 140,  0),  # teal
    'BREAK':    (200,  0, 200),  # magenta
    'ISOLATED': (160, 160, 160), # grey
}


def _draw_panel(canvas, frame_num, fps, batch):
    """Render the 280px tactical-events carousel batch into canvas."""
    h = canvas.shape[0]

    # Panel background
    canvas[:, :PANEL_W] = PANEL_BG

    # Header bar
    cv2.rectangle(canvas, (0, 0), (PANEL_W, HEADER_H), (10, 10, 10), -1)
    cv2.putText(canvas, "TACTICAL EVENTS", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

    # Running clock (top-right of header)
    t_sec = frame_num / max(fps, 1)
    ts    = f"{int(t_sec // 60):02d}:{int(t_sec % 60):02d}"
    ts_w  = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
    cv2.putText(canvas, ts, (PANEL_W - ts_w - 6, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)

    cv2.line(canvas, (0, HEADER_H), (PANEL_W, HEADER_H), (55, 55, 55), 1)

    # Ranked batch — highest score at top, one line each:
    # "[MM:SS] EVENTNAME P<id>"
    y = HEADER_H + ROW_H - 4
    for ev in batch:
        if y > h - 10:
            break
        color = EVENT_COLORS.get(ev['type'], (180, 180, 180))
        et    = ev['frame'] / max(fps, 1)
        line  = f"[{int(et // 60):02d}:{int(et % 60):02d}] {ev['type']} P{ev['player_id']}"

        # Semi-transparent dark row background
        ov = canvas[:, :PANEL_W].copy()
        cv2.rectangle(ov, (4, y - ROW_H + 6), (PANEL_W - 4, y + 4), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, canvas[:, :PANEL_W], 0.45, 0, canvas[:, :PANEL_W])

        cv2.putText(canvas, line, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        y += ROW_H

    # Vertical divider between panel and video
    cv2.line(canvas, (PANEL_W, 0), (PANEL_W, h), (60, 60, 60), 1)


def render_output1(tracks, fps, annotated_frames, ranked_windows=None, window_frames=None):
    """
    Parameters
    ----------
    tracks           : full tracking dict (players, ball, referees) — kept
                       for call-site compatibility; not used for detection.
    fps              : frames per second
    annotated_frames : list of BGR frames (1920px wide) — the base tracking video
    ranked_windows   : dict window_idx -> list of scored-event dicts
                       (base_weight/intensity/score/metric), precomputed
                       ONCE by tactical_events.event_ranking.rank_events_by_window.
    window_frames    : int, frames per window — window_idx = frame_num // window_frames.
    """
    if not annotated_frames:
        print("render_output1: no frames, skipping.")
        return

    ranked_windows = ranked_windows or {}
    window_frames  = window_frames or (fps * 20)

    orig_h, orig_w = annotated_frames[0].shape[:2]
    out_w = PANEL_W + orig_w

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out    = cv2.VideoWriter('output_videos/output1.avi', fourcc, fps,
                             (out_w, orig_h))

    n = len(annotated_frames)
    total_events = sum(len(v) for v in ranked_windows.values())

    for frame_num in tqdm(range(n), desc="Rendering output1 (tactical events carousel)"):
        window_idx = frame_num // window_frames
        batch = ranked_windows.get(window_idx, [])

        # Build canvas: panel on left, video on right
        canvas = np.zeros((orig_h, out_w, 3), dtype=np.uint8)
        canvas[:, PANEL_W:] = annotated_frames[frame_num]

        _draw_panel(canvas, frame_num, fps, batch)

        if frame_num % 50 == 0:
            print(f"output1: frame {frame_num}/{n}  window={window_idx}  batch_size={len(batch)}")

        out.write(canvas)

    out.release()
    path    = 'output_videos/output1.avi'
    size_mb = os.path.getsize(path) / 1e6
    print(f"output1.avi saved ({n} frames, {size_mb:.1f} MB).")
    print(f"Event summary: {total_events} total ranked tactical events across "
          f"{len(ranked_windows)} windows.")
