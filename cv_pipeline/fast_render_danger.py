"""
fast_render_danger.py — "Danger Meter" video overlay: renders the exact
validated danger-score pipeline from analyze_danger_score.py directly
onto match footage, showing not just THAT a moment is dangerous but WHY,
in the same quantified terms already proven in that script's reports.

Standalone, additive render script. Does NOT import, modify, or touch
main.py, fast_setup.py, fast_render_1/2/3/4/6.py, or any render_output*.py
file. Reuses analyze_danger_score.run() as-is for ALL scoring and
classification — loading, team assignment, dedup/referee/touchline
exclusion, the five-signal open-play formula and its weights, smoothing,
shot/set-piece detection, and the UNCERTAIN ball-detection-gap
classification are 100% the same code path already validated there. This
script only adds drawing.

Four VISIBLE overlay states, one per classify_frames() category:
  - OPEN_PLAY / NONE: the live meter — main 0-100 composite score
    colour-coded green/yellow/red, five signal bars with the current top
    contributor highlighted, small category tag. ("NONE" just means the
    composite score is below DANGER_PEAK_THRESHOLD — still real data,
    still worth showing, unlike UNCERTAIN.)
  - SHOT: live meter unchanged, PLUS a callout banner with the shot's own
    velocity+proximity score (shot_score) — a separate scale from the
    composite, per analyze_danger_score's shot-detection design.
  - SET_PIECE: live meter unchanged, PLUS a callout banner with the
    dead-ball detector's own confidence score — never the open-play
    formula's number, per that category's design (a static congregation
    isn't scored as organized attacking danger).

One INVISIBLE state:
  - UNCERTAIN (long ball-detection gap, see analyze_danger_score's
    UNCERTAIN_GAP_MAX_FRAMES): renders NOTHING — no panel, no label, no
    score, no category text. Pure pass-through of the source frame. These
    windows are written to a notes file (via run()'s notes_path) for the
    written report, deliberately never surfaced on screen — a fabricated
    interpolated-data score must never look as confident as a real one.

No end-of-clip summary card in this build (omitted per current spec).

Usage:
    python fast_render_danger.py [start_frame] [end_frame] [--source fastpipe]
    (defaults to 39000:39400, 26:00-26:16 — spans both the validated
    26:07 open-play peak and the 26:12 shot/UNCERTAIN gap)
"""

import os

import cv2
import numpy as np

from analyze_danger_score import run as score_run, WEIGHTS, DANGER_PEAK_THRESHOLD

# ── Overlay tuning ───────────────────────────────────────────────────────────
DANGER_YELLOW_THRESHOLD = 40.0   # below: green: "under control". Above DANGER_PEAK_THRESHOLD (55): red.

SIGNAL_LABELS = {
    'ball_proximity':      'Ball Proximity',
    'space_control':       'Attack Space Ctrl',
    'compactness':         'Def. Disorganization',   # inverted label — same score, see analyze_danger_score
    'numerical_advantage': 'Numerical Adv.',
    'ball_velocity':       'Ball Velocity',
}
SIGNAL_ORDER = list(WEIGHTS.keys())

CATEGORY_TAG_LABELS = {
    'OPEN_PLAY': 'OPEN PLAY',
    'NONE':      'CALM',
    'SHOT':      'SHOT',
    'SET_PIECE': 'SET PIECE',
}
CATEGORY_TAG_COLORS = {
    'OPEN_PLAY': (0, 210, 235),
    'NONE':      (150, 150, 150),
    'SHOT':      (40, 40, 230),
    'SET_PIECE': (0, 200, 255),
}

COLOR_GREEN  = (60, 200, 60)
COLOR_YELLOW = (0, 210, 235)
COLOR_RED    = (40, 40, 230)
COLOR_WHITE  = (255, 255, 255)
COLOR_GOLD   = (0, 200, 255)
COLOR_PANEL_BG = (20, 20, 20)
COLOR_BAR_BG   = (70, 70, 70)
COLOR_BAR_FG   = (150, 150, 150)
COLOR_BAR_HOT  = (0, 200, 255)

DEFAULT_START = 39000
DEFAULT_END   = 39400
OUT_PATH = 'output_videos/longer_output_danger.avi'


def _danger_color(score):
    if score >= DANGER_PEAK_THRESHOLD:
        return COLOR_RED
    if score >= DANGER_YELLOW_THRESHOLD:
        return COLOR_YELLOW
    return COLOR_GREEN


def _blend_rect(frame, x0, y0, x1, y1, color, alpha):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _shot_callout_text(row):
    """SHOT's own velocity+proximity score — not the composite danger
    score, per analyze_danger_score's independent shot-scoring design."""
    speed = row.get('shot_signal_kmh') or 0.0
    dist = row.get('ball_dist_to_goal_m') or 0.0
    score = row.get('shot_score') or 0.0
    return f"SHOT -- {score:.0f}/100  ({speed:.0f} km/h, {dist:.1f}m out)"


def _set_piece_callout_text(row):
    """SET_PIECE's own dead-ball confidence — never the open-play formula's
    number, per that category's design (static congestion != organized
    attacking danger)."""
    conf = row.get('set_piece_confidence') or 0.0
    ev = row.get('set_piece_evidence') or {}
    static_pct = 100.0 * (ev.get('ball_static_fraction') or 0.0)
    return f"SET PIECE -- {conf:.0f}/100 confidence  (ball static {static_pct:.0f}%)"


def _draw_live_meter(frame, r):
    px0, py0 = 30, 30
    pw, ph = 480, 300
    _blend_rect(frame, px0, py0, px0 + pw, py0 + ph, COLOR_PANEL_BG, 0.6)

    danger = r['danger_smoothed']
    color = _danger_color(danger)
    category = r['category']

    cv2.putText(frame, "DANGER METER", (px0 + 16, py0 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1, cv2.LINE_AA)
    tag = CATEGORY_TAG_LABELS.get(category, category)
    tag_color = CATEGORY_TAG_COLORS.get(category, COLOR_WHITE)
    tag_size = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(frame, tag, (px0 + pw - tag_size[0] - 16, py0 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, tag_color, 1, cv2.LINE_AA)

    cv2.putText(frame, f"{danger:5.1f}", (px0 + 16, py0 + 82),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
    cv2.putText(frame, "/ 100", (px0 + 190, py0 + 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1, cv2.LINE_AA)

    contribs = r['contributions_smoothed']
    dominant = max(contribs, key=contribs.get)

    bar_x0 = px0 + 190
    bar_w_max = 270
    bar_h = 16
    row_gap = 32
    y = py0 + 112
    for sig in SIGNAL_ORDER:
        value = r['scores'][sig]
        is_dominant = (sig == dominant)
        label_color = COLOR_BAR_HOT if is_dominant else COLOR_WHITE
        cv2.putText(frame, SIGNAL_LABELS[sig], (px0 + 16, y + bar_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_color, 1, cv2.LINE_AA)

        cv2.rectangle(frame, (bar_x0, y), (bar_x0 + bar_w_max, y + bar_h), COLOR_BAR_BG, -1)
        fill_w = int(bar_w_max * max(0.0, min(100.0, value)) / 100.0)
        bar_color = COLOR_BAR_HOT if is_dominant else COLOR_BAR_FG
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x0, y), (bar_x0 + fill_w, y + bar_h), bar_color, -1)
        if is_dominant:
            cv2.rectangle(frame, (bar_x0, y), (bar_x0 + bar_w_max, y + bar_h), COLOR_BAR_HOT, 1)
        cv2.putText(frame, f"{value:5.1f}", (bar_x0 + bar_w_max + 10, y + bar_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_color, 1, cv2.LINE_AA)
        y += row_gap


def _draw_callout(frame, text):
    fw = frame.shape[1]
    box_w, box_h = 720, 56
    x0 = (fw - box_w) // 2
    y0 = 40
    _blend_rect(frame, x0, y0, x0 + box_w, y0 + box_h, (0, 0, 0), 0.55)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), COLOR_GOLD, 2)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
    tx = x0 + (box_w - text_size[0]) // 2
    ty = y0 + (box_h + text_size[1]) // 2
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_GOLD, 2, cv2.LINE_AA)


def render(source='fastpipe', start_frame=DEFAULT_START, end_frame=DEFAULT_END, out_path=OUT_PATH):
    print(f"fast_render_danger: scoring [{start_frame}:{end_frame}) via "
          f"analyze_danger_score.run() (source='{source}') ...")
    csv_path = f"danger_score_render_{start_frame}_{end_frame}.csv"
    notes_path = f"danger_score_notes_render_{start_frame}_{end_frame}.txt"
    result = score_run(source=source, start_frame=start_frame, end_frame=end_frame,
                        out_csv=csv_path, top_n=10, clean_only=True, notes_path=notes_path)

    rows = result['rows']
    fps = result['fps']
    video_path = result['video_path']
    start_frame = result['start_frame']
    end_frame = result['end_frame']
    n = end_frame - start_frame

    row_by_frame = {r['frame']: r for r in rows}
    counts = result['category_counts']
    print("Category breakdown for this clip: "
          + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"{len(result['uncertain_gaps'])} UNCERTAIN gap(s) in this clip "
          f"(rendered blank — see {notes_path})")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = None

    fn = 0
    n_blank = 0
    while fn < n:
        ret, frame = cap.read()
        if not ret:
            print(f"WARNING: video ended early at local frame {fn}/{n}")
            break
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        abs_frame = start_frame + fn
        r = row_by_frame.get(abs_frame)
        if r is not None and r['category'] != 'UNCERTAIN':
            _draw_live_meter(frame, r)
            if r['category'] == 'SHOT':
                _draw_callout(frame, _shot_callout_text(r))
            elif r['category'] == 'SET_PIECE':
                _draw_callout(frame, _set_piece_callout_text(r))
        else:
            n_blank += 1   # UNCERTAIN (or no row) — pure pass-through, no overlay

        writer.write(frame)
        fn += 1

    cap.release()
    if writer is not None:
        writer.release()
    cap_check = os.path.exists(out_path)
    print(f"Saved {out_path}  ({fn} frames written, {n_blank} rendered blank/UNCERTAIN, "
          f"exists={cap_check}, size={os.path.getsize(out_path) if cap_check else 0} bytes)")
    return out_path


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Render the Danger Meter overlay onto match footage.")
    p.add_argument('start_frame', type=int, nargs='?', default=DEFAULT_START)
    p.add_argument('end_frame', type=int, nargs='?', default=DEFAULT_END)
    p.add_argument('--source', default='fastpipe')
    p.add_argument('--out', default=OUT_PATH)
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    render(source=args.source, start_frame=args.start_frame, end_frame=args.end_frame, out_path=args.out)
