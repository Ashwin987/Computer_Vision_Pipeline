"""
Diagnostic-only script (not part of the pipeline). For a list of pids
reported as visually red-kit but locked to Team 2 (green), this:
  1. Reruns the same evidence-gathering resolve_all_teams uses, but records
     per-frame (color, distance-to-each-centroid, reliable, guess) instead
     of just the final aggregate vote.
  2. Reports how each pid actually got resolved (normal lock / fallback /
     merge) and the evidence that drove it.
  3. Saves the exact wide-torso crop (the region get_player_color samples)
     for each confident sample frame, plus a padded full-bbox crop for
     context, as PNGs for visual inspection.
"""

import os
import pickle
import cv2
import numpy as np

from utils import read_video
from team_assigner import TeamAssigner
from team_assigner.team_assigner_color_bak import (
    CONFIDENCE_THRESHOLD, MIN_SAMPLES_TO_LOCK, AGREEMENT_THRESHOLD,
)

TRACK_STUB = 'stubs/track_stubs_121364_ball_fallback.pkl'
VIDEO_PATH = 'Match_videos/121364_0.mp4'
TARGET_PIDS = [48, 4, 11, 21, 16]

OUT_DIR = ('C:/Users/latha/AppData/Local/Temp/claude/c--UCLA-yolo-model-2/'
           '07e961b4-3c2c-4ec6-93c0-cfd011bdf006/scratchpad/red_as_green')
os.makedirs(OUT_DIR, exist_ok=True)

with open(TRACK_STUB, 'rb') as f:
    tracks = pickle.load(f)

print("Reading video frames...")
video_frames = read_video(VIDEO_PATH)

ta = TeamAssigner()
ta.assign_team_color(video_frames, tracks['players'])
c1 = ta.locked_colors[1]
c2 = ta.locked_colors[2]
print(f"\nTeam 1 centroid BGR: {c1}")
print(f"Team 2 centroid BGR: {c2}")

# ── Per-frame evidence for target pids only ─────────────────────────────
evidence = {pid: [] for pid in TARGET_PIDS}
n_frames = min(len(video_frames), len(tracks['players']))
for fn in range(n_frames):
    for pid in TARGET_PIDS:
        det = tracks['players'][fn].get(pid)
        if det is None:
            continue
        bbox = det.get('bbox')
        if bbox is None:
            continue
        is_gk = det.get('is_goalkeeper', False)
        color, reliable = ta.get_player_color(video_frames[fn], bbox, return_reliability=True)
        d1 = float(np.linalg.norm(color - c1))
        d2 = float(np.linalg.norm(color - c2))
        d = min(d1, d2)
        guess = 1 if d1 <= d2 else 2
        confident = reliable and d < CONFIDENCE_THRESHOLD
        evidence[pid].append({
            'fn': fn, 'bbox': bbox, 'is_gk': is_gk, 'color': color,
            'reliable': reliable, 'd1': d1, 'd2': d2, 'guess': guess,
            'confident': confident,
        })

# ── Full resolution (to know HOW each pid actually got resolved) ────────
ta2 = TeamAssigner()
ta2.resolve_all_teams(video_frames, tracks['players'])
fallback = ta2.finalize_fallback_assignments()
merged, rejected = ta2.merge_fragmented_tracks(video_frames, tracks['players'])

for pid in TARGET_PIDS:
    print("\n" + "=" * 70)
    print(f"PID {pid}")
    print("=" * 70)
    ev = evidence[pid]
    print(f"Total frames present: {len(ev)} "
          f"(first={ev[0]['fn'] if ev else None}, last={ev[-1]['fn'] if ev else None})")
    print(f"is_goalkeeper ever True: {any(e['is_gk'] for e in ev)}")

    final_team = ta2.player_team_dict.get(pid, 0)
    mechanism = 'unresolved'
    if pid in fallback:
        mechanism = f"FALLBACK (dist={fallback[pid]})" if isinstance(fallback[pid], dict) else \
                    f"FALLBACK (team={fallback[pid]}, dist={ta2.fallback_assigned.get(pid)})"
    elif pid in ta2.merged_pids:
        mechanism = "MERGE propagation"
    elif final_team != 0:
        mechanism = "NORMAL lock (resolve_all_teams majority vote)"
    print(f"FINAL team: {final_team}  via: {mechanism}")

    confident = [e for e in ev if e['confident']]
    print(f"Confident (reliable + d<{CONFIDENCE_THRESHOLD}) samples: {len(confident)}")
    guess_counts = {1: sum(1 for e in confident if e['guess'] == 1),
                     2: sum(1 for e in confident if e['guess'] == 2)}
    print(f"  guess=1: {guess_counts[1]}   guess=2: {guess_counts[2]}")
    for e in confident:
        print(f"    fn={e['fn']:4d} d1={e['d1']:5.1f} d2={e['d2']:5.1f} "
              f"guess={e['guess']} color(BGR)={np.round(e['color'],1)}")

    best = ta2.best_unresolved_distance.get(pid)
    print(f"best_unresolved_distance entry: {best}")

    # Save crops: every confident sample frame (capped at 6 for sanity),
    # plus the single best-distance frame if not already included.
    to_save = confident[:6]
    if best is not None:
        best_fn = None
        for e in ev:
            if abs(e['d1'] if e['guess']==1 else e['d2'] - 0) >= 0:
                pass
        # find the frame matching the best distance value
        for e in ev:
            dmin = min(e['d1'], e['d2'])
            if abs(dmin - best[0]) < 1e-6:
                best_fn = e
                break
        if best_fn is not None and best_fn not in to_save:
            to_save.append(best_fn)

    for e in to_save:
        fn = e['fn']
        frame = video_frames[fn]
        x1, y1, x2, y2 = e['bbox']
        fh, fw = frame.shape[:2]
        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
        ix2, iy2 = min(fw - 1, int(x2)), min(fh - 1, int(y2))
        bw, bh = ix2 - ix1, iy2 - iy1

        # Wide torso crop -- exact region get_player_color samples
        cx1 = ix1 + int(bw * 0.15); cx2 = ix1 + int(bw * 0.85)
        cy1 = iy1 + int(bh * 0.15); cy2 = iy1 + int(bh * 0.65)
        cx2 = max(cx1 + 1, min(cx2, fw - 1)); cy2 = max(cy1 + 1, min(cy2, fh - 1))
        torso_crop = frame[cy1:cy2, cx1:cx2]

        # Padded full-bbox crop for context
        PAD = 60
        px1, py1 = max(0, ix1 - PAD), max(0, iy1 - PAD)
        px2, py2 = min(fw, ix2 + PAD), min(fh, iy2 + PAD)
        context_crop = frame[py1:py2, px1:px2]

        tag = f"pid{pid}_fn{fn}_g{e['guess']}_d{min(e['d1'],e['d2']):.0f}"
        if torso_crop.size:
            cv2.imwrite(os.path.join(OUT_DIR, f"{tag}_torso.png"),
                        cv2.resize(torso_crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
        if context_crop.size:
            cv2.imwrite(os.path.join(OUT_DIR, f"{tag}_context.png"), context_crop)
        print(f"  saved crops for fn={fn}")

print(f"\nAll crops saved under: {OUT_DIR}")
print(f"\nrejected pairs (for context on 48/4/11/21/16 if any): "
      f"{[r for r in rejected if r['a'] in TARGET_PIDS or r['b'] in TARGET_PIDS]}")
print(f"merged pairs (for context on 48/4/11/21/16 if any): "
      f"{[m for m in merged if m['a'] in TARGET_PIDS or m['b'] in TARGET_PIDS]}")
