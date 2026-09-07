# -*- coding: utf-8 -*-
"""
Diagnostic: why does the Voronoi flicker in output3?

Simulates the render_output3 recompute/LERP logic for frames 50-75 WITHOUT
actually rendering, and prints the player-set and state changes that explain
the flickering.

Does NOT change any code.
"""

import sys, os, pickle
import numpy as np
import cv2
from scipy.spatial import Voronoi
from tqdm import tqdm

sys.path.insert(0, r'c:\UCLA\yolo model 2')
from utils import get_foot_position

TRACK_STUB = 'stubs/track_stubs_121364.pkl'
CAM_STUB   = 'stubs/camera_movement_stub_121364.pkl'
HOG_STUB   = 'stubs/homography_stub_121364.pkl'

DIAG_START = 50
DIAG_END   = 75
SIMUL_FPS  = 24          # as passed by run_outputs_3_4.py and main.py

# render_output3 constants (copied from file)
LERP_ALPHA  = 0.20
FADE_FRAMES = 10
RESAMPLE_N  = 32
PITCH_CLIP  = [
    (110.0,  1035.0),
    (265.0,   275.0),
    (1750.0,  260.0),
    (1900.0, 1035.0),
]

# ── Load stubs ────────────────────────────────────────────────────────────────
print("Loading stubs...")
with open(TRACK_STUB, 'rb') as f: tracks = pickle.load(f)
with open(CAM_STUB,   'rb') as f: cam_mv = pickle.load(f)

update_every = max(1, round(SIMUL_FPS))
print(f"fps={SIMUL_FPS}  update_every={update_every}")
print(f"Recompute frames in 0-750: "
      f"{[fn for fn in range(0, len(tracks['players'])) if fn % update_every == 0][:20]} ...")
recompute_in_range = [fn for fn in range(DIAG_START, DIAG_END + 1)
                      if fn % update_every == 0]
print(f"Recompute frames in {DIAG_START}-{DIAG_END}: {recompute_in_range}")
print()

# ── Add positions (no team assignment — check raw tracker stability first) ────
for fn, frame in enumerate(tracks['players']):
    cm = cam_mv[fn] if fn < len(cam_mv) else [0, 0]
    for pid, info in frame.items():
        pos = get_foot_position(info['bbox'])
        info['position_adjusted'] = (pos[0] - float(cm[0]), pos[1] - float(cm[1]))

# ── Load team assignments if available (try loading from a saved diagnostic) ──
# Team assignment requires full video; we'll do a lightweight load here.
# For the stability check we do TWO passes:
#   Pass A: All players (ignore team filter) — shows raw tracker stability
#   Pass B: Filtered players (team 1 or 2) — requires team assignment

NEED_FRAMES = set(range(0, 46, 10)) | set(range(DIAG_START, DIAG_END + 1))
video_src   = 'Match_videos/121364_0.mp4'
print(f"Loading {len(NEED_FRAMES)} video frames for team assignment ...")
cap = cv2.VideoCapture(video_src)
video_frames = {}
for fn in sorted(NEED_FRAMES):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frm = cap.read()
    if ret:
        video_frames[fn] = frm
cap.release()
print(f"  loaded {len(video_frames)} frames")

class _SparseFrameList:
    def __init__(self, d, n): self.d = d; self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i): return self.d[i]

from team_assigner import TeamAssigner
ta = TeamAssigner()
ta.assign_team_color(_SparseFrameList(video_frames, len(tracks['players'])),
                     tracks['players'])

print("Assigning teams for frames 50-75 ...")
for fn in range(DIAG_START, DIAG_END + 1):
    if fn not in video_frames:
        continue
    for pid, info in tracks['players'][fn].items():
        team = ta.get_player_team(video_frames[fn], info['bbox'], pid)
        info['team'] = team

print()

# ── Sutherland-Hodgman (same as render_output3) ───────────────────────────────
def _sh_inside(p, a, b):
    return ((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])) >= 0

def _sh_intersect(s, e, a, b):
    dc = (a[0]-b[0], a[1]-b[1]); dp = (s[0]-e[0], s[1]-e[1])
    n1 = a[0]*b[1] - a[1]*b[0]; n2 = s[0]*e[1] - s[1]*e[0]
    d  = dc[0]*dp[1] - dc[1]*dp[0]
    if abs(d) < 1e-10: return s
    r = 1.0 / d
    return ((n1*dp[0] - n2*dc[0])*r, (n1*dp[1] - n2*dc[1])*r)

def _sutherland_hodgman(subject, clip):
    output = list(subject)
    if not output: return []
    for i in range(len(clip)):
        if not output: return []
        inp = output; output = []
        a, b = clip[i], clip[(i+1) % len(clip)]
        for j in range(len(inp)):
            curr, prev = inp[j], inp[j-1]
            if _sh_inside(curr, a, b):
                if not _sh_inside(prev, a, b):
                    output.append(_sh_intersect(prev, curr, a, b))
                output.append(curr)
            elif _sh_inside(prev, a, b):
                output.append(_sh_intersect(prev, curr, a, b))
    return output

def _resample_polygon(pts_list, n=RESAMPLE_N):
    pts = np.array(pts_list, dtype=np.float64)
    m = len(pts)
    if m < 2: return np.tile(pts[0] if m else [0,0], (n, 1))
    rolled = np.roll(pts, -1, axis=0)
    seg_lens = np.sqrt(((rolled - pts)**2).sum(axis=1))
    total = seg_lens.sum()
    if total < 1e-6: return np.tile(pts[0], (n, 1))
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    ts = np.linspace(0.0, total, n, endpoint=False)
    seg_idx = np.clip(np.searchsorted(cum, ts, side='right') - 1, 0, m-1)
    t_local = np.clip((ts - cum[seg_idx]) / np.maximum(seg_lens[seg_idx%m], 1e-10), 0, 1)
    return pts[seg_idx%m] + t_local[:,None] * (pts[(seg_idx+1)%m] - pts[seg_idx%m])

# ── PASS A: Raw tracker stability (no team filter) ────────────────────────────
print("=" * 68)
print("PASS A — RAW TRACKER STABILITY (no team filter, frames 50-75)")
print("=" * 68)

w_f, h_f = float(video_frames[DIAG_START].shape[1]), float(video_frames[DIAG_START].shape[0])
guard = [
    (w_f*0.5, -2*h_f), (3*w_f, -2*h_f), (3*w_f, h_f*0.5), (3*w_f, 3*h_f),
    (w_f*0.5, 3*h_f), (-2*w_f, 3*h_f), (-2*w_f, h_f*0.5), (-2*w_f, -2*h_f),
]

prev_pid_set = None
for fn in range(DIAG_START, DIAG_END + 1):
    if fn >= len(tracks['players']): break
    player_frame = tracks['players'][fn]
    # Collect all players with valid position_adjusted (no team filter)
    pid_set = set()
    for pid, info in player_frame.items():
        adj = info.get('position_adjusted')
        if adj is None: continue
        x, y = float(adj[0]), float(adj[1])
        if x == 0.0 and y == 0.0: continue
        pid_set.add(pid)

    is_recompute = (fn % update_every == 0)
    marker = " <-- RECOMPUTE" if is_recompute else ""
    diff_str = ""
    if prev_pid_set is not None:
        added   = pid_set - prev_pid_set
        removed = prev_pid_set - pid_set
        if added or removed:
            diff_str = f"  DELTA: +{sorted(added)} -{sorted(removed)}"
    print(f"  frame {fn:3d} | {len(pid_set):2d} players | "
          f"{sorted(pid_set)}{marker}{diff_str}")
    prev_pid_set = pid_set

# ── PASS B: Team-filtered stability ──────────────────────────────────────────
print()
print("=" * 68)
print("PASS B — TEAM-FILTERED STABILITY (team 1 or 2 only, frames 50-75)")
print("=" * 68)

prev_pid_set = None
for fn in range(DIAG_START, DIAG_END + 1):
    if fn >= len(tracks['players']): break
    player_frame = tracks['players'][fn]
    pid_set = set()
    team_map = {}
    for pid, info in player_frame.items():
        team = info.get('team', 0)
        if team not in (1, 2): continue
        adj = info.get('position_adjusted')
        if adj is None: continue
        x, y = float(adj[0]), float(adj[1])
        if x == 0.0 and y == 0.0: continue
        pid_set.add(pid)
        team_map[pid] = team

    is_recompute = (fn % update_every == 0)
    marker = " <-- RECOMPUTE" if is_recompute else ""
    diff_str = ""
    if prev_pid_set is not None:
        added   = pid_set - prev_pid_set
        removed = prev_pid_set - pid_set
        if added or removed:
            diff_str = f"  DELTA: +{sorted(added)} -{sorted(removed)}"
    print(f"  frame {fn:3d} | {len(pid_set):2d} players | "
          f"{sorted(pid_set)}{marker}{diff_str}")
    prev_pid_set = pid_set

# ── PASS C: Simulate render_output3 LERP state ───────────────────────────────
print()
print("=" * 68)
print("PASS C — SIMULATE render_output3 STATE (frames 0 to 75, team-filtered)")
print("=" * 68)
print("(Running recompute from frame 0 to build up state correctly)")

target_poly  = {}
current_poly = {}
player_age   = {}
lost_since   = {}
lost_poly    = {}
last_recompute_pids = set()

for fn in range(0, DIAG_END + 1):
    if fn >= len(tracks['players']): break
    player_frame = tracks['players'][fn]
    is_recompute = (fn % update_every == 0)

    if is_recompute:
        player_pos = {}
        for pid, info in player_frame.items():
            team = info.get('team', 0)
            if team not in (1, 2): continue
            adj = info.get('position_adjusted')
            if adj is None: continue
            x, y = float(adj[0]), float(adj[1])
            if x == 0.0 and y == 0.0: continue
            player_pos[pid] = (x, y)

        if len(player_pos) >= 2:
            pids_list = list(player_pos.keys())
            coords    = [player_pos[p] for p in pids_list]
            pts_arr   = np.array(coords + guard, dtype=np.float64)
            try:
                vor = Voronoi(pts_arr)
                new_targets = {}
                for i, pid in enumerate(pids_list):
                    ridx   = vor.point_region[i]
                    region = vor.regions[ridx]
                    if not region or -1 in region or len(region) < 3: continue
                    verts   = [(float(vor.vertices[v][0]),
                                float(vor.vertices[v][1])) for v in region]
                    clipped = _sutherland_hodgman(verts, PITCH_CLIP)
                    if len(clipped) >= 3:
                        new_targets[pid] = _resample_polygon(clipped)

                added_pids   = set(new_targets) - set(target_poly)
                removed_pids = set(target_poly) - set(new_targets)

                for pid in list(target_poly.keys()):
                    if pid not in new_targets and pid not in lost_since:
                        if pid in current_poly:
                            lost_since[pid] = 1
                            lost_poly[pid]  = current_poly[pid].copy()

                for pid, poly in new_targets.items():
                    if pid in lost_since:
                        lost_since.pop(pid); lost_poly.pop(pid, None)
                    target_poly[pid] = poly
                    if pid not in current_poly:
                        current_poly[pid] = poly.copy()
                        player_age[pid]   = 0

                last_recompute_pids = set(new_targets.keys())
            except Exception as e:
                added_pids = removed_pids = set()

        if fn >= DIAG_START:
            print(f"\n  frame {fn:3d} RECOMPUTE:")
            print(f"    player_pos set   : {sorted(player_pos.keys())} ({len(player_pos)})")
            print(f"    new_targets set  : {sorted(last_recompute_pids)} ({len(last_recompute_pids)})")
            print(f"    target_poly keys : {sorted(target_poly.keys())} ({len(target_poly)})")
            print(f"    current_poly keys: {sorted(current_poly.keys())} ({len(current_poly)})")
            print(f"    lost_since keys  : {sorted(lost_since.keys())}")
            if added_pids or removed_pids:
                print(f"    TOPOLOGY CHANGE vs prev recompute: +{sorted(added_pids)} -{sorted(removed_pids)}")
            else:
                print(f"    topology vs prev recompute: SAME")

    # Advance lost counters
    for pid in list(lost_since.keys()):
        lost_since[pid] += 1
        if lost_since[pid] > FADE_FRAMES:
            lost_since.pop(pid); lost_poly.pop(pid, None)
            current_poly.pop(pid, None); target_poly.pop(pid, None)
            player_age.pop(pid, None)

    # LERP
    for pid in list(current_poly.keys()):
        if pid in lost_since: continue
        if pid in target_poly:
            current_poly[pid] = ((1 - LERP_ALPHA)*current_poly[pid]
                                 + LERP_ALPHA*target_poly[pid])
        player_age[pid] = min(player_age.get(pid, 0) + 1, FADE_FRAMES)

    if fn >= DIAG_START:
        if not is_recompute:
            lerp_in_sync = all(pid in target_poly for pid in current_poly
                               if pid not in lost_since)
            nv_match = all(current_poly[pid].shape == target_poly[pid].shape
                           for pid in current_poly if pid in target_poly)
            print(f"  frame {fn:3d} LERP  | "
                  f"current_poly={len(current_poly)} target_poly={len(target_poly)} "
                  f"lost={len(lost_since)} "
                  f"all_pids_synced={lerp_in_sync} shapes_match={nv_match}")

# ── PASS D: Cell identity stability for player 13 ────────────────────────────
print()
print("=" * 68)
print("PASS D — CELL IDENTITY CHECK FOR PLAYER 13 (frames 50-75)")
print("=" * 68)
print("Re-running simulation and tracking what Voronoi cell shape PID 13 gets")
print("at each recompute. If shape changes wildly, the cell is not stable.")

target_poly  = {}
current_poly = {}
player_age   = {}
lost_since   = {}
lost_poly    = {}

for fn in range(0, DIAG_END + 1):
    if fn >= len(tracks['players']): break
    player_frame = tracks['players'][fn]
    is_recompute = (fn % update_every == 0)

    if is_recompute:
        player_pos = {}
        for pid, info in player_frame.items():
            team = info.get('team', 0)
            if team not in (1, 2): continue
            adj = info.get('position_adjusted')
            if adj is None: continue
            x, y = float(adj[0]), float(adj[1])
            if x == 0.0 and y == 0.0: continue
            player_pos[pid] = (x, y)

        if len(player_pos) >= 2:
            pids_list = list(player_pos.keys())
            coords    = [player_pos[p] for p in pids_list]
            pts_arr   = np.array(coords + guard, dtype=np.float64)
            try:
                vor = Voronoi(pts_arr)
                new_targets = {}
                for i, pid in enumerate(pids_list):
                    ridx   = vor.point_region[i]
                    region = vor.regions[ridx]
                    if not region or -1 in region or len(region) < 3: continue
                    verts   = [(float(vor.vertices[v][0]),
                                float(vor.vertices[v][1])) for v in region]
                    clipped = _sutherland_hodgman(verts, PITCH_CLIP)
                    if len(clipped) >= 3:
                        new_targets[pid] = _resample_polygon(clipped)

                for pid in list(target_poly.keys()):
                    if pid not in new_targets and pid not in lost_since:
                        if pid in current_poly:
                            lost_since[pid] = 1; lost_poly[pid] = current_poly[pid].copy()

                for pid, poly in new_targets.items():
                    if pid in lost_since:
                        lost_since.pop(pid); lost_poly.pop(pid, None)
                    target_poly[pid] = poly
                    if pid not in current_poly:
                        current_poly[pid] = poly.copy(); player_age[pid] = 0

                if fn >= DIAG_START and 13 in new_targets:
                    p13 = new_targets[13]
                    cx  = float(p13[:,0].mean())
                    cy  = float(p13[:,1].mean())
                    rng_x = float(p13[:,0].max() - p13[:,0].min())
                    rng_y = float(p13[:,1].max() - p13[:,1].min())
                    print(f"  frame {fn:3d} RECOMPUTE  pid13 target: "
                          f"centroid=({cx:.0f},{cy:.0f}) "
                          f"span=({rng_x:.0f}x{rng_y:.0f}px)  "
                          f"n_verts_preresample=len(clipped)?")
                elif fn >= DIAG_START:
                    print(f"  frame {fn:3d} RECOMPUTE  pid13: NOT in new_targets "
                          f"(not in player set this recompute)")
            except Exception:
                pass

    for pid in list(lost_since.keys()):
        lost_since[pid] += 1
        if lost_since[pid] > FADE_FRAMES:
            lost_since.pop(pid); lost_poly.pop(pid, None)
            current_poly.pop(pid, None); target_poly.pop(pid, None)
            player_age.pop(pid, None)

    for pid in list(current_poly.keys()):
        if pid in lost_since: continue
        if pid in target_poly:
            current_poly[pid] = ((1-LERP_ALPHA)*current_poly[pid]
                                 + LERP_ALPHA*target_poly[pid])
        player_age[pid] = min(player_age.get(pid, 0) + 1, FADE_FRAMES)

    if fn >= DIAG_START and not is_recompute:
        if 13 in current_poly:
            p13 = current_poly[13]
            cx  = float(p13[:,0].mean())
            cy  = float(p13[:,1].mean())
            print(f"  frame {fn:3d} LERP  pid13: centroid=({cx:.0f},{cy:.0f})")
        else:
            print(f"  frame {fn:3d} LERP  pid13: NOT in current_poly")

print()
print("=" * 68)
print("SUMMARY OF HYPOTHESES")
print("=" * 68)
print("""
  From the above output, the causes of flickering fall into:

  1. PLAYER-SET CHURN (Pass A/B): If player IDs change every few frames,
     the Voronoi topology restructures at every recompute, and the LERP
     cannot smooth a topology change (different cells, different shapes).

  2. TEAM-FILTER INSTABILITY (Pass B vs A): If team filter drops more
     players than raw tracking, we may have only 5-8 players per team
     in the Voronoi, making each individual disappearance more impactful.

  3. RECOMPUTE FREQUENCY (Pass C): If update_every != 24, recomputes
     happen more often, causing more frequent topology snaps.

  4. SHAPE MISMATCH IN LERP (Pass C shapes_match): If target_poly and
     current_poly have different vertex counts, LERP is wrong.

  5. CELL IDENTITY SHUFFLE (Pass D): If player 13 always maps to the
     same Voronoi cell, the colour is stable. If the cell centroid jumps
     wildly between recomputes, the cell boundaries are unstable.
""")
