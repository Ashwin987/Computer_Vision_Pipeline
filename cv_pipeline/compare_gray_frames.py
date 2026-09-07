"""
Diagnostic-only script (not part of the pipeline). Compares gray-frame
(team==0) percentage under the OLD sequential/online locking path
(get_player_team, called per-frame in temporal order) vs the NEW two-pass
batch path (resolve_all_teams + finalize_fallback_assignments +
merge_fragmented_tracks + apply_final_teams), on the exact same stub/video,
using today's code for both — isolates the sequential-vs-batch effect from
any other tuning.
"""

import copy
import pickle
from collections import Counter

from utils import read_video
from team_assigner import TeamAssigner

TRACK_STUB = 'stubs/track_stubs_121364_ball_fallback.pkl'
VIDEO_PATH = 'Match_videos/121364_0.mp4'

with open(TRACK_STUB, 'rb') as f:
    base_tracks = pickle.load(f)

print("Reading video frames...")
video_frames = read_video(VIDEO_PATH)


def total_and_gray(players):
    total = 0
    gray = 0
    for fd in players:
        for pid, info in fd.items():
            total += 1
            if info.get('team', 0) == 0:
                gray += 1
    return total, gray


def per_player_gray_frac(players):
    counts = {}
    for fd in players:
        for pid, info in fd.items():
            t, g = counts.get(pid, (0, 0))
            counts[pid] = (t + 1, g + (1 if info.get('team', 0) == 0 else 0))
    return counts


# ── OLD: sequential/online path ────────────────────────────────────────────
old_tracks = copy.deepcopy(base_tracks)
ta_old = TeamAssigner()
ta_old.assign_team_color(video_frames, old_tracks['players'])
for fn, fd in enumerate(old_tracks['players']):
    for pid, info in fd.items():
        team = ta_old.get_player_team(video_frames[fn], info['bbox'], pid,
                                       is_goalkeeper=info.get('is_goalkeeper', False))
        info['team'] = team
old_fallback = ta_old.finalize_fallback_assignments()
if old_fallback:
    for fn, fd in enumerate(old_tracks['players']):
        for pid, info in fd.items():
            if pid in old_fallback:
                info['team'] = old_fallback[pid]
old_merged, old_rejected = ta_old.merge_fragmented_tracks(video_frames, old_tracks['players'])

old_total, old_gray = total_and_gray(old_tracks['players'])
old_per_player = per_player_gray_frac(old_tracks['players'])

# ── NEW: two-pass batch path ───────────────────────────────────────────────
new_tracks = copy.deepcopy(base_tracks)
ta_new = TeamAssigner()
ta_new.resolve_all_teams(video_frames, new_tracks['players'])
new_fallback = ta_new.finalize_fallback_assignments()
new_merged, new_rejected = ta_new.merge_fragmented_tracks(video_frames, new_tracks['players'])
ta_new.apply_final_teams(new_tracks['players'])

new_total, new_gray = total_and_gray(new_tracks['players'])
new_per_player = per_player_gray_frac(new_tracks['players'])

print("\n=== OVERALL GRAY-FRAME PERCENTAGE (all player-frame instances) ===")
print(f"OLD (sequential/online): {old_gray}/{old_total} = {100*old_gray/old_total:.1f}% gray")
print(f"NEW (two-pass batch):    {new_gray}/{new_total} = {100*new_gray/new_total:.1f}% gray")

print("\n=== END-OF-CLIP RESOLVED/UNRESOLVED PLAYER COUNTS ===")
old_unresolved_players = [pid for pid, (t, g) in old_per_player.items() if g == t]
new_unresolved_players = [pid for pid, (t, g) in new_per_player.items() if g == t]
print(f"OLD: {len(old_unresolved_players)}/{len(old_per_player)} players 100% gray "
      f"(never resolved a single frame): {sorted(old_unresolved_players)}")
print(f"NEW: {len(new_unresolved_players)}/{len(new_per_player)} players 100% gray "
      f"(never resolved a single frame): {sorted(new_unresolved_players)}")

print("\n=== PIDS 10, 81, 82 — per-player gray fraction ===")
for pid in (10, 81, 82):
    ot, og = old_per_player.get(pid, (0, 0))
    nt, ng = new_per_player.get(pid, (0, 0))
    print(f"pid {pid}: OLD {og}/{ot} ({100*og/ot:.0f}% gray)  ->  "
          f"NEW {ng}/{nt} ({100*ng/nt:.0f}% gray)")

print("\n=== Players with >20% gray screen time despite eventually resolving (OLD) ===")
bad_old = sorted([(pid, g, t) for pid, (t, g) in old_per_player.items() if 0 < g < t and g/t > 0.2],
                  key=lambda x: -x[1]/x[2])
for pid, g, t in bad_old:
    print(f"    pid {pid}: {g}/{t} = {100*g/t:.0f}% gray")
print(f"  count: {len(bad_old)}")

print("\n=== Same check under NEW (should be ~0, since Pass 2 applies one team uniformly) ===")
bad_new = sorted([(pid, g, t) for pid, (t, g) in new_per_player.items() if 0 < g < t],
                  key=lambda x: -x[1]/x[2])
for pid, g, t in bad_new:
    print(f"    pid {pid}: {g}/{t} = {100*g/t:.0f}% gray (partial — should not happen under two-pass)")
print(f"  count: {len(bad_new)}")

print("\n=== pid 71 / pid 88 investigation (newly-unresolved under NEW) ===")
print(f"OLD merged pairs: {old_merged}")
print(f"OLD rejected pairs involving 71/88: "
      f"{[r for r in old_rejected if r['a'] in (71,88) or r['b'] in (71,88)]}")
print(f"NEW merged pairs: {new_merged}")
print(f"NEW rejected pairs involving 71/88: "
      f"{[r for r in new_rejected if r['a'] in (71,88) or r['b'] in (71,88)]}")
print(f"OLD player_team_dict[71]={ta_old.player_team_dict.get(71)} [88]={ta_old.player_team_dict.get(88)}")
print(f"NEW player_team_dict[71]={ta_new.player_team_dict.get(71)} [88]={ta_new.player_team_dict.get(88)}")
print(f"OLD best_unresolved_distance[71]={ta_old.best_unresolved_distance.get(71)}")
print(f"OLD best_unresolved_distance[88]={ta_old.best_unresolved_distance.get(88)}")
print(f"NEW best_unresolved_distance[71]={ta_new.best_unresolved_distance.get(71)}")
print(f"NEW best_unresolved_distance[88]={ta_new.best_unresolved_distance.get(88)}")
print(f"NEW fallback_assigned: {new_fallback}")
print(f"OLD fallback_assigned: {old_fallback}")
