"""
Diagnostic-only script (not part of the pipeline). Reproduces main.py's
exact team-assignment block (same stub, same TeamAssigner import through
the shim, same call sequence) and reports the final resolved team for
specific pids, to check whether pids 81/10/82 are actually unresolved in
the underlying data or only appear gray due to a rendering-layer bug.
"""

import pickle
from utils import read_video
from team_assigner import TeamAssigner

TRACK_STUB = 'stubs/track_stubs_121364_ball_fallback.pkl'
VIDEO_PATH = 'Match_videos/121364_0.mp4'
CHECK_PIDS = [81, 10, 82]

print(f"team_assigner module resolves to: {TeamAssigner.__module__}")

with open(TRACK_STUB, 'rb') as f:
    tracks = pickle.load(f)

print("Reading video frames...")
video_frames = read_video(VIDEO_PATH)
print(f"  -> {len(video_frames)} frames")

for pid in CHECK_PIDS:
    frames_present = [fn for fn, fd in enumerate(tracks['players']) if pid in fd]
    print(f"pid {pid}: present in {len(frames_present)} frames "
          f"(first={frames_present[0] if frames_present else None}, "
          f"last={frames_present[-1] if frames_present else None}), "
          f"is_goalkeeper flags seen: "
          f"{set(tracks['players'][fn][pid].get('is_goalkeeper', False) for fn in frames_present)}")

team_assigner = TeamAssigner()
team_assigner.assign_team_color(video_frames, tracks['players'])

for frame_num, player_track in enumerate(tracks['players']):
    for player_id, track in player_track.items():
        team = team_assigner.get_player_team(video_frames[frame_num],
                                              track['bbox'],
                                              player_id,
                                              is_goalkeeper=track.get('is_goalkeeper', False))
        tracks['players'][frame_num][player_id]['team'] = team
        tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

print("\n=== After assign_team_color + get_player_team loop (pre-fallback, pre-merge) ===")
for pid in CHECK_PIDS:
    frames_present = [fn for fn, fd in enumerate(tracks['players']) if pid in fd]
    teams_seen = [tracks['players'][fn][pid]['team'] for fn in frames_present]
    from collections import Counter
    print(f"pid {pid}: team value counts across its {len(frames_present)} frames: {Counter(teams_seen)}")

fallback_assignments = team_assigner.finalize_fallback_assignments()
print(f"\nfallback_assignments: {fallback_assignments}")
if fallback_assignments:
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id in player_track:
            if player_id in fallback_assignments:
                team = fallback_assignments[player_id]
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
                tracks['players'][frame_num][player_id]['team_fallback'] = True

merged_pairs, rejected_pairs = team_assigner.merge_fragmented_tracks(video_frames, tracks['players'])
print(f"\nmerged_pairs: {merged_pairs}")
print(f"rejected_pairs: {rejected_pairs}")

print("\n=== FINAL (post-fallback, post-merge) — matches what main.py hands to render_output1 ===")
for pid in CHECK_PIDS:
    frames_present = [fn for fn, fd in enumerate(tracks['players']) if pid in fd]
    from collections import Counter
    teams_seen = [tracks['players'][fn][pid]['team'] for fn in frames_present]
    fallback_flag = any(tracks['players'][fn][pid].get('team_fallback', False) for fn in frames_present)
    merged_flag = any(tracks['players'][fn][pid].get('team_merged', False) for fn in frames_present)
    print(f"pid {pid}: FINAL team value counts: {Counter(teams_seen)} "
          f"team_fallback={fallback_flag} team_merged={merged_flag}")

print(f"\nplayer_team_dict entries for {CHECK_PIDS}: "
      f"{ {p: team_assigner.player_team_dict.get(p) for p in CHECK_PIDS} }")
print(f"locked_colors keys present: { {p: (p in team_assigner.locked_colors) for p in CHECK_PIDS} }")
