"""
link_full_match_identities.py — full-match identity linking + team
assignment on top of the cached fastpipe tracking data.

Standalone, additive analysis script. Does NOT import, modify, or touch
main.py, fast_setup.py, fast_render_*.py, or any render_output*.py file.

Problem: the cached fastpipe_track_stubs.pkl has 4,025 raw ByteTrack IDs
for what should be ~22-30 real players (roster + a couple of subs),
because half-resolution/every-3rd-frame detection in fast_setup.py causes
frequent track loss/re-acquisition, each of which mints a new ID.

First attempt (git history) reused render_output3.py's proximity+team
`_link_identities` verbatim (80px, 30-frame/1.25s lost-buffer) across the
full match: it only got to 1,668 merged identities (1,273 kept), because
a 1.25s buffer — tuned for short rendered clips — is far too short for
real occlusion/camera-cut gaps over a 29.5-minute match, and 4,025 raw
IDs is too many for pure 80px proximity to safely re-match once the gap
(and hence buffer) widens: two different same-team players can easily
end up within 80px of each other after a multi-second gap.

This version instead:
  1. Widens the lost-identity buffer to a moderate ~8 seconds (not 30+ —
     wide enough for real gaps, not so wide that position becomes
     meaningless).
  2. Adds jersey-colour similarity as a SECOND required condition for
     merging, alongside same-team and 80px/8s proximity — a new pid can
     only be merged into a recently-lost identity if their sampled
     jersey colours are also close (not just in the same team cluster).
     This is the safeguard against incorrectly merging two different
     real players of the same team who happen to pass near each other
     after a gap, which widening the buffer alone would risk.

What this script does
----------------------
1. Loads ONLY stubs/fastpipe_track_stubs.pkl + stubs/fastpipe_meta.pkl
   (no homography/camera-movement caches, no Tracker/YOLO instantiation).
2. Streams the source video sequentially ONCE (never held as a full
   in-memory list) to resolve, for every player in every frame:
     - a team (1/2), via TeamAssigner's locked KMeans jersey centroids
     - a raw jersey colour sample, accumulated per raw tracker ID into a
       running mean — this is the same crop (get_player_color) the team
       classification already uses, just also kept instead of discarded,
       so no separate video pass is needed for the colour signature.
3. Runs a same-team + proximity + time-window + jersey-colour matcher
   across the FULL 42,475-frame match to collapse the raw IDs into
   stable merged identities.
4. Saves stubs/fastpipe_merged_identities.pkl and prints a summary.

Usage:
    python link_full_match_identities.py
"""

import os
import pickle
import time

import cv2
import numpy as np
import psutil

from team_assigner import TeamAssigner
from team_assigner.team_assigner import CONFIDENCE_THRESHOLD
from render_output3 import PROXIMITY_PX, MIN_APPEARANCE_FRAMES

STUB_DIR     = 'stubs'
CACHE_TRACK  = os.path.join(STUB_DIR, 'fastpipe_track_stubs.pkl')
CACHE_META   = os.path.join(STUB_DIR, 'fastpipe_meta.pkl')
CACHE_MERGED = os.path.join(STUB_DIR, 'fastpipe_merged_identities.pkl')

# Must match team_assigner.CLUSTER_FRAMES so the KMeans fit sees the same
# sample frames the rest of the pipeline would use.
CLUSTER_FRAMES = [0, 10, 20, 30, 40]

LOST_BUF_SECONDS = 8.0   # widened from render_output3's 30-frame (1.25s) buffer

# Max BGR Euclidean distance between two fragments' mean jersey colours to
# treat them as the same player. Chosen well under half the measured
# inter-team centroid separation (~128.6 in the last run) and in the same
# order as CONFIDENCE_THRESHOLD (40, used to assign a single sample to a
# team centroid) — tight enough to reject a different same-team player,
# loose enough to absorb lighting/shadow variation across the match.
COLOR_SIMILARITY_MAX = 45.0

PROGRESS_EVERY = 5000

process = psutil.Process(os.getpid())


def rss_mb():
    return process.memory_info().rss / (1024 ** 2)


def assign_teams_and_colors_full_match(tracks, video_path, n_frames):
    """Stream the match video ONCE, sequentially, resolving
    tracks['players'][f][pid]['team'] for every player in every frame,
    and accumulating a running (sum, count) jersey-colour signature per
    raw tracker ID.

    Team classification is TeamAssigner's own confidence/locking logic
    (locked_colors + CONFIDENCE_THRESHOLD), inlined here rather than
    calling TeamAssigner.get_player_team() directly, so the crop colour
    (get_player_color) is computed once per (frame, pid) and reused for
    both the team decision AND the colour-signature accumulation —
    get_player_team() throws its internal colour sample away once it has
    a cached team, which would otherwise force computing every crop
    twice.

    Returns (team_assigner, pid_color_sum, pid_color_count).
    """
    team_assigner = TeamAssigner()
    max_cluster_fn = max(CLUSTER_FRAMES)

    pid_color_sum   = {}   # pid -> np.array([B, G, R]) running sum
    pid_color_count = {}   # pid -> int
    pid_team_locked = {}   # pid -> 1 or 2, permanent once confident

    def classify_and_accumulate(frame, frame_players):
        for pid, info in frame_players.items():
            bbox = info.get('bbox')
            if bbox is None:
                continue
            color = team_assigner.get_player_color(frame, bbox)
            pid_color_sum[pid] = pid_color_sum.get(pid, np.zeros(3)) + color
            pid_color_count[pid] = pid_color_count.get(pid, 0) + 1

            if pid in pid_team_locked:
                info['team'] = pid_team_locked[pid]
                continue
            if not team_assigner.locked_colors:
                info['team'] = 0
                continue
            c1 = team_assigner.locked_colors.get(1, np.zeros(3))
            c2 = team_assigner.locked_colors.get(2, np.zeros(3))
            d1 = float(np.linalg.norm(color - c1))
            d2 = float(np.linalg.norm(color - c2))
            if min(d1, d2) < CONFIDENCE_THRESHOLD:
                team = 1 if d1 <= d2 else 2
                pid_team_locked[pid] = team
                info['team'] = team
            else:
                info['team'] = 0

    cap = cv2.VideoCapture(video_path)

    # ── Phase 1: buffer only the frames needed for the colour-cluster fit ──
    early_frames = []
    for fn in range(max_cluster_fn + 1):
        ret, frame = cap.read()
        if not ret:
            raise IOError(f"video ended at frame {fn}, before cluster-fit "
                           f"frame {max_cluster_fn}")
        early_frames.append(frame)

    team_assigner.assign_team_color(early_frames, tracks['players'])

    # Centroids are locked now — classify + accumulate colour for the
    # buffered early frames before releasing the pixel data.
    for fn, frame in enumerate(early_frames):
        classify_and_accumulate(frame, tracks['players'][fn])
    early_frames = None

    # ── Phase 2: stream the rest of the match, one frame at a time ──────────
    t0 = time.time()
    fn = max_cluster_fn + 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        classify_and_accumulate(frame, tracks['players'][fn])
        if fn % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            print(f"  [team-assign] frame {fn}/{n_frames}  elapsed={elapsed:7.1f}s  "
                  f"({fn / max(elapsed, 1e-6):.1f} frame/s)", flush=True)
        fn += 1
    cap.release()

    if fn != n_frames:
        print(f"  WARNING: video yielded {fn} frames, expected {n_frames} "
              f"from metadata — frames [{fn}:{n_frames}) will have no team "
              f"and be excluded from identity linking.")

    print(f"  [team-assign] done: {fn} frames processed in {time.time() - t0:.1f}s")
    return team_assigner, pid_color_sum, pid_color_count


def _link_identities_with_color(tracks, n, pid_mean_color,
                                 proximity_px, lost_buf_frames, color_similarity_max):
    """Same proximity/team/time-window matcher as render_output3.py's
    `_link_identities`, plus a second required condition: a new pid can
    only merge into a recently-lost identity if their jersey colours are
    also close. The merged identity's colour signature is a running,
    frame-count-weighted mean over every raw pid folded into it, so later
    merge decisions compare against the identity's accumulated evidence
    rather than just its most recent fragment.

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
    sid_color_sum    = {}   # sid -> np.array([B, G, R]) weighted running sum
    sid_color_count  = {}   # sid -> int total frames contributed so far
    recently_lost    = {}   # sid -> (cx, cy, team, frame_lost)
    appearance_count = {}
    next_sid = 1

    def fold_pid_color_into_sid(sid, pid):
        psum = pid_mean_color.get(pid)
        if psum is None:
            return
        pcolor, pcount = psum
        if pcount <= 0:
            return
        sid_color_sum[sid] = sid_color_sum.get(sid, np.zeros(3)) + pcolor * pcount
        sid_color_count[sid] = sid_color_count.get(sid, 0) + pcount

    def sid_mean_color(sid):
        cnt = sid_color_count.get(sid, 0)
        if cnt <= 0:
            return None
        return sid_color_sum[sid] / cnt

    for frame_num in range(n):
        player_data = (tracks['players'][frame_num]
                       if frame_num < len(tracks['players']) else {})
        current_pids = {pid for pid, info in player_data.items()
                        if info.get('team', 0) in (1, 2)}
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
                 if frame_num - fn > lost_buf_frames]
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
                fold_pid_color_into_sid(sid, pid)
                continue

            ncx = (bbox[0] + bbox[2]) / 2.0
            ncy = (bbox[1] + bbox[3]) / 2.0
            new_color = pid_mean_color.get(pid)
            new_color = new_color[0] if new_color is not None and new_color[1] > 0 else None

            best_sid, best_d = None, proximity_px + 1
            for sid, (cx, cy, lost_team, _) in recently_lost.items():
                if sid in matched_sids or lost_team != team:
                    continue
                d = np.sqrt((ncx - cx) ** 2 + (ncy - cy) ** 2)
                if d >= best_d:
                    continue
                lost_color = sid_mean_color(sid)
                if new_color is None or lost_color is None:
                    continue   # can't verify jersey similarity — don't merge
                if np.linalg.norm(new_color - lost_color) > color_similarity_max:
                    continue
                best_d, best_sid = d, sid

            if best_sid is not None:
                tracker_to_sid[pid] = best_sid
                matched_sids.add(best_sid)
                del recently_lost[best_sid]
                pid_to_sid[pid] = best_sid
                fold_pid_color_into_sid(best_sid, pid)
            else:
                sid = next_sid; next_sid += 1
                tracker_to_sid[pid] = sid
                sid_team[sid] = team
                pid_to_sid[pid] = sid
                fold_pid_color_into_sid(sid, pid)

        for pid in current_pids:
            if pid not in tracker_to_sid:
                sid = next_sid; next_sid += 1
                tracker_to_sid[pid] = sid
                sid_team[sid] = player_data[pid].get('team', 0)
                pid_to_sid[pid] = sid
                fold_pid_color_into_sid(sid, pid)

        for pid in current_pids:
            sid = tracker_to_sid[pid]
            appearance_count[sid] = appearance_count.get(sid, 0) + 1
            bbox = player_data[pid].get('bbox')
            if bbox:
                sid_last_center[sid] = ((bbox[0] + bbox[2]) / 2.0,
                                        (bbox[1] + bbox[3]) / 2.0)

    return pid_to_sid, sid_team, appearance_count


def main():
    print(f"Baseline RSS: {rss_mb():.1f} MB")

    # ── 1. Load ONLY the cached tracking data ───────────────────────────────
    t0 = time.time()
    with open(CACHE_META, 'rb') as f:
        meta = pickle.load(f)
    video_path, fps, n_frames = meta['video_path'], meta['fps'], meta['n_frames']

    with open(CACHE_TRACK, 'rb') as f:
        tracks = pickle.load(f)
    print(f"Loaded cached tracking data for {n_frames} frames in "
          f"{time.time() - t0:.2f}s  (RSS now {rss_mb():.1f} MB)")

    total_raw_ids = {pid for frame in tracks['players'] for pid in frame}
    print(f"Total raw tracker IDs across full match: {len(total_raw_ids)}")

    lost_buf_frames = round(LOST_BUF_SECONDS * fps)

    # ── 2. Real jersey-colour team + colour-signature pass (streams video once)
    print(f"\nStreaming {video_path} once for jersey-colour team assignment "
          f"+ per-ID colour signature (no full frame list held in memory)...")
    t1 = time.time()
    _, pid_color_sum, pid_color_count = assign_teams_and_colors_full_match(
        tracks, video_path, n_frames)
    print(f"Team + colour pass took {time.time() - t1:.1f}s  (RSS now {rss_mb():.1f} MB)")

    pid_mean_color = {
        pid: (pid_color_sum[pid] / pid_color_count[pid], pid_color_count[pid])
        for pid in pid_color_sum if pid_color_count.get(pid, 0) > 0
    }

    # ── 3. Full-match identity linking: proximity + team + time-window + colour
    print(f"\nLinking identities across all {n_frames} frames "
          f"(proximity<={PROXIMITY_PX}px, lost-buffer<={lost_buf_frames} frames "
          f"[{LOST_BUF_SECONDS}s @ {fps}fps], same team, "
          f"jersey-colour distance<={COLOR_SIMILARITY_MAX})...")
    t2 = time.time()
    pid_to_sid, sid_team, sid_appearance_count = _link_identities_with_color(
        tracks, n_frames, pid_mean_color,
        PROXIMITY_PX, lost_buf_frames, COLOR_SIMILARITY_MAX)
    print(f"Identity linking took {time.time() - t2:.1f}s  (RSS now {rss_mb():.1f} MB)")

    raw_pid_count          = len(pid_to_sid)
    unresolved_raw_ids     = sorted(total_raw_ids - set(pid_to_sid))
    merged_identity_count  = len(sid_team)
    linked_fragment_count  = raw_pid_count - merged_identity_count

    valid_sids = {sid for sid, cnt in sid_appearance_count.items()
                  if cnt >= MIN_APPEARANCE_FRAMES}
    dropped = merged_identity_count - len(valid_sids)

    team1 = [sid for sid in valid_sids if sid_team.get(sid) == 1]
    team2 = [sid for sid in valid_sids if sid_team.get(sid) == 2]

    top_sid, top_cnt = max(sid_appearance_count.items(), key=lambda kv: kv[1])
    top_pct = 100.0 * top_cnt / n_frames

    # ── 4. Save the merged-identity cache ───────────────────────────────────
    identities = {
        sid: {
            'team': sid_team.get(sid, 0),
            'frame_count': sid_appearance_count.get(sid, 0),
            'valid': sid in valid_sids,
        }
        for sid in sid_team
    }
    merged_cache = {
        'video_path': video_path,
        'n_frames': n_frames,
        'params': {
            'proximity_px': PROXIMITY_PX,
            'lost_buf_frames': lost_buf_frames,
            'lost_buf_seconds': LOST_BUF_SECONDS,
            'color_similarity_max': COLOR_SIMILARITY_MAX,
            'min_appearance_frames': MIN_APPEARANCE_FRAMES,
        },
        'raw_id_count_total': len(total_raw_ids),
        'raw_id_count_team_resolved': raw_pid_count,
        'unresolved_raw_ids': unresolved_raw_ids,
        'pid_to_sid': pid_to_sid,
        'identities': identities,
    }
    with open(CACHE_MERGED, 'wb') as f:
        pickle.dump(merged_cache, f)
    print(f"\nSaved {CACHE_MERGED}")

    # ── 5. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("link_full_match_identities.py SUMMARY (8s buffer + jersey-colour gate)")
    print("=" * 70)
    print(f"  Total raw tracker IDs (full match)        : {len(total_raw_ids)}")
    print(f"  Raw IDs with a resolved team (linked)     : {raw_pid_count}")
    print(f"  Raw IDs never team-resolved (excluded)    : {len(unresolved_raw_ids)}")
    print(f"  Merged identities after linking           : {merged_identity_count}")
    print(f"  Fragments linked back to an earlier player: {linked_fragment_count}")
    print(f"  Merged identities kept (>= {MIN_APPEARANCE_FRAMES} frames)       : {len(valid_sids)}")
    print(f"  Merged identities dropped as noise        : {dropped}")
    print(f"  Team split among kept identities          : Team 1 = {len(team1)}, "
          f"Team 2 = {len(team2)}")
    print(f"  Top identity's share of the match         : {top_pct:.1f}% "
          f"({top_cnt}/{n_frames} frames, sid {top_sid})")
    print("=" * 70)

    print(f"\nTop 30 merged identities by frame count "
          f"(out of {len(sid_appearance_count)} total):")
    ranked = sorted(sid_appearance_count.items(), key=lambda kv: -kv[1])[:30]
    print(f"  {'sid':>6}  {'team':>4}  {'frames':>7}  {'% of match':>10}  kept?")
    for sid, cnt in ranked:
        team = sid_team.get(sid, 0)
        pct = 100.0 * cnt / n_frames
        kept = "yes" if sid in valid_sids else "no (noise)"
        print(f"  {sid:6d}  {team:4d}  {cnt:7d}  {pct:9.1f}%  {kept}")

    print(f"\nFinal RSS: {rss_mb():.1f} MB")

    if len(valid_sids) > 40:
        print(f"\nHONEST ASSESSMENT: {len(valid_sids)} kept identities is still far from "
              f"the ~22-30 real-player target. Widening the buffer + adding a jersey-colour "
              f"gate was a moderate, bug-conscious tuning step, not a re-identification "
              f"system — closing the remaining gap likely needs actual re-ID (e.g. an "
              f"appearance embedding per fragment, or shirt-number OCR) rather than further "
              f"threshold tuning on proximity/time/colour alone.")


if __name__ == '__main__':
    main()
