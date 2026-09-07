"""
test_team_classification.py — empirical before/after test of the
goalkeeper-exclusion fix in team_assigner.py, on real player-track data
from both pipelines.

"Before" is simulated by re-running the OLD clustering behavior (include
goalkeeper samples in the 2-team KMeans fit, force-fit every player
including keepers against those 2 centroids) using the same underlying
color samples the NEW code collects — so before/after are directly
comparable on identical crops, not re-detected separately.

"After" is the real, current team_assigner.py behavior (goalkeepers
excluded from clustering, assigned team=3 directly).

Also reports CONFIDENCE_THRESHOLD diagnostics: how many player track_ids
never resolve (team stays 0 for their entire tracked lifetime), and how
close their best-ever color sample came to the threshold.

Standalone: does not modify or import main.py.
"""
import json
import pickle
import time

# `trackers` pulls in torch (via ultralytics). On this machine, importing
# sklearn (MKL BLAS) before torch reliably crashes torch's DLL init
# (WinError 1114) — import order matters here, so torch goes first.
from trackers import Tracker

import numpy as np
from sklearn.cluster import KMeans

from team_assigner.team_assigner import TeamAssigner, CONFIDENCE_THRESHOLD, CLUSTER_FRAMES

RESULTS_PATH = 'team_classification_test_results.json'


def stream_window(video_path, start_frame, end_frame):
    import cv2
    frames = []
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def run_real_detection(video_frames, label):
    """Real Tracker.get_object_tracks (full-res, same as main.py), fresh —
    fresh detection so is_goalkeeper is populated (old cached stubs predate
    that flag)."""
    print(f"\n[{label}] running real detection on {len(video_frames)} frames...")
    t0 = time.time()
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path=None)
    print(f"[{label}] detection done in {time.time()-t0:.1f}s")
    return tracks


def analyze(video_frames, tracks, label):
    players = tracks['players']

    # ---- gather ALL color samples + goalkeeper flags across CLUSTER_FRAMES,
    # exactly like team_assigner.assign_team_color does, but keep the flag
    # so we can build BOTH the old (polluted) and new (clean) fits from the
    # same underlying data ----
    ta_probe = TeamAssigner()
    sample_fns = [fn for fn in CLUSTER_FRAMES if fn < len(video_frames) and fn < len(players)]
    all_samples = []   # (color, is_goalkeeper)
    for fn in sample_fns:
        for _, det in players[fn].items():
            bbox = det.get('bbox')
            if bbox is None:
                continue
            c = ta_probe.get_player_color(video_frames[fn], bbox)
            all_samples.append((c, bool(det.get('is_goalkeeper', False))))

    n_gk_in_sample = sum(1 for _, gk in all_samples if gk)
    print(f"[{label}] {len(all_samples)} color samples from frames {sample_fns}, "
          f"{n_gk_in_sample} flagged goalkeeper")

    # ---- OLD behavior: fit KMeans(3) on ALL samples including goalkeepers,
    # take the two largest clusters as the two teams (this is exactly what
    # assign_team_color used to do before the fix) ----
    colors_all = np.array([c for c, _ in all_samples], dtype=np.float32)
    old_result = {}
    if len(colors_all) >= 3:
        km_old = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=0)
        km_old.fit(colors_all)
        counts = np.bincount(km_old.labels_, minlength=3)
        sorted_idx = np.argsort(counts)[::-1]
        old_t1 = km_old.cluster_centers_[sorted_idx[0]]
        old_t2 = km_old.cluster_centers_[sorted_idx[1]]
        old_sep = float(np.linalg.norm(old_t1 - old_t2))
        old_result = {'team1_centroid': old_t1.tolist(), 'team2_centroid': old_t2.tolist(),
                      'centroid_separation': old_sep}

        # how would OLD code classify each real goalkeeper sample?
        gk_old_assignments = []
        for c, is_gk in all_samples:
            if not is_gk:
                continue
            d1 = float(np.linalg.norm(c - old_t1))
            d2 = float(np.linalg.norm(c - old_t2))
            team = 1 if d1 <= d2 else 2
            conf = min(d1, d2) < CONFIDENCE_THRESHOLD
            gk_old_assignments.append({'assigned_team': team, 'distance': min(d1, d2),
                                        'within_threshold': conf})
        old_result['goalkeeper_force_fit'] = gk_old_assignments

    # ---- NEW behavior: real team_assigner.py, goalkeepers excluded from fit ----
    ta_new = TeamAssigner()
    ta_new.assign_team_color(video_frames, players)
    new_t1, new_t2 = ta_new.locked_colors.get(1), ta_new.locked_colors.get(2)
    new_sep = float(np.linalg.norm(new_t1 - new_t2)) if new_t1 is not None and new_t2 is not None else None

    # run REAL get_player_team across every frame/track (mirrors main.py's loop)
    resolved_teams = {}       # pid -> final team (1/2/3/0)
    ever_confident = {}       # pid -> True once ever resolved to 1/2/3
    best_distance = {}        # pid -> best (min) color distance ever seen, for never-resolved diagnosis
    n_frames_total = 0
    n_gk_frames = 0
    gk_pids = set()

    for fn, frame_players in enumerate(players):
        for pid, det in frame_players.items():
            n_frames_total += 1
            is_gk = bool(det.get('is_goalkeeper', False))
            if is_gk:
                n_gk_frames += 1
                gk_pids.add(pid)
            team = ta_new.get_player_team(video_frames[fn], det['bbox'], pid, is_goalkeeper=is_gk)
            resolved_teams[pid] = team
            if team != 0:
                ever_confident[pid] = True
            elif not is_gk:
                # track best (smallest) distance for still-unresolved players for diagnostics
                color = ta_probe.get_player_color(video_frames[fn], det['bbox'])
                if new_t1 is not None and new_t2 is not None:
                    d = min(float(np.linalg.norm(color - new_t1)), float(np.linalg.norm(color - new_t2)))
                    best_distance[pid] = min(best_distance.get(pid, 1e9), d)

    unresolved_pids = [pid for pid, team in resolved_teams.items() if team == 0]
    all_pids = list(resolved_teams.keys())

    resolvable_with_higher_threshold = sum(
        1 for pid in unresolved_pids if pid in best_distance and best_distance[pid] < 60.0)

    result = {
        'label': label,
        'n_frames': len(video_frames),
        'n_player_track_ids_total': len(all_pids),
        'n_goalkeeper_track_ids': len(gk_pids),
        'n_goalkeeper_detections_total': n_gk_frames,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'new_team1_centroid': new_t1.tolist() if new_t1 is not None else None,
        'new_team2_centroid': new_t2.tolist() if new_t2 is not None else None,
        'new_centroid_separation': new_sep,
        'old_behavior_centroid_separation_with_gk_polluting_fit': old_result.get('centroid_separation'),
        'old_behavior_goalkeeper_force_fit_examples': old_result.get('goalkeeper_force_fit'),
        'n_unresolved_pids_final': len(unresolved_pids),
        'unresolved_pids': unresolved_pids,
        'unresolved_best_distances': {str(k): v for k, v in best_distance.items()},
        'n_unresolved_within_60_of_threshold': resolvable_with_higher_threshold,
        'final_team_distribution': {str(t): sum(1 for v in resolved_teams.values() if v == t)
                                     for t in (0, 1, 2, 3)},
    }
    return result


def main():
    all_results = {}

    # ── Short clip: full real fresh detection ──────────────────────────────
    short_frames = stream_window('Match_videos/121364_0.mp4', 0, 750)
    short_tracks = run_real_detection(short_frames, 'short_clip')
    all_results['short_clip'] = analyze(short_frames, short_tracks, 'short_clip')
    print(json.dumps(all_results['short_clip'], indent=2, default=str))

    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote partial results (short_clip) to {RESULTS_PATH}")

    # ── Long clip: real fresh detection on a 750-frame window (fair size
    # comparison to the short clip), full-res Tracker for apples-to-apples
    # color fidelity (see report for the fast-pipeline half-res caveat) ────
    long_frames = stream_window('Sample_Videos/LiverpoolPSG_short.mp4', 33475, 34225)
    long_tracks = run_real_detection(long_frames, 'long_clip_sample')
    all_results['long_clip_sample'] = analyze(long_frames, long_tracks, 'long_clip_sample')
    print(json.dumps(all_results['long_clip_sample'], indent=2, default=str))

    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote full results to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
