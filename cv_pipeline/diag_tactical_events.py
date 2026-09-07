"""
diag_tactical_events.py — read-only diagnostic for
tactical_events/tactical_events_detector.py's 10 event types.

Standalone, additive. Does NOT modify tactical_events_detector.py,
speed_and_distance_estimator.py, main.py, render_output*.py, fast_setup.py,
or fast_render_*.py — only imports and runs them exactly as main.py does,
against the full 42,475-frame fastpipe match, to see what they actually
produce on real data. No fixes applied here.

Reused as-is: TacticalEventsDetector (the actual class main.py runs, with
main.py's actual fps=24 argument), SpeedAndDistance_Estimator (the actual
speed/distance computation main.py runs), tactical_events.space_control's
grid functions, and render_output3.PITCH_VERTS/STEP (the actual pitch
polygon main.py's precompute step uses).
"""

import math
from collections import defaultdict

import numpy as np

from analyze_formation_window import load_window_tracks, assign_teams_for_window
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from tactical_events.space_control import build_pitch_mask, build_sampling_grid, compute_space_control_per_frame
from tactical_events import tactical_events_detector as TED
from render_output3 import PITCH_VERTS, STEP

SOURCE = 'fastpipe'
START, END = 0, 42475
DETECTOR_FPS = 24   # matches main.py's actual `TacticalEventsDetector(fps=24)` call, not the real 25fps
N_EXAMPLES = 5

EVENT_TYPES = ('SPRINT', 'BURST', 'PRESS', 'RECOVERY', 'OVERLAP',
               'SPACE', 'SWITCH', 'DROP', 'BREAK', 'ISOLATED')


def main():
    print(f"Loading tracks [{START}:{END}) source='{SOURCE}' ...")
    tracks, video_path, real_fps, start_frame, end_frame = load_window_tracks(START, END, source=SOURCE)
    n = end_frame - start_frame
    print(f"  {n} frames, real video fps={real_fps} (detector will use fps={DETECTOR_FPS}, "
          f"matching main.py's actual hardcoded call)")

    print(f"Streaming {video_path} once for jersey-colour team assignment...")
    assign_teams_for_window(tracks, video_path, start_frame, end_frame)

    print("Computing speed/distance via SpeedAndDistance_Estimator (unmodified, as main.py runs it)...")
    sde = SpeedAndDistance_Estimator(fps=real_fps)
    sde.add_speed_and_distance_to_tracks(tracks)

    print("Computing space-control grid (render_output3.PITCH_VERTS/STEP, unmodified)...")
    h, w = 1080, 1920
    pitch_mask = build_pitch_mask(PITCH_VERTS, h, w)
    grid, grid_rows, grid_cols, row_idx, col_idx = build_sampling_grid(pitch_mask, STEP)
    space_control_per_frame = compute_space_control_per_frame(tracks, grid, top_frac=0.10)

    print(f"Running TacticalEventsDetector(fps={DETECTOR_FPS}).detect() — unmodified ...")
    detector = TED.TacticalEventsDetector(fps=DETECTOR_FPS)
    events_by_frame = detector.detect(tracks, space_control_per_frame)

    print("\nEvent counts (full match):")
    for et in EVENT_TYPES:
        print(f"  {et:10s}: {detector.event_counts.get(et, 0)}")

    # ── Reproduce the detector's own attack_dir computation for reporting ──
    all_pids = sorted({pid for frame in tracks['players'] for pid in frame})
    pos = {pid: np.full((n, 2), np.nan, dtype=np.float64) for pid in all_pids}
    spd = {pid: np.full(n, np.nan, dtype=np.float64) for pid in all_pids}
    team_of = {}
    for fn, player_data in enumerate(tracks['players']):
        for pid, info in player_data.items():
            p = TED._valid_pos(info.get('position_transformed'))
            if p is not None:
                pos[pid][fn] = p
            s = info.get('speed')
            if s is not None:
                try:
                    spd[pid][fn] = float(s)
                except Exception:
                    pass
            t = info.get('team', 0)
            if t in (1, 2) and pid not in team_of:
                team_of[pid] = t

    team_x = {1: [], 2: []}
    n_fragments = {1: 0, 2: 0}
    frame_weighted_sum = {1: 0.0, 2: 0.0}
    frame_weighted_n = {1: 0, 2: 0}
    for pid in all_pids:
        t = team_of.get(pid)
        if t not in (1, 2):
            continue
        xs = pos[pid][:, 0]
        xs = xs[~np.isnan(xs)]
        if len(xs):
            team_x[t].append(xs.mean())
            n_fragments[t] += 1
            frame_weighted_sum[t] += xs.sum()
            frame_weighted_n[t] += len(xs)
    mean_x = {t: (float(np.mean(team_x[t])) if team_x[t] else 52.5) for t in (1, 2)}
    fw_mean_x = {t: (frame_weighted_sum[t] / frame_weighted_n[t] if frame_weighted_n[t] else 52.5)
                 for t in (1, 2)}
    if mean_x[1] <= mean_x[2]:
        attack_dir = {1: 1.0, 2: -1.0}
    else:
        attack_dir = {1: -1.0, 2: 1.0}

    print(f"\nAttack-direction computation (unweighted per-fragment mean, as the detector itself uses):")
    print(f"  team 1: {n_fragments[1]} fragments, unweighted mean_x={mean_x[1]:.2f}  "
          f"(frame-weighted mean_x={fw_mean_x[1]:.2f} for comparison)")
    print(f"  team 2: {n_fragments[2]} fragments, unweighted mean_x={mean_x[2]:.2f}  "
          f"(frame-weighted mean_x={fw_mean_x[2]:.2f} for comparison)")
    print(f"  -> attack_dir = {attack_dir}  "
          f"(team {'1' if attack_dir[1]>0 else '2'} attacks toward x=105, "
          f"team {'2' if attack_dir[2]>0 else '1'} attacks toward x=0)")
    print(f"  Cross-check against danger-score full-match runs (independent team-color fit, "
          f"'team1 defends x=105, team2 defends x=0' was the repeated finding there):")

    # ── Flatten events by type ──────────────────────────────────────────────
    events_by_type = defaultdict(list)
    for fn, evs in events_by_frame.items():
        for e in evs:
            events_by_type[e['type']].append(e)

    def t_str(fn):
        t = fn / real_fps
        m, s = divmod(t, 60)
        h_, m = divmod(int(m), 60)
        return f"{h_:02d}:{m:02d}:{s:05.2f}"

    def pinfo(fn, pid):
        info = tracks['players'][fn].get(pid, {})
        return info

    def sample(etype, k=N_EXAMPLES):
        evs = sorted(events_by_type[etype], key=lambda e: e['frame'])
        if len(evs) <= k:
            return evs
        # spread across the match rather than all from the first minute
        step = len(evs) / k
        return [evs[int(i * step)] for i in range(k)]

    print("\n" + "=" * 78)
    print("PER-EVENT-TYPE DIAGNOSTIC")
    print("=" * 78)

    # ---- SPRINT ----
    print(f"\n--- SPRINT ---  ({detector.event_counts.get('SPRINT',0)} total)")
    print(f"Trigger: speed > {TED.SPRINT_KMH} km/h sustained for >= {TED.SPRINT_MIN_FRAMES} "
          f"consecutive frames (rising edge only, {detector.cooldown_frames}f cooldown per player)")
    for e in sample('SPRINT'):
        fn, pid = e['frame'], e['player_id']
        info = pinfo(fn, pid)
        print(f"  frame={fn} t={t_str(fn)} pid={pid} team={team_of.get(pid)} "
              f"speed={info.get('speed'):.1f}km/h pos={info.get('position_transformed')}")

    # ---- BURST ----
    print(f"\n--- BURST ---  ({detector.event_counts.get('BURST',0)} total)")
    w_burst = max(1, int(round(TED.BURST_WINDOW_SEC * DETECTOR_FPS)))
    print(f"Trigger: speed increase > {TED.BURST_DELTA_KMH} km/h over {TED.BURST_WINDOW_SEC}s "
          f"({w_burst} frames @ fps={DETECTOR_FPS}), sustained >= {TED.BURST_MIN_FRAMES} consecutive frames")
    for e in sample('BURST'):
        fn, pid = e['frame'], e['player_id']
        info = pinfo(fn, pid)
        fn0 = fn - w_burst
        info0 = pinfo(fn0, pid) if fn0 >= 0 else {}
        s0 = info0.get('speed')
        s1 = info.get('speed')
        delta = (s1 - s0) if (s0 is not None and s1 is not None) else None
        print(f"  frame={fn} t={t_str(fn)} pid={pid} team={team_of.get(pid)}  "
              f"speed[t-{w_burst}f]={s0}  speed[t]={s1}  delta={delta}")

    # ---- PRESS ----
    print(f"\n--- PRESS ---  ({detector.event_counts.get('PRESS',0)} total)")
    print(f"Trigger: >=2 same-team players within {TED.PRESS_DIST_M}m of the SAME opponent, "
          f"all pressers moving >= {TED.PRESS_MIN_KMH}km/h")
    for e in sample('PRESS'):
        fn, pid = e['frame'], e['player_id']
        info = pinfo(fn, pid)
        p = info.get('position_transformed')
        t = team_of.get(pid)
        # find which opponent(s) they were pressing, for context
        opp_team = 2 if t == 1 else 1
        nearest = []
        for opid, oinfo in tracks['players'][fn].items():
            if team_of.get(opid) != opp_team:
                continue
            op = oinfo.get('position_transformed')
            if op is None or p is None:
                continue
            d = math.hypot(p[0]-op[0], p[1]-op[1])
            nearest.append((d, opid))
        nearest.sort()
        near_txt = f"nearest opponent {nearest[0][1]} at {nearest[0][0]:.1f}m" if nearest else "n/a"
        print(f"  frame={fn} t={t_str(fn)} pid={pid} team={t} speed={info.get('speed'):.1f}km/h "
              f"pos={p}  {near_txt}")

    # ---- RECOVERY ----
    print(f"\n--- RECOVERY ---  ({detector.event_counts.get('RECOVERY',0)} total)")
    print(f"Trigger: speed >= {TED.RECOVERY_KMH}km/h AND moving toward OWN goal "
          f"(dx * attack_dir[team] < 0), sustained >= {TED.RECOVERY_MIN_FRAMES} consecutive frames")
    for e in sample('RECOVERY'):
        fn, pid = e['frame'], e['player_id']
        info = pinfo(fn, pid)
        t = team_of.get(pid)
        prev = pinfo(fn-1, pid) if fn >= 1 else {}
        p, pprev = info.get('position_transformed'), prev.get('position_transformed')
        dx = (p[0]-pprev[0]) if (p and pprev) else None
        print(f"  frame={fn} t={t_str(fn)} pid={pid} team={t} attack_dir={attack_dir.get(t)}  "
              f"speed={info.get('speed'):.1f}km/h  x(t-1)={pprev[0] if pprev else None}  "
              f"x(t)={p[0] if p else None}  dx={dx}  "
              f"(dx*attack_dir={dx*attack_dir.get(t,0) if dx is not None else None})")

    # ---- OVERLAP ----
    print(f"\n--- OVERLAP ---  ({detector.event_counts.get('OVERLAP',0)} total)")
    print(f"Trigger: same-team pair within {TED.OVERLAP_FLANK_TOL_M}m lateral (y) tolerance, "
          f"both moving forward (attack_dir) at >= {TED.OVERLAP_MIN_KMH}km/h, "
          f"crosses from behind to ahead along x*attack_dir")
    for e in sample('OVERLAP'):
        fn, pid = e['frame'], e['player_id']
        info = pinfo(fn, pid)
        t = team_of.get(pid)
        print(f"  frame={fn} t={t_str(fn)} pid={pid}(crosser) team={t} attack_dir={attack_dir.get(t)} "
              f"speed={info.get('speed'):.1f}km/h pos={info.get('position_transformed')}")

    # ---- SPACE ----
    print(f"\n--- SPACE ---  ({detector.event_counts.get('SPACE',0)} total)")
    print(f"Trigger: player is in the top 10% of grid-cell space control this frame "
          f"(precomputed by tactical_events.space_control)")
    for e in sample('SPACE'):
        fn, pid = e['frame'], e['player_id']
        info = pinfo(fn, pid)
        sc = space_control_per_frame[fn].get(pid) if fn < len(space_control_per_frame) else None
        print(f"  frame={fn} t={t_str(fn)} pid={pid} team={team_of.get(pid)} "
              f"cells={sc.get('cells') if sc else None} frac={sc.get('frac') if sc else None} "
              f"rank={sc.get('rank') if sc else None}")

    # ---- SWITCH ----
    print(f"\n--- SWITCH ---  ({detector.event_counts.get('SWITCH',0)} total)")
    w_switch = max(1, int(round(TED.SWITCH_WINDOW_SEC * DETECTOR_FPS)))
    print(f"Trigger: |y(t) - y(t-{w_switch}f)| > {TED.SWITCH_DIST_M}m over {TED.SWITCH_WINDOW_SEC}s window "
          f"(no minimum speed, no direction/attack_dir check at all)")
    for e in sample('SWITCH'):
        fn, pid = e['frame'], e['player_id']
        info = pinfo(fn, pid)
        fn0 = fn - w_switch
        info0 = pinfo(fn0, pid) if fn0 >= 0 else {}
        p, p0 = info.get('position_transformed'), info0.get('position_transformed')
        dy = (p[1]-p0[1]) if (p and p0) else None
        print(f"  frame={fn} t={t_str(fn)} pid={pid} team={team_of.get(pid)} "
              f"y(t-{w_switch}f)={p0[1] if p0 else None}  y(t)={p[1] if p else None}  dy={dy}")

    # ---- DROP ----
    print(f"\n--- DROP ---  ({detector.event_counts.get('DROP',0)} total)")
    w_drop = max(1, int(round(TED.DROP_WINDOW_SEC * DETECTOR_FPS)))
    print(f"Trigger: net x-displacement > {TED.DROP_DIST_M}m toward OWN half over "
          f"{TED.DROP_WINDOW_SEC}s ({w_drop}f), AND >= {TED.DROP_MAJORITY_FRAC*100:.0f}% of "
          f"per-frame steps in that window are backward")
    for e in sample('DROP'):
        fn, pid = e['frame'], e['player_id']
        info = pinfo(fn, pid)
        t = team_of.get(pid)
        fn0 = fn - w_drop
        info0 = pinfo(fn0, pid) if fn0 >= 0 else {}
        p, p0 = info.get('position_transformed'), info0.get('position_transformed')
        net = (p[0]-p0[0])*attack_dir.get(t,1) if (p and p0) else None
        print(f"  frame={fn} t={t_str(fn)} pid={pid} team={t} attack_dir={attack_dir.get(t)}  "
              f"x(t-{w_drop}f)={p0[0] if p0 else None}  x(t)={p[0] if p else None}  "
              f"net(toward own half)={net}")

    # ---- BREAK ----
    print(f"\n--- BREAK ---  ({detector.event_counts.get('BREAK',0)} total)")
    print(f"Trigger: speed >= {TED.BREAK_KMH}km/h AND inside attacking final third "
          f"(FINAL_THIRD_M={TED.FINAL_THIRD_M}), sustained >= {TED.BREAK_MIN_FRAMES} consecutive frames")
    for e in sample('BREAK'):
        fn, pid = e['frame'], e['player_id']
        info = pinfo(fn, pid)
        t = team_of.get(pid)
        p = info.get('position_transformed')
        thresh = (TED.PITCH_X_MAX - TED.FINAL_THIRD_M) if attack_dir.get(t,1) > 0 else TED.FINAL_THIRD_M
        print(f"  frame={fn} t={t_str(fn)} pid={pid} team={t} attack_dir={attack_dir.get(t)}  "
              f"speed={info.get('speed'):.1f}km/h  x={p[0] if p else None}  "
              f"final-third boundary={thresh:.1f}")

    # ---- ISOLATED ----
    print(f"\n--- ISOLATED ---  ({detector.event_counts.get('ISOLATED',0)} total)")
    print(f"Trigger: nearest SAME-TEAM teammate (any location on pitch, no wide-channel/zone "
          f"restriction found in the code) is > {TED.ISOLATED_DIST_M}m away")
    for e in sample('ISOLATED'):
        fn, pid = e['frame'], e['player_id']
        info = pinfo(fn, pid)
        t = team_of.get(pid)
        p = info.get('position_transformed')
        nearest = []
        for opid, oinfo in tracks['players'][fn].items():
            if opid == pid or team_of.get(opid) != t:
                continue
            op = oinfo.get('position_transformed')
            if op is None or p is None:
                continue
            d = math.hypot(p[0]-op[0], p[1]-op[1])
            nearest.append((d, opid))
        nearest.sort()
        near_txt = f"nearest teammate {nearest[0][1]} at {nearest[0][0]:.1f}m" if nearest else "NO valid teammates this frame"
        print(f"  frame={fn} t={t_str(fn)} pid={pid} team={t} pos={p}  {near_txt}")

    print("\nDone.")


if __name__ == '__main__':
    main()
