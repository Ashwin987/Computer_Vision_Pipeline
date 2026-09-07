"""
tactical_events/tactical_events_detector.py

Position/movement-based tactical event detection — replaces the ball-based
event log that used to drive render_output1.py's panel. Computed ONCE, in
a single pass over the whole match (called from main.py, before any
rendering), and stored keyed by frame number so render_output1.py only
looks events up; it never recomputes anything.

Inputs used (all already computed earlier in main.py's pipeline):
  - position_transformed  : world position in metres (0-105 x 0-68)
  - speed                 : smoothed km/h (3-frame rolling average)
  - team                  : 1 or 2
  - space_control_per_frame: precomputed ONCE by
    tactical_events.space_control.compute_space_control_per_frame — reused
    here for the SPACE event rather than recomputed.

Threshold checks only, no new models, no ball data.

Rising-edge (onset) triggering: every event's underlying condition is
tracked as a per-(player, event-type) boolean each frame. An event only
fires the frame that boolean transitions False -> True, and cannot fire
again until it has gone back to False (fully cleared) AND the per-player
per-type cooldown has elapsed. This stops a sustained condition (e.g. a
player staying in the final third above the BREAK speed for several
seconds) from re-firing every cooldown window — it fires once per
genuine onset instead. Events requiring a minimum sustained duration
(RECOVERY, BREAK, BURST) fold that into the "condition" itself via a
per-player consecutive-frame run counter. SPRINT instead evaluates a
trailing window directly each frame: the window's AVERAGE speed must
exceed the threshold, rather than every individual frame — a real
sprint effort is a rise-peak-decay curve, not a sustained plateau, so a
strict per-frame (even gap-tolerant) rule rejected nearly all genuine
sprints when checked against this pipeline's real speed data. DROP
additionally requires a MAJORITY of the
per-frame steps inside its 2s window to be backward-moving (not just a
net displacement past the threshold), so noise that nets out past the
threshold without sustained backward motion doesn't count.

General noise guard: every event is skipped for a player on any frame
where their position_transformed fails pitch-bounds validation — this
already happens once, up front, for the current frame (see the initial
NaN check in the per-player loop). BURST and DROP additionally validate
the POSITION (not just the speed value) at every reference frame their
windowed comparison touches, since a stale/held-over speed reading from
an invalid-position frame was the main source of phantom spikes.
"""

import numpy as np
from collections import defaultdict

from speed_and_distance_estimator.speed_and_distance_estimator import (
    BUFFER_SIZE as SPEED_SMOOTH_FRAMES,
    MAX_SPEED_KMH as MAX_PLAUSIBLE_KMH,
)

PITCH_X_MAX = 105.0
PITCH_Y_MAX = 68.0

SPRINT_KMH          = 27.0
SPRINT_MIN_SEC      = 2.0   # the window average speed must clear SPRINT_KMH over this whole duration, not just briefly touch it
SPRINT_MIN_COVERAGE_FRAC = 0.5   # fraction of the window that must have an evaluable (valid position + speed) reading for the average to be meaningful
BURST_DELTA_KMH     = 15.0
BURST_WINDOW_SEC    = 2.0   # the increase must be sustained/confirmed across this whole window, not a brief spike
BURST_MIN_FRAMES    = 3    # elevated-acceleration must persist, not a single-frame spike
BURST_MAX_ACCEL_MS2 = 10.0 # elite human peak accel is ~8-9 m/s²; hard ceiling with a little margin
PRESS_DIST_M        = 3.0
PRESS_MIN_KMH       = 12.0
RECOVERY_KMH        = 20.0
RECOVERY_MIN_FRAMES = 5
RECOVERY_WINDOW_SEC = 2.0   # direction + backward-displacement gate are measured over this window, for stability
RECOVERY_BACKWARD_FRAC = 0.6   # backward x-component must be >= this fraction of total displacement
OVERLAP_FLANK_TOL_M = 12.0
OVERLAP_X_PROXIMITY_M = 18.0   # max along-pitch separation — an overlap is a run near the same area of play, not across the whole pitch
OVERLAP_MIN_KMH     = 15.0
OVERLAP_EXPIRE_SEC  = 2.0   # a pair's crossover state resets after this long out of the proximity window
LATERAL_DIST_M      = 20.0
LATERAL_WINDOW_SEC  = 3.0
SPEED_CORROBORATION_FRAC = 0.5   # tracked smoothed-speed average must reach >= this fraction of the raw-displacement-implied pace
DROP_DIST_M         = 15.0
DROP_WINDOW_SEC     = 2.0
DROP_MAJORITY_FRAC  = 0.5   # fraction of per-frame steps in the window that must be backward
DROP_SPACE_PROXIMITY_M = 30.0   # max distance from the team's own most-space-controlling teammate — stays connected to the team's zone, not disconnected backward jogging
BREAK_KMH           = 20.0
BREAK_MIN_FRAMES    = 5
FINAL_THIRD_M       = PITCH_X_MAX / 3.0   # 35.0
ISOLATED_DIST_M     = 25.0
ISOLATED_MAX_DIST_M = 50.0   # readings beyond this are treated as tracking/occlusion artifacts, not real isolation
ISOLATED_WIDE_Y_LOW  = 15.0   # outer 22% of the 68m pitch width on either touchline
ISOLATED_WIDE_Y_HIGH = 53.0
COOLDOWN_SEC        = 5.0   # per-player, per-event-type cooldown


def _valid_pos(p):
    """world position_transformed -> (x, y) if in-bounds, else None."""
    if p is None:
        return None
    try:
        x, y = float(p[0]), float(p[1])
    except Exception:
        return None
    if not (0.0 <= x <= PITCH_X_MAX and 0.0 <= y <= PITCH_Y_MAX):
        return None
    return (x, y)


class TacticalEventsDetector:
    """Single-pass detector for the 10 position/movement tactical events."""

    def __init__(self, fps):
        self.fps = fps
        self.cooldown_frames = max(1, int(round(COOLDOWN_SEC * fps)))
        self.overlap_expire_frames = max(1, int(round(OVERLAP_EXPIRE_SEC * fps)))
        self.event_counts = defaultdict(int)
        self._overlap_prev = {}   # (pid_a, pid_b) -> bool, a ahead of b
        self._overlap_last_qual_frame = {}   # (pid_a, pid_b) -> last frame the pair qualified

    def detect(self, tracks, space_control_per_frame):
        """Run once. Returns events_by_frame: dict frame_num -> list of
        {'frame': int, 'type': str, 'player_id': int}."""
        n = len(tracks['players'])
        all_pids = sorted({pid for frame in tracks['players'] for pid in frame})

        pos = {pid: np.full((n, 2), np.nan, dtype=np.float64) for pid in all_pids}
        spd = {pid: np.full(n, np.nan, dtype=np.float64) for pid in all_pids}
        team_of = {}

        for fn, player_data in enumerate(tracks['players']):
            for pid, info in player_data.items():
                p = _valid_pos(info.get('position_transformed'))
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

        # ── Attacking direction per team ────────────────────────────────────
        # Mean world-x across the whole clip: the team defending the x=0 end
        # sits at lower mean x and attacks toward +x. Short highlight clips
        # don't swap ends, so one fixed direction per team for the whole clip
        # is a reasonable simplification.
        team_x = {1: [], 2: []}
        for pid in all_pids:
            t = team_of.get(pid)
            if t not in (1, 2):
                continue
            xs = pos[pid][:, 0]
            xs = xs[~np.isnan(xs)]
            if len(xs):
                team_x[t].append(xs.mean())
        mean_x = {t: (float(np.mean(team_x[t])) if team_x[t] else PITCH_X_MAX / 2)
                  for t in (1, 2)}
        if mean_x[1] <= mean_x[2]:
            attack_dir = {1: 1.0, 2: -1.0}
        else:
            attack_dir = {1: -1.0, 2: 1.0}

        w_burst   = max(1, int(round(BURST_WINDOW_SEC * self.fps)))
        w_lateral = max(1, int(round(LATERAL_WINDOW_SEC * self.fps)))
        w_drop    = max(1, int(round(DROP_WINDOW_SEC * self.fps)))
        w_recov   = max(1, int(round(RECOVERY_WINDOW_SEC * self.fps)))
        sprint_min_frames = max(1, int(round(SPRINT_MIN_SEC * self.fps)))

        events_by_frame = defaultdict(list)
        last_emit = {}    # (pid, event_type) -> last frame emitted
        active    = {}    # (pid, event_type) -> bool, condition state last frame

        def emit(fn, etype, pid):
            key = (pid, etype)
            last = last_emit.get(key)
            if last is not None and fn - last < self.cooldown_frames:
                return
            last_emit[key] = fn
            events_by_frame[fn].append({'frame': fn, 'type': etype, 'player_id': int(pid)})
            self.event_counts[etype] += 1

        def rising_edge(pid, etype, cond):
            """Fire (return True) only the frame `cond` goes False -> True.
            The event still separately has to clear the cooldown inside
            emit() before it's actually logged."""
            key = (pid, etype)
            was = active.get(key, False)
            active[key] = cond
            if cond and not was:
                emit(fn, etype, pid)

        recovery_run = defaultdict(int)
        break_run    = defaultdict(int)
        burst_run    = defaultdict(int)

        for fn in range(n):
            frame_valid = []   # (pid, team, (x,y), speed) for cross-player events below

            for pid in all_pids:
                t = team_of.get(pid)
                if t not in (1, 2):
                    continue
                p = pos[pid][fn]
                if np.isnan(p[0]):
                    recovery_run[pid] = 0
                    break_run[pid] = 0
                    burst_run[pid] = 0
                    continue   # invalid position this frame -> skip detection for them
                s = spd[pid][fn]
                s = 0.0 if np.isnan(s) else s
                frame_valid.append((pid, t, p, s))

                # 1. SPRINT — the AVERAGE speed over the trailing
                # SPRINT_MIN_SEC (2s / 50-frame) window exceeds 27 km/h. A
                # per-frame "every single frame must clear the threshold"
                # rule (even with generous gap tolerance) turned out to
                # reject essentially all real sprints: actual sprint effort
                # is a rise-peak-decay curve, not a plateau — checked against
                # this pipeline's real data, the longest true frame-by-frame
                # above-threshold run in a 6-minute sample was ~1s, well
                # under any 2s bar. Averaging over the window captures a
                # genuine ~2s sprint EFFORT (including its natural decay)
                # without requiring an unrealistic sustained peak. Only
                # frames with an evaluable reading (valid position AND
                # speed) count toward the average; the window needs at least
                # SPRINT_MIN_COVERAGE_FRAC (50%) evaluable coverage for that
                # average to be meaningful, rather than a sparse handful of
                # points. The speed field is itself already bounded by
                # MAX_PLAUSIBLE_KMH, so no separate outlier cap is needed.
                fn0sp = fn - sprint_min_frames + 1
                sprint_cond_now = False
                if fn0sp >= 0:
                    pos_window_sp = pos[pid][fn0sp:fn + 1, 0]
                    spd_window_sp = spd[pid][fn0sp:fn + 1]
                    evaluable = ~np.isnan(pos_window_sp) & ~np.isnan(spd_window_sp)
                    n_evaluable = int(np.count_nonzero(evaluable))
                    if n_evaluable >= SPRINT_MIN_COVERAGE_FRAC * sprint_min_frames:
                        window_avg_speed = float(np.mean(spd_window_sp[evaluable]))
                        sprint_cond_now = window_avg_speed > SPRINT_KMH
                rising_edge(pid, 'SPRINT', sprint_cond_now)

                # 2. BURST — speed increases > 15 km/h within a 2s window
                # (sustained/confirmed across the full window, not a brief
                # spike), SUSTAINED for 3+ consecutive frames (not a single
                # noisy-window comparison). The
                # reference frame's POSITION must also be valid — a stale speed
                # value held over from an invalid-position frame was the main
                # source of phantom spikes. Plausibility: the implied
                # acceleration is hard-capped at a physically realistic
                # ceiling, AND the whole window must show continuous valid
                # tracking (position and smoothed speed present every frame,
                # not just at the two endpoints) — a player re-entering
                # tracking already at speed (occlusion/ID loss then
                # reacquisition) produces the same two-point jump without
                # ever being continuously tracked while accelerating, so
                # requiring continuity catches that artifact directly.
                fn0 = fn - w_burst
                burst_cond_now = False
                if fn0 >= 0:
                    s0 = spd[pid][fn0]
                    p0 = pos[pid][fn0]
                    if not np.isnan(s0) and not np.isnan(p0[0]) and (s - s0) > BURST_DELTA_KMH:
                        implied_accel = ((s - s0) / 3.6) / BURST_WINDOW_SEC   # m/s^2
                        pos_window_b = pos[pid][fn0:fn + 1, 0]
                        spd_window_b = spd[pid][fn0:fn + 1]
                        continuously_tracked = (not np.isnan(pos_window_b).any()
                                                and not np.isnan(spd_window_b).any())
                        burst_cond_now = implied_accel <= BURST_MAX_ACCEL_MS2 and continuously_tracked
                burst_run[pid] = burst_run[pid] + 1 if burst_cond_now else 0
                rising_edge(pid, 'BURST', burst_run[pid] >= BURST_MIN_FRAMES)

                # 4. RECOVERY — sustained 20+ km/h toward own goal for 5+ frames.
                # Direction and the backward-displacement plausibility gate are
                # both computed over a fixed RECOVERY_WINDOW_SEC (2s) window —
                # long enough that homography/position jitter (which dominated
                # at the old 3-frame/0.12s window) averages out, rather than a
                # single raw frame-to-frame delta deciding direction.
                # Plausibility: the backward x-component must account for most
                # (>=60%) of the total displacement in that window, so a mostly-
                # lateral/forward run with a small backward wobble doesn't count.
                recov_cond_now = False
                fn0r = fn - w_recov
                if fn0r >= 0 and s >= RECOVERY_KMH:
                    p_prev = pos[pid][fn0r]
                    if not np.isnan(p_prev[0]):
                        disp_x = p[0] - p_prev[0]
                        disp_y = p[1] - p_prev[1]
                        total_disp = float(np.hypot(disp_x, disp_y))
                        backward_x = -disp_x * attack_dir[t]
                        recov_cond_now = (total_disp > 1e-6
                                          and backward_x >= RECOVERY_BACKWARD_FRAC * total_disp)
                recovery_run[pid] = recovery_run[pid] + 1 if recov_cond_now else 0
                rising_edge(pid, 'RECOVERY', recovery_run[pid] >= RECOVERY_MIN_FRAMES)

                # 7. LATERAL_RUN — a player's own lateral displacement > 20m
                # over a 3s window (renamed from SWITCH: it measures player
                # movement, not a ball-driven "switch of play"). Plausibility
                # is checked against the already-smoothed, teleport-guarded
                # speed field across the whole window rather than trusting a
                # single raw two-point position diff (which bypasses the
                # per-frame teleport/cap guards speed_and_distance_estimator
                # applies and can blow up from one bad homography frame or an
                # ID swap): the implied speed of the displacement must be
                # hard-capped at the same physical ceiling, AND the window's
                # average tracked speed must itself reach at least half that
                # implied pace — a large lateral jump with a near-zero tracked
                # speed trace is a tracking artifact, not a real run, even
                # though its implied speed alone stays under the cap.
                fn0l = fn - w_lateral
                lateral_cond = False
                if fn0l >= 0:
                    p0 = pos[pid][fn0l]
                    if not np.isnan(p0[1]):
                        lateral_dist = abs(p[1] - p0[1])
                        if lateral_dist > LATERAL_DIST_M:
                            implied_kmh = (lateral_dist / LATERAL_WINDOW_SEC) * 3.6
                            spd_window_l = spd[pid][fn0l:fn + 1]
                            valid_spd_l = spd_window_l[~np.isnan(spd_window_l)]
                            if len(valid_spd_l) > 0:
                                avg_tracked_kmh = float(np.mean(valid_spd_l))
                                lateral_cond = (implied_kmh <= MAX_PLAUSIBLE_KMH
                                                and avg_tracked_kmh >= SPEED_CORROBORATION_FRAC * implied_kmh)
                rising_edge(pid, 'LATERAL_RUN', lateral_cond)

                # 8. DROP — backward movement > 15m over 2s toward own half,
                # AND a majority of the per-frame steps inside that window must
                # themselves be backward-moving (not just a net displacement
                # that happens to net out past the threshold from noise). The
                # whole window's positions must be valid — any gap makes the
                # majority-fraction computation unreliable. The implied speed
                # of the net displacement is hard-capped at the same physical
                # ceiling as LATERAL_RUN/the speed estimator; where the window
                # has valid smoothed-speed data, that average must also clear
                # the cap AND reach at least half the implied pace — an upper
                # bound alone doesn't catch a large position jump paired with
                # a tracked-speed trace that stayed near zero throughout.
                # Space-awareness: the player's LANDING position must be within
                # DROP_SPACE_PROXIMITY_M of their own team's most-space-
                # controlling teammate this frame (reusing space_control_
                # per_frame, no new infrastructure/ball tracking) — a proxy
                # for staying connected to the team's active zone, so DROP
                # represents a real supporting/passing option rather than
                # disconnected backward jogging away from play.
                fn0d = fn - w_drop
                drop_cond = False
                if fn0d >= 0:
                    window = pos[pid][fn0d:fn + 1, 0]   # world-x across the window
                    if not np.isnan(window).any():
                        net_signed = window[-1] - window[0]
                        net = net_signed * attack_dir[t]
                        if net < -DROP_DIST_M:
                            implied_kmh = (abs(net_signed) / DROP_WINDOW_SEC) * 3.6
                            spd_window = spd[pid][fn0d:fn + 1]
                            valid_spd = spd_window[~np.isnan(spd_window)]
                            plausible = implied_kmh <= MAX_PLAUSIBLE_KMH
                            if plausible and len(valid_spd) > 0:
                                corroborated_kmh = float(np.mean(valid_spd))
                                plausible = (corroborated_kmh <= MAX_PLAUSIBLE_KMH
                                            and corroborated_kmh >= SPEED_CORROBORATION_FRAC * implied_kmh)
                            if plausible:
                                steps = np.diff(window) * attack_dir[t]
                                backward_frac = np.mean(steps < 0) if len(steps) else 0.0
                                if backward_frac >= DROP_MAJORITY_FRAC:
                                    drop_space_ok = False
                                    if fn < len(space_control_per_frame):
                                        space_frame_d = space_control_per_frame[fn]
                                        team_others = [pid2 for pid2, info2 in space_frame_d.items()
                                                      if team_of.get(pid2) == t and pid2 != pid]
                                        if team_others:
                                            anchor_pid = max(team_others,
                                                             key=lambda pid2: space_frame_d[pid2]['frac'])
                                            anchor_pos = pos[anchor_pid][fn]
                                            if not np.isnan(anchor_pos[0]):
                                                dist_to_anchor = np.hypot(p[0] - anchor_pos[0], p[1] - anchor_pos[1])
                                                drop_space_ok = dist_to_anchor <= DROP_SPACE_PROXIMITY_M
                                    drop_cond = drop_space_ok
                rising_edge(pid, 'DROP', drop_cond)

                # 9. BREAK — sustained 20+ km/h into the final third for 5+
                # frames, AND running into above-average space control for
                # their own team this frame (reusing the precomputed
                # space-control grid, no new infrastructure) — this keeps
                # BREAK to genuinely dangerous runs into open space, not just
                # any fast forward sprint into a crowd of defenders.
                if s >= BREAK_KMH:
                    in_final_third = (p[0] >= (PITCH_X_MAX - FINAL_THIRD_M)) if attack_dir[t] > 0 \
                        else (p[0] <= FINAL_THIRD_M)
                else:
                    in_final_third = False

                break_space_ok = False
                if in_final_third and fn < len(space_control_per_frame):
                    space_frame_b = space_control_per_frame[fn]
                    my_space_b = space_frame_b.get(pid)
                    if my_space_b is not None:
                        team_fracs = [info2['frac'] for pid2, info2 in space_frame_b.items()
                                     if team_of.get(pid2) == t]
                        if team_fracs:
                            break_space_ok = my_space_b['frac'] > (sum(team_fracs) / len(team_fracs))

                break_run[pid] = break_run[pid] + 1 if (in_final_third and break_space_ok) else 0
                rising_edge(pid, 'BREAK', break_run[pid] >= BREAK_MIN_FRAMES)

                # 6. SPACE — top 10% of space control this frame (precomputed)
                space_info = space_control_per_frame[fn].get(pid) if fn < len(space_control_per_frame) else None
                rising_edge(pid, 'SPACE', bool(space_info and space_info.get('top10')))

            team1 = [f for f in frame_valid if f[1] == 1]
            team2 = [f for f in frame_valid if f[1] == 2]

            # 3. PRESS — two same-team players within 3m of the same opponent,
            # both moving 12+ km/h
            pressing_now = set()
            for defenders, opponents in ((team1, team2), (team2, team1)):
                for opid, ot, opos, ospd in opponents:
                    pressers = []
                    for dpid, dt, dpos, dspd in defenders:
                        if dspd < PRESS_MIN_KMH:
                            continue
                        d = np.hypot(dpos[0] - opos[0], dpos[1] - opos[1])
                        if d <= PRESS_DIST_M:
                            pressers.append(dpid)
                    if len(pressers) >= 2:
                        pressing_now.update(pressers)
            for pid, _, _, _ in frame_valid:
                rising_edge(pid, 'PRESS', pid in pressing_now)

            # 10. ISOLATED — wide-channel player (outer 22% of pitch width,
            # y<15m or y>53m on the 68m pitch) whose nearest same-team
            # teammate is > 25m away. Restricted to wide players: a central
            # player 25m from the nearest teammate is often just normal
            # defensive-line/midfield spacing, not genuine isolation. Readings
            # above 50m are capped out as tracking/occlusion artifacts (e.g.
            # the true nearest teammate briefly missing a valid position that
            # frame), not real isolation — both conditions (wide + plausible
            # distance) are required together.
            isolated_now = set()
            for group in (team1, team2):
                if len(group) < 2:
                    continue
                for i, (pid_i, _, pos_i, _) in enumerate(group):
                    if not (pos_i[1] < ISOLATED_WIDE_Y_LOW or pos_i[1] > ISOLATED_WIDE_Y_HIGH):
                        continue   # not a wide player this frame
                    best = min(
                        np.hypot(pos_i[0] - pos_j[0], pos_i[1] - pos_j[1])
                        for j, (_, _, pos_j, _) in enumerate(group) if j != i
                    )
                    if ISOLATED_DIST_M < best <= ISOLATED_MAX_DIST_M:
                        isolated_now.add(pid_i)
            for pid, _, _, _ in frame_valid:
                rising_edge(pid, 'ISOLATED', pid in isolated_now)

            # 5. OVERLAP — player runs from behind to ahead of a teammate on
            # the same flank, both moving forward at 15+ km/h. The crossover
            # itself is already a discrete, momentary state transition (not a
            # sustained condition), so it needs no separate rising-edge gate.
            # Both the flank (y) tolerance AND an along-pitch (x) proximity
            # bound are required together — an "overlap" is a supporting run
            # near the same area of play; without the x bound, two players on
            # opposite ends of the pitch that happen to share a y-band and
            # both swap x-order at some point would false-fire.
            for t, group in ((1, team1), (2, team2)):
                d = attack_dir[t]
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        pid_a, _, pos_a, spd_a = group[i]
                        pid_b, _, pos_b, spd_b = group[j]
                        if abs(pos_a[1] - pos_b[1]) > OVERLAP_FLANK_TOL_M:
                            continue
                        if abs(pos_a[0] - pos_b[0]) > OVERLAP_X_PROXIMITY_M:
                            continue
                        if spd_a < OVERLAP_MIN_KMH or spd_b < OVERLAP_MIN_KMH:
                            continue
                        if fn < 1:
                            continue
                        pa_prev = pos[pid_a][fn - 1]
                        pb_prev = pos[pid_b][fn - 1]
                        if np.isnan(pa_prev[0]) or np.isnan(pb_prev[0]):
                            continue
                        fwd_a = (pos_a[0] - pa_prev[0]) * d > 0
                        fwd_b = (pos_b[0] - pb_prev[0]) * d > 0
                        if not (fwd_a and fwd_b):
                            continue
                        key = (min(pid_a, pid_b), max(pid_a, pid_b))

                        # Stale-state guard: if this pair last qualified more
                        # than OVERLAP_EXPIRE_SEC ago (they dropped out of the
                        # flank/speed proximity window and just re-qualified),
                        # discard any lingering a_ahead_now state instead of
                        # comparing against it — otherwise a crossover could
                        # fire based on a stale position relationship from a
                        # much earlier, unrelated passage of play.
                        last_qual = self._overlap_last_qual_frame.get(key)
                        if last_qual is not None and fn - last_qual > self.overlap_expire_frames:
                            self._overlap_prev.pop(key, None)
                        self._overlap_last_qual_frame[key] = fn

                        a_ahead_now = (pos_a[0] * d) > (pos_b[0] * d)
                        prev_state = self._overlap_prev.get(key)
                        if prev_state is not None and prev_state != a_ahead_now:
                            crosser = pid_a if a_ahead_now else pid_b
                            emit(fn, 'OVERLAP', crosser)
                        self._overlap_prev[key] = a_ahead_now

        return dict(events_by_frame)
