"""
match_events/rule_based_events.py

Per-frame incremental rule-based event detector.
Call update(frame_num, tracks) once per frame; events accumulate in self.events.

All thresholds are in pixel space (position_adjusted coordinates).
"""

import numpy as np
from collections import defaultdict, deque


def _bbox_center(bbox):
    if bbox is None:
        return None
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _pdist(p1, p2):
    if p1 is None or p2 is None:
        return float('inf')
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


class RuleBasedEventDetector:
    """
    Stateful per-frame event detector for soccer video.

    Detects: PASS, INTERCEPTION, SHOT, TACKLE, DRIBBLE.
    All distance thresholds are in pixels (position_adjusted space).
    """

    NEAR_BALL_PX       = 60    # ~2m equivalent; player "in possession" zone
    DRIBBLE_PX         = 80    # ball within this of player to count as dribble
    TACKLE_PX          = 60    # inter-player contact distance
    TACKLE_MIN_FRAMES  = 3     # consecutive contact frames needed
    TACKLE_POSS_WINDOW = 10    # frames to observe possession change post-contact
    PASS_TIMEOUT       = 60    # max frames to confirm a pass
    BALL_PASS_SPD      = 8     # pixels/frame → potential pass
    BALL_SHOT_SPD      = 20    # pixels/frame → shot
    SHOT_ZONE_FRAC     = 0.10  # leftmost/rightmost fraction = goal zones
    SHOT_COOLDOWN_F    = 30    # frames between consecutive shot events

    def __init__(self, fps, frame_w=1920, frame_h=1080):
        self.fps      = fps
        self.frame_w  = frame_w
        self.frame_h  = frame_h
        self.events   = []

        # Ball state
        self._ball_pos       = None
        self._ball_speed     = 0.0
        self._ball_dir       = (0.0, 0.0)

        # Pass / interception state
        self._pass_pending   = None   # dict or None
        self._near_ball_prev = None   # pid nearest ball last frame

        # Possession
        self._poss_prev = None
        self._poss_curr = None

        # Dribble: pid → frame dribble started
        self._dribble_start    = {}
        self._dribble_reported = set()   # (pid, start_frame) already emitted

        # Tackle: (pid1, pid2) → consecutive contact count
        self._contact_run      = defaultdict(int)
        # (pid1, pid2) → (team_that_had_ball, frame_contact_started)
        self._contact_poss     = {}
        self._tackle_reported  = set()   # (pid1, pid2, frame) already emitted

        # Shot cooldown
        self._shot_cooldown = 0

    # ── helpers ──────────────────────────────────────────────────────────────

    def _player_positions(self, player_data):
        """Return {pid: (cx, cy)} using position_adjusted or bbox center."""
        out = {}
        for pid, info in player_data.items():
            pos = info.get('position_adjusted')
            if pos is not None:
                try:
                    out[pid] = (float(pos[0]), float(pos[1]))
                    continue
                except Exception:
                    pass
            bbox = info.get('bbox')
            c = _bbox_center(bbox)
            if c:
                out[pid] = c
        return out

    def _nearest_to_ball(self, player_data, ppos, ball_pos, max_dist):
        """Return pid of closest player within max_dist, or None."""
        if ball_pos is None:
            return None
        best_pid, best_d = None, max_dist + 1
        for pid, xy in ppos.items():
            d = _pdist(xy, ball_pos)
            if d < best_d:
                best_d, best_pid = d, pid
        return best_pid if best_d <= max_dist else None

    def _possession(self, player_data):
        """Return (pid, team) with has_ball, or None."""
        for pid, info in player_data.items():
            if info.get('has_ball', False):
                return (pid, info.get('team', 0))
        return None

    # ── main update ──────────────────────────────────────────────────────────

    def update(self, frame_num, tracks):
        """Process one frame. Returns list of new event dicts."""
        new_events = []

        player_data = (tracks['players'][frame_num]
                       if frame_num < len(tracks['players']) else {})
        ball_data   = (tracks['ball'][frame_num]
                       if frame_num < len(tracks['ball']) else {})

        # Ball position
        ball_bbox = ball_data.get(1, {}).get('bbox')
        ball_pos  = _bbox_center(ball_bbox)

        prev_ball_pos  = self._ball_pos
        self._ball_pos = ball_pos

        if ball_pos and prev_ball_pos:
            dx = ball_pos[0] - prev_ball_pos[0]
            dy = ball_pos[1] - prev_ball_pos[1]
            spd = float(np.sqrt(dx ** 2 + dy ** 2))
            self._ball_speed = spd
            self._ball_dir   = (dx / spd, dy / spd) if spd > 0 else (0.0, 0.0)
        else:
            self._ball_speed = 0.0
            self._ball_dir   = (0.0, 0.0)

        # Player positions
        ppos = self._player_positions(player_data)

        # Possession
        self._poss_prev = self._poss_curr
        self._poss_curr = self._possession(player_data)

        # Nearest player to ball this frame
        near_now = self._nearest_to_ball(player_data, ppos, ball_pos, self.NEAR_BALL_PX)

        # ── DRIBBLE ──────────────────────────────────────────────────────────
        dribble_pid = self._nearest_to_ball(player_data, ppos, ball_pos, self.DRIBBLE_PX)
        if dribble_pid is not None:
            spd_kmh = float(player_data[dribble_pid].get('speed', 0))
            if spd_kmh > 5.0:
                if dribble_pid not in self._dribble_start:
                    self._dribble_start[dribble_pid] = frame_num
                else:
                    elapsed = frame_num - self._dribble_start[dribble_pid]
                    key     = (dribble_pid, self._dribble_start[dribble_pid])
                    if elapsed >= self.fps * 2 and key not in self._dribble_reported:
                        new_events.append({
                            'type':      'DRIBBLE',
                            'frame':     frame_num,
                            'player_id': dribble_pid,
                            'duration':  elapsed,
                            'text':      f"P{dribble_pid} DRIBBLE {elapsed/self.fps:.1f}s",
                        })
                        self._dribble_reported.add(key)
            else:
                self._dribble_start.pop(dribble_pid, None)
        else:
            self._dribble_start.clear()

        # ── TACKLE ────────────────────────────────────────────────────────────
        t1 = [p for p in player_data if player_data[p].get('team') == 1 and p in ppos]
        t2 = [p for p in player_data if player_data[p].get('team') == 2 and p in ppos]

        active_pairs = set()
        for p1 in t1:
            for p2 in t2:
                key = (min(p1, p2), max(p1, p2))
                if _pdist(ppos[p1], ppos[p2]) <= self.TACKLE_PX:
                    active_pairs.add(key)
                    self._contact_run[key] += 1
                    if key not in self._contact_poss and self._poss_curr:
                        self._contact_poss[key] = (self._poss_curr[1], frame_num)
                else:
                    if self._contact_run.get(key, 0) > 0:
                        self._contact_run[key] = 0
                        self._contact_poss.pop(key, None)

        # Check if any contact qualifies as a tackle
        for key in list(active_pairs):
            if self._contact_run.get(key, 0) < self.TACKLE_MIN_FRAMES:
                continue
            poss_info = self._contact_poss.get(key)
            if not poss_info:
                continue
            orig_team, contact_start = poss_info
            if frame_num - contact_start > self.TACKLE_POSS_WINDOW:
                continue
            if self._poss_curr and self._poss_curr[1] != orig_team:
                t_key = (key[0], key[1], contact_start)
                if t_key not in self._tackle_reported:
                    p1, p2 = key
                    tackler = p2 if self._poss_curr[1] == 2 else p1
                    tackled = p1 if tackler == p2 else p2
                    new_events.append({
                        'type':       'TACKLE',
                        'frame':      frame_num,
                        'player_id':  tackler,
                        'tackled_id': tackled,
                        'text':       f"P{tackler} TACKLE P{tackled}",
                    })
                    self._tackle_reported.add(t_key)
                    self._contact_run[key] = 0
                    self._contact_poss.pop(key, None)

        # ── SHOT ─────────────────────────────────────────────────────────────
        if self._shot_cooldown > 0:
            self._shot_cooldown -= 1
        elif ball_pos and self._ball_speed >= self.BALL_SHOT_SPD:
            dx_dir   = self._ball_dir[0]
            goal_w   = self.frame_w * self.SHOT_ZONE_FRAC
            shot_left  = dx_dir < -0.4 and ball_pos[0] < self.frame_w * 0.5
            shot_right = dx_dir >  0.4 and ball_pos[0] > self.frame_w * 0.5
            if (shot_left or shot_right) and self._poss_prev:
                direction = 'left' if shot_left else 'right'
                new_events.append({
                    'type':      'SHOT',
                    'frame':     frame_num,
                    'player_id': self._poss_prev[0],
                    'direction': direction,
                    'text':      f"P{self._poss_prev[0]} SHOT ({direction})",
                })
                self._shot_cooldown = self.SHOT_COOLDOWN_F

        # ── PASS / INTERCEPTION ───────────────────────────────────────────────
        if self._pass_pending is None:
            # Potential pass start: ball was near a player, now moving fast and free
            if (self._ball_speed >= self.BALL_PASS_SPD
                    and self._near_ball_prev is not None
                    and near_now is None
                    and self._poss_prev is not None):
                self._pass_pending = {
                    'passer_id':   self._poss_prev[0],
                    'passer_team': self._poss_prev[1],
                    'start_frame': frame_num,
                }
        else:
            pending      = self._pass_pending
            frames_since = frame_num - pending['start_frame']

            if near_now is not None:
                recv_team = player_data[near_now].get('team', 0)
                if recv_team == pending['passer_team'] and near_now != pending['passer_id']:
                    new_events.append({
                        'type':        'PASS',
                        'frame':       frame_num,
                        'player_id':   pending['passer_id'],
                        'receiver_id': near_now,
                        'text':        f"P{pending['passer_id']}->P{near_now} PASS",
                    })
                elif recv_team != 0 and recv_team != pending['passer_team']:
                    new_events.append({
                        'type':      'INTERCEPTION',
                        'frame':     frame_num,
                        'player_id': near_now,
                        'text':      f"P{near_now} INTERCEPTION",
                    })
                self._pass_pending = None
            elif frames_since > self.PASS_TIMEOUT:
                self._pass_pending = None

        self._near_ball_prev = near_now
        self.events.extend(new_events)
        return new_events
