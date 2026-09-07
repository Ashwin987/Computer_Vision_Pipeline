"""
render_output2.py - Player stamina dashboard.

Layout : 1440px video (75%)  +  480px stamina panel (25%)  =  1920px total.
Stamina drain rates (% per second):
    Sprinting >20 km/h : 0.50
    Running  10-20     : 0.20
    Jogging   5-10     : 0.10
    Walking    <5      : 0.02
Display updates every 10 seconds to avoid per-frame flicker.
ID re-matching: proximity 80px within last 30 frames.
Counter-attack: retroactive off-screen drain applied on re-entry.

Saves: output_videos/output2.avi
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

VIDEO_W  = 1440
PANEL_W  = 480
OUT_W    = VIDEO_W + PANEL_W   # 1920

PANEL_BG  = (30, 30, 30)
HEADER_H  = 44
FOOTER_H  = 52
ROW_H     = 42

PROXIMITY_PX      = 80
LOST_BUF_FRAMES   = 30
UPDATE_SEC        = 10    # seconds between display refresh
COUNTER_ATK_KMH   = 15.0
SWITCH_SHOW_SEC   = 2     # seconds to show "ID: old→new"

DRAIN_SPRINT = 0.50   # %/sec  >20 km/h
DRAIN_RUN    = 0.20   # %/sec  10-20
DRAIN_JOG    = 0.10   # %/sec   5-10
DRAIN_WALK   = 0.02   # %/sec   <5


def _drain_rate(speed_kmh):
    if speed_kmh > 20.0: return DRAIN_SPRINT
    if speed_kmh > 10.0: return DRAIN_RUN
    if speed_kmh >  5.0: return DRAIN_JOG
    return DRAIN_WALK


def _bar_color(stam, team, team_color=None):
    """Stamina bar color: red<25%, orange<50%, else team color. The
    low/medium thresholds are stamina-STATE indicators (not team identity)
    and stay fixed red/orange regardless of team. team_color, when given,
    is the actual per-clip fitted kit color (tracks['players'][...]
    ['team_color'], the same field tracker.draw_annotations() uses) —
    falls back to the old red/green constants only if unavailable, so a
    clip whose kits aren't literally red/green (e.g. PSG's navy) still
    renders its real color instead of a hardcoded wrong one."""
    if stam < 25.0:
        return (0, 0, 220)      # red
    if stam < 50.0:
        return (0, 165, 255)    # orange
    if team == 1:
        return tuple(int(c) for c in team_color) if team_color is not None else (0, 0, 220)
    if team == 2:
        return tuple(int(c) for c in team_color) if team_color is not None else (0, 200, 0)
    if team == 3:
        return (255, 220, 0)    # goalkeeper — cyan/sky-blue, distinct from red/green/referee-yellow
    return (110, 110, 110)      # unknown/unresolved — gray, matches _dot_color


def _dot_color(team, team_color=None):
    if team == 1: return tuple(int(c) for c in team_color) if team_color is not None else (0, 0, 220)
    if team == 2: return tuple(int(c) for c in team_color) if team_color is not None else (0, 200, 0)
    if team == 3: return (255, 220, 0)    # goalkeeper — cyan/sky-blue
    return (110, 110, 110)                # unknown/referee — gray


def _draw_row(canvas, x0, ry, pid, team, stam, sp, in_frame,
              sid, switch_announce, frame_num, fps, team_color=None):
    pw       = PANEL_W
    text_col = (210, 210, 210) if in_frame else (90, 90, 90)

    # Team colour dot
    cv2.circle(canvas, (x0 + 12, ry + 14), 5, _dot_color(team, team_color), -1)

    # Player ID
    id_str = f"#{pid}"
    cv2.putText(canvas, id_str, (x0 + 22, ry + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_col, 1)
    id_w = cv2.getTextSize(id_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]

    # Speed / "Not in frame"
    if in_frame:
        status_str = f"{sp:.1f} km/h"
        status_col = (170, 170, 170)
    else:
        status_str = "Not in frame"
        status_col = (80, 80, 80)
    cv2.putText(canvas, status_str, (x0 + 24 + id_w, ry + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, status_col, 1)

    # ID-switch badge (top-right of row)
    if sid in switch_announce:
        old_p, new_p, ann_fn = switch_announce[sid]
        if frame_num - ann_fn < fps * SWITCH_SHOW_SEC:
            sw = f"{old_p}>{new_p}"
            sw_w = cv2.getTextSize(sw, cv2.FONT_HERSHEY_SIMPLEX, 0.28, 1)[0][0]
            cv2.putText(canvas, sw, (x0 + pw - sw_w - 6, ry + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 210, 230), 1)

    # Horizontal stamina bar
    bar_x = x0 + 8
    bar_y = ry + 22
    bar_w = pw - 8 - 8 - 38   # leave 38px for % label
    bar_h = 11
    cv2.rectangle(canvas, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (55, 55, 55), -1)
    fill_w = max(0, int(stam / 100.0 * bar_w))
    if fill_w > 0:
        cv2.rectangle(canvas, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h),
                      _bar_color(stam, team, team_color), -1)

    # % label
    cv2.putText(canvas, f"{stam:.0f}%", (bar_x + bar_w + 4, bar_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, text_col, 1)

    # Row separator
    cv2.line(canvas, (x0, ry + ROW_H - 1), (x0 + pw, ry + ROW_H - 1),
             (50, 50, 50), 1)


def render_output2(output_video_frames, tracks, team_ball_control, fps,
                   homography_per_frame=None):
    if not output_video_frames:
        print("render_output2: no frames, skipping.")
        return

    base_h   = output_video_frames[0].shape[0]
    n        = len(output_video_frames)
    update_every = max(1, int(fps * UPDATE_SEC))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out    = cv2.VideoWriter('output_videos/output2.avi', fourcc, fps,
                             (OUT_W, base_h))

    # ── Persistent player state ───────────────────────────────────────────────
    # all_players[sid] = {team, stamina_actual, stamina_display,
    #                     last_speed, in_frame, last_center,
    #                     last_frame_seen, tracker_pid}
    all_players   = {}
    tracker_to_sid = {}       # current tracker pid → stable id
    recently_lost  = {}       # sid → (cx, cy, frame_lost)
    switch_announce = {}      # sid → (old_pid, new_pid, frame_announced)
    next_sid       = 1

    for frame_num in tqdm(range(n), desc="Rendering output2 (stamina)"):
        raw_player_data = (tracks['players'][frame_num]
                           if frame_num < len(tracks['players']) else {})
        # Only resolved outfield players belong on this panel -- team=0
        # (never confidently resolved: brief/occluded tracks, or a
        # rejected goalkeeper detection demoted by
        # TeamAssigner._reject_non_goalkeeper_positions /
        # _reject_static_goalkeeper_detections) and team=3 (goalkeepers,
        # a real role but not one of the two teams being compared here)
        # both used to leak into the panel as rows. Filtering here, before
        # any sid registration/bookkeeping, keeps them from ever occupying
        # a row slot at all -- not even a "Not in frame" ghost row -- and
        # is what fixes the panel's ONLY other visible symptom (looking
        # uniformly tan/brown with no team split): the row list is sorted
        # by (team, sid) ascending, so team=0 rows sorted first were
        # filling the panel's ~23-row budget before team 2's rows ever
        # got a turn, on a clip with 42 distinct stable ids seen by frame
        # 217 alone (13 team=0, 18 team=1, 11 team=2 -- team 2 never once
        # made it into the visible window). The per-row color logic itself
        # (_bar_color/_dot_color reading each player's real fitted
        # team_color) was already correct.
        player_data  = {pid: info for pid, info in raw_player_data.items()
                        if info.get('team') in (1, 2)}
        current_pids = set(player_data.keys())
        prev_pids    = set(tracker_to_sid.keys())

        appeared    = current_pids - prev_pids
        disappeared = prev_pids    - current_pids

        # ── Handle disappeared players ────────────────────────────────────────
        for pid in disappeared:
            sid = tracker_to_sid.pop(pid, None)
            if sid is None or sid not in all_players:
                continue
            p = all_players[sid]
            recently_lost[sid] = (*p.get('last_center', (0, 0)), frame_num)
            p['in_frame']        = False
            p['last_frame_seen'] = frame_num

        # Expire old lost-player buffer entries
        stale = [s for s, (_, _, fn) in recently_lost.items()
                 if frame_num - fn > LOST_BUF_FRAMES]
        for s in stale:
            del recently_lost[s]

        # ── Match appeared players to recently-lost by proximity ──────────────
        matched_sids = set()
        for pid in sorted(appeared):   # deterministic order
            bbox = player_data[pid].get('bbox')
            if bbox is None:
                sid = next_sid; next_sid += 1
                tracker_to_sid[pid] = sid
                continue

            ncx = (bbox[0] + bbox[2]) / 2.0
            ncy = (bbox[1] + bbox[3]) / 2.0

            best_sid, best_d = None, PROXIMITY_PX + 1
            for sid, (cx, cy, _) in recently_lost.items():
                if sid in matched_sids:
                    continue
                d = np.sqrt((ncx - cx) ** 2 + (ncy - cy) ** 2)
                if d < best_d:
                    best_d, best_sid = d, sid

            if best_sid is not None:
                # Returning player — counter-attack retroactive drain
                p = all_players.get(best_sid, {})
                off_secs     = (frame_num - p.get('last_frame_seen', frame_num)) / max(fps, 1)
                re_spd       = float(player_data[pid].get('speed', 0))
                drain        = (DRAIN_SPRINT if re_spd > COUNTER_ATK_KMH else DRAIN_WALK)
                if best_sid in all_players:
                    all_players[best_sid]['stamina_actual'] = max(
                        0.0, all_players[best_sid]['stamina_actual'] - drain * off_secs)
                    all_players[best_sid]['in_frame'] = True

                # Record ID switch if tracker assigned a new number
                old_pid = all_players[best_sid].get('tracker_pid', best_sid)
                if old_pid != pid:
                    switch_announce[best_sid] = (old_pid, pid, frame_num)

                tracker_to_sid[pid] = best_sid
                matched_sids.add(best_sid)
                del recently_lost[best_sid]
            else:
                # Brand-new player
                sid = next_sid; next_sid += 1
                tracker_to_sid[pid] = sid

        # ── Register new sids in all_players ─────────────────────────────────
        for pid in current_pids:
            sid = tracker_to_sid.get(pid)
            if sid is None:
                sid = next_sid; next_sid += 1
                tracker_to_sid[pid] = sid
            if sid not in all_players:
                all_players[sid] = {
                    'team':            player_data[pid].get('team', 0),
                    'team_color':      player_data[pid].get('team_color'),
                    'stamina_actual':  100.0,
                    'stamina_display': 100.0,
                    'last_speed':      0.0,
                    'in_frame':        True,
                    'last_center':     (0.0, 0.0),
                    'last_frame_seen': frame_num,
                    'tracker_pid':     pid,
                }

        # ── Per-frame stamina drain for active players ────────────────────────
        for pid, info in player_data.items():
            sid = tracker_to_sid.get(pid)
            if sid is None or sid not in all_players:
                continue
            p  = all_players[sid]
            sp = float(info.get('speed', 0))

            p['team']        = info.get('team', p['team'])
            p['team_color']  = info.get('team_color', p.get('team_color'))
            p['last_speed']  = sp
            p['in_frame']    = True
            p['tracker_pid'] = pid

            bbox = info.get('bbox')
            if bbox:
                p['last_center'] = ((bbox[0] + bbox[2]) / 2.0,
                                    (bbox[1] + bbox[3]) / 2.0)
            p['last_frame_seen'] = frame_num

            # Continuous drain
            p['stamina_actual'] = max(
                0.0, p['stamina_actual'] - _drain_rate(sp) / max(fps, 1))

            # Refresh display every UPDATE_SEC seconds
            if frame_num % update_every == 0:
                p['stamina_display'] = p['stamina_actual']

        # Clean up expired switch announcements
        expired_sw = [s for s, (_, _, fn) in switch_announce.items()
                      if frame_num - fn >= fps * SWITCH_SHOW_SEC]
        for s in expired_sw:
            del switch_announce[s]

        # ── Build canvas ──────────────────────────────────────────────────────
        src = output_video_frames[frame_num]
        # Crop to 1920px if events panel was appended
        if src.shape[1] > 1920:
            src = src[:, :1920]
        vid_frame = cv2.resize(src, (VIDEO_W, base_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((base_h, OUT_W, 3), dtype=np.uint8)
        canvas[:, :VIDEO_W]  = vid_frame
        canvas[:, VIDEO_W:]  = PANEL_BG

        x0 = VIDEO_W
        pw = PANEL_W

        # Panel divider
        cv2.line(canvas, (x0, 0), (x0, base_h), (60, 60, 60), 1)

        # Header
        cv2.rectangle(canvas, (x0, 0), (x0 + pw, HEADER_H), (20, 20, 20), -1)
        hdr = "PLAYER STAMINA"
        hw  = cv2.getTextSize(hdr, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[0][0]
        cv2.putText(canvas, hdr, (x0 + (pw - hw) // 2, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        cv2.line(canvas, (x0, HEADER_H), (x0 + pw, HEADER_H), (60, 60, 60), 1)

        # Sort: Team 1 first, Team 2 second, then by stable_id within team
        sorted_sids = sorted(
            all_players.keys(),
            key=lambda s: (all_players[s].get('team', 9), s)
        )

        avail_h  = base_h - HEADER_H - FOOTER_H
        max_rows = avail_h // ROW_H

        for i, sid in enumerate(sorted_sids[:max_rows]):
            p        = all_players[sid]
            ry       = HEADER_H + i * ROW_H
            _draw_row(canvas, x0, ry,
                      pid       = p.get('tracker_pid', sid),
                      team      = p.get('team', 0),
                      stam      = p['stamina_display'],
                      sp        = p['last_speed'],
                      in_frame  = p['in_frame'],
                      sid       = sid,
                      switch_announce = switch_announce,
                      frame_num = frame_num,
                      fps       = fps,
                      team_color = p.get('team_color'))

        # Footer: live team average speeds
        fy = base_h - FOOTER_H
        cv2.line(canvas, (x0, fy), (x0 + pw, fy), (60, 60, 60), 1)
        cv2.rectangle(canvas, (x0, fy + 1), (x0 + pw, base_h), (20, 20, 20), -1)

        t1_spd = [all_players[s]['last_speed'] for s in all_players
                  if all_players[s].get('team') == 1 and all_players[s]['in_frame']]
        t2_spd = [all_players[s]['last_speed'] for s in all_players
                  if all_players[s].get('team') == 2 and all_players[s]['in_frame']]
        avg1 = f"T1 avg: {np.mean(t1_spd):.1f} km/h" if t1_spd else "T1 avg: --"
        avg2 = f"T2 avg: {np.mean(t2_spd):.1f} km/h" if t2_spd else "T2 avg: --"
        t1_color_dyn = next((all_players[s]['team_color'] for s in all_players
                             if all_players[s].get('team') == 1
                             and all_players[s].get('team_color') is not None), None)
        t2_color_dyn = next((all_players[s]['team_color'] for s in all_players
                             if all_players[s].get('team') == 2
                             and all_players[s].get('team_color') is not None), None)
        avg1_color = tuple(int(c) for c in t1_color_dyn) if t1_color_dyn is not None else (80, 80, 220)
        avg2_color = tuple(int(c) for c in t2_color_dyn) if t2_color_dyn is not None else (0, 180, 0)
        cv2.putText(canvas, avg1, (x0 + 8, fy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, avg1_color, 1)
        cv2.putText(canvas, avg2, (x0 + 8, fy + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, avg2_color, 1)

        if frame_num % 50 == 0:
            print(f"output2: frame {frame_num}/{n}  "
                  f"total_players={len(all_players)}")

        out.write(canvas)

    out.release()
    path    = 'output_videos/output2.avi'
    size_mb = os.path.getsize(path) / 1e6
    print(f"output2.avi saved ({n} frames, {size_mb:.1f} MB).")
