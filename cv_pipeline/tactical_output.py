"""
tactical_output.py — generates output_videos/tactical_video.avi

Canvas layout  (original_w + 220)  x  original_h
  Zone A : top half    – tactical map  (Voronoi + pressure wash + formation + players)
  Zone B : bottom half – dedicated pressure heatmap
  Zone C : right 220px – fatigue / fitness sidebar
"""

import cv2
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon
from collections import defaultdict, deque
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

# ── Pitch real-world dimensions (from view_transformer) ──────────────────────
PITCH_W_M = 23.32   # visible depth  (position_transformed axis-0)
PITCH_H_M = 68.0    # pitch width    (position_transformed axis-1)
SIDEBAR_W = 220

# ── Colours (BGR) ────────────────────────────────────────────────────────────
T1_FILL     = (221, 138,  55)   # Voronoi fill  – Team 1
T2_FILL     = ( 48,  90, 216)   # Voronoi fill  – Team 2
T1_FORM     = (255, 180,  80)   # formation line – Team 1
T2_FORM     = ( 80, 130, 255)   # formation line – Team 2
T1_DOT      = (180,  80,  20)   # player dot     – Team 1
T2_DOT      = ( 30,  30, 180)   # player dot     – Team 2
GK_DOT      = (200,  50, 150)   # player dot     – Goalkeeper (team 3) — purple, distinct from referee-yellow
PITCH_LINE  = (200, 200, 200)


# ─────────────────────────────────────────────────────────────────────────────
# FatigueTracker
# ─────────────────────────────────────────────────────────────────────────────

class FatigueTracker:
    def __init__(self, fps, game_max_window_minutes=15):
        self.fps = fps
        self.game_max_window = game_max_window_minutes * 60 * fps
        self.speed_history    = defaultdict(list)
        self.game_max         = {}
        self.sprint_max_2min  = {}
        self.meta_power_hist  = defaultdict(list)
        self.prev_speed_ms    = {}

    def update(self, frame_num, player_id, speed_kmh):
        speed_ms = speed_kmh / 3.6
        self.speed_history[player_id].append(speed_kmh)

        # game-max: track throughout (falls back to running max on short clips)
        self.game_max[player_id] = max(self.game_max.get(player_id, 0), speed_kmh)

        # sprint-max over last 2 min
        window = self.fps * 120
        recent = self.speed_history[player_id][-window:]
        self.sprint_max_2min[player_id] = max(recent) if recent else 0

        # metabolic power (|v·a|)
        if player_id in self.prev_speed_ms:
            accel = (speed_ms - self.prev_speed_ms[player_id]) * self.fps
            self.meta_power_hist[player_id].append(abs(speed_ms * accel))
        self.prev_speed_ms[player_id] = speed_ms

    def is_gassed(self, player_id):
        gmax = self.game_max.get(player_id, 0)
        if gmax < 1.0:
            return False
        return self.sprint_max_2min.get(player_id, 0) < 0.80 * gmax

    def is_critical(self, player_id):
        history = self.speed_history[player_id][-(self.fps * 600):]
        if len(history) < 50:
            return False
        try:
            x = np.arange(len(history), dtype=float)
            y = np.array(history, dtype=float)
            y_safe = np.clip(y, 0.1, None)
            coeffs = np.polyfit(x, np.log(y_safe), 1)
            b = coeffs[0]
            y_pred = np.exp(np.polyval(coeffs, x))
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            return b < -0.0001 and r2 > 0.75
        except Exception:
            return False

    def get_reserve_pct(self, player_id):
        gmax = self.game_max.get(player_id, 1)
        recent = self.sprint_max_2min.get(player_id, 0)
        return min(100, int(recent / gmax * 100)) if gmax > 0 else 100

    def get_sparkline(self, player_id, n=60):
        return list(self.speed_history[player_id][-n:])


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_scale(zone_w, zone_h, margin=20):
    sx = (zone_w - 2 * margin) / PITCH_W_M
    sy = (zone_h - 2 * margin) / PITCH_H_M
    return sx, sy


def _to_px(real_x, real_y, sx, sy, margin, zone_w, zone_h):
    px = int(real_x * sx) + margin
    py = int(real_y * sy) + margin
    px = max(0, min(zone_w - 1, px))
    py = max(0, min(zone_h - 1, py))
    return px, py


# ─────────────────────────────────────────────────────────────────────────────
# Pitch markings
# ─────────────────────────────────────────────────────────────────────────────

def draw_pitch_markings(zone, zone_w, zone_h, margin=20):
    sx, sy = _make_scale(zone_w, zone_h, margin)

    def tp(rx, ry):
        return _to_px(rx, ry, sx, sy, margin, zone_w, zone_h)

    overlay = zone.copy()

    # Outer boundary
    cv2.rectangle(overlay, tp(0, 0), tp(PITCH_W_M, PITCH_H_M), PITCH_LINE, 2)

    # Halfway line (depth mid)
    mid = PITCH_W_M / 2
    cv2.line(overlay, tp(mid, 0), tp(mid, PITCH_H_M), PITCH_LINE, 1)

    # Centre circle  (radius 9.15m in world → pixels)
    cx, cy = tp(mid, PITCH_H_M / 2)
    r_px = int(9.15 * sx)
    cv2.circle(overlay, (cx, cy), r_px, PITCH_LINE, 1)
    cv2.circle(overlay, (cx, cy), 3, PITCH_LINE, -1)

    # Left penalty box  (x: 0..16.5, y: 13.85..54.15)
    if 16.5 <= PITCH_W_M:
        cv2.rectangle(overlay, tp(0, 13.85), tp(16.5, 54.15), PITCH_LINE, 1)
        cv2.rectangle(overlay, tp(0, 24.85), tp(5.5, 43.15), PITCH_LINE, 1)
        lcx, lcy = tp(11.0, PITCH_H_M / 2)
        cv2.circle(overlay, (lcx, lcy), 3, PITCH_LINE, -1)

    # Right penalty box
    r_start = PITCH_W_M - 16.5
    if r_start >= 0:
        cv2.rectangle(overlay, tp(r_start, 13.85), tp(PITCH_W_M, 54.15), PITCH_LINE, 1)
        cv2.rectangle(overlay, tp(PITCH_W_M - 5.5, 24.85), tp(PITCH_W_M, 43.15), PITCH_LINE, 1)
        rcx, rcy = tp(PITCH_W_M - 11.0, PITCH_H_M / 2)
        cv2.circle(overlay, (rcx, rcy), 3, PITCH_LINE, -1)

    # Zone dividers (thirds of visible section)
    for xd in [PITCH_W_M / 3, 2 * PITCH_W_M / 3]:
        cv2.line(overlay, tp(xd, 0), tp(xd, PITCH_H_M), (180, 180, 180), 1)

    cv2.addWeighted(overlay, 0.30, zone, 0.70, 0, zone)


# ─────────────────────────────────────────────────────────────────────────────
# Velocity-aware Voronoi
# ─────────────────────────────────────────────────────────────────────────────

def draw_voronoi(zone, player_data, position_history, zone_w, zone_h, margin=20):
    sx, sy = _make_scale(zone_w, zone_h, margin)
    pitch_poly = Polygon([
        (0, 0), (PITCH_W_M, 0), (PITCH_W_M, PITCH_H_M), (0, PITCH_H_M)
    ])

    try:
        pts, teams = [], []

        for pid, info in player_data.items():
            pos = info.get('position_transformed')
            team = info.get('team')
            if pos is None or team not in (1, 2):
                continue

            speed_ms = info.get('speed', 0) / 3.6
            hist = list(position_history.get(pid, []))

            vx = vy = 0.0
            if len(hist) >= 2:
                ref = hist[max(0, len(hist) - 5)]
                dx = pos[0] - ref[0]
                dy = pos[1] - ref[1]
                norm = sqrt(dx * dx + dy * dy) or 1.0
                vx = dx / norm * speed_ms
                vy = dy / norm * speed_ms

            proj_x = max(0, min(PITCH_W_M, pos[0] + vx * 0.5))
            proj_y = max(0, min(PITCH_H_M, pos[1] + vy * 0.5))
            pts.append([proj_x, proj_y])
            teams.append(team)

        if len(pts) < 4:
            return

        pts_arr = np.array(pts)
        large = max(PITCH_W_M, PITCH_H_M) * 15
        cx0, cy0 = PITCH_W_M / 2, PITCH_H_M / 2
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        bounds = np.array([[cx0 + large * np.cos(a), cy0 + large * np.sin(a)] for a in angles])
        all_pts = np.vstack([pts_arr, bounds])

        vor = Voronoi(all_pts)

        for i, (team, color) in enumerate(zip(teams, [T1_FILL if t == 1 else T2_FILL for t in teams])):
            ridx = vor.point_region[i]
            region = vor.regions[ridx]
            if not region or -1 in region or len(region) < 3:
                continue

            verts = vor.vertices[region]
            try:
                poly = Polygon(verts)
                clipped = poly.intersection(pitch_poly)
            except Exception:
                continue

            if clipped.is_empty:
                continue

            polys = list(clipped.geoms) if isinstance(clipped, MultiPolygon) else [clipped]

            for subpoly in polys:
                if subpoly.is_empty or not hasattr(subpoly, 'exterior'):
                    continue
                coords = list(subpoly.exterior.coords)
                px_coords = np.array([
                    [int(c[0] * sx) + margin, int(c[1] * sy) + margin]
                    for c in coords
                ], dtype=np.int32)

                ov = zone.copy()
                cv2.fillPoly(ov, [px_coords], color)
                cv2.addWeighted(ov, 0.13, zone, 0.87, 0, zone)
                cv2.polylines(zone, [px_coords], True, color, 1)

    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Pressure heatmap
# ─────────────────────────────────────────────────────────────────────────────

def _make_heat(buffer, zone_w, zone_h, sx, sy, margin, sigma=22):
    heat = np.zeros((zone_h, zone_w), dtype=np.float32)
    for rx, ry in buffer:
        px = max(0, min(zone_w - 1, int(rx * sx) + margin))
        py = max(0, min(zone_h - 1, int(ry * sy) + margin))
        heat[py, px] += 1.0
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if heat.max() > 0:
        heat = (heat / heat.max() * 255).astype(np.uint8)
    else:
        heat = heat.astype(np.uint8)
    return heat


def draw_pressure_wash(zone, buf_t1, buf_t2, zone_w, zone_h, alpha=0.28, margin=20):
    if not buf_t1 and not buf_t2:
        return
    sx, sy = _make_scale(zone_w, zone_h, margin)

    if buf_t1:
        h1 = _make_heat(buf_t1, zone_w, zone_h, sx, sy, margin)
        bgr1 = np.zeros((zone_h, zone_w, 3), dtype=np.uint8)
        bgr1[:, :, 0] = h1  # blue channel
        cv2.addWeighted(zone, 1.0, bgr1, alpha, 0, zone)

    if buf_t2:
        h2 = _make_heat(buf_t2, zone_w, zone_h, sx, sy, margin)
        bgr2 = np.zeros((zone_h, zone_w, 3), dtype=np.uint8)
        bgr2[:, :, 2] = h2  # red channel
        cv2.addWeighted(zone, 1.0, bgr2, alpha, 0, zone)


def draw_pressure_heatmap_full(zone, buf_t1, buf_t2, zone_w, zone_h, alpha=0.55, margin=20):
    if not buf_t1 and not buf_t2:
        return
    sx, sy = _make_scale(zone_w, zone_h, margin)

    h1 = _make_heat(buf_t1, zone_w, zone_h, sx, sy, margin) if buf_t1 else np.zeros((zone_h, zone_w), dtype=np.uint8)
    h2 = _make_heat(buf_t2, zone_w, zone_h, sx, sy, margin) if buf_t2 else np.zeros((zone_h, zone_w), dtype=np.uint8)

    bgr1 = np.zeros((zone_h, zone_w, 3), dtype=np.uint8)
    bgr1[:, :, 0] = h1
    bgr2 = np.zeros((zone_h, zone_w, 3), dtype=np.uint8)
    bgr2[:, :, 2] = h2

    cv2.addWeighted(zone, 1.0, bgr1, alpha, 0, zone)
    cv2.addWeighted(zone, 1.0, bgr2, alpha, 0, zone)

    # Contested zone (yellow) where both teams overlap
    h1f = h1.astype(np.float32) / 255.0
    h2f = h2.astype(np.float32) / 255.0
    combined = (h1f + h2f) / 2
    contested = (combined > 0.6).astype(np.uint8)
    yellow = np.zeros((zone_h, zone_w, 3), dtype=np.uint8)
    yellow[:, :, 1] = contested * 200
    yellow[:, :, 2] = contested * 200
    cv2.addWeighted(zone, 0.7, yellow, 0.3, 0, zone)


# ─────────────────────────────────────────────────────────────────────────────
# Formation & phase
# ─────────────────────────────────────────────────────────────────────────────

def detect_formation_and_phase(player_data, ball_data, team, frame_num, team_ball_control):
    players = [
        info for info in player_data.values()
        if info.get('team') == team and info.get('position_transformed') is not None
    ]
    if not players:
        return "N/A", "N/A"

    xs = [p['position_transformed'][0] for p in players]
    xs_sorted = sorted(xs)

    # Group into lines by natural gaps > 2.5m
    line_id, line_of = 0, [0]
    for i in range(1, len(xs_sorted)):
        if xs_sorted[i] - xs_sorted[i - 1] > 2.5:
            line_id += 1
        line_of.append(line_id)

    counts = [line_of.count(l) for l in range(line_id + 1)]
    # Exclude GK line (single player deepest for team1 or shallowest for team2)
    outfield = counts[1:] if len(counts) > 1 else counts
    formation = "-".join(str(c) for c in outfield) if outfield else "N/A"

    # Phase of play
    ball_pos = None
    if ball_data and 1 in ball_data:
        ball_pos = ball_data[1].get('position_transformed')

    avg_x = np.mean(xs) if xs else 0
    mid = PITCH_W_M / 2
    t_has_ball = (frame_num < len(team_ball_control) and
                  int(team_ball_control[frame_num]) == team)

    if ball_pos is not None:
        bx = ball_pos[0]
        if team == 1:
            if bx > 0.67 * PITCH_W_M and t_has_ball:
                phase = "ATK"
            elif bx < 0.33 * PITCH_W_M and not t_has_ball:
                phase = "DEF"
            elif avg_x > 0.57 * PITCH_W_M:
                phase = "PRESS"
            else:
                phase = "BUILD"
        else:
            if bx < 0.33 * PITCH_W_M and t_has_ball:
                phase = "ATK"
            elif bx > 0.67 * PITCH_W_M and not t_has_ball:
                phase = "DEF"
            elif avg_x < 0.43 * PITCH_W_M:
                phase = "PRESS"
            else:
                phase = "BUILD"
    else:
        phase = "ATK" if (team == 1 and avg_x > mid) or (team == 2 and avg_x < mid) else "DEF"

    return formation, phase


def smooth_formation(state, f1, f2):
    """Only update displayed formation after 30 consecutive identical frames."""
    for key, new_val in [('t1', f1), ('t2', f2)]:
        count_key = key + '_count'
        cand_key = key + '_cand'
        if state.get(cand_key) == new_val:
            state[count_key] = state.get(count_key, 0) + 1
        else:
            state[cand_key] = new_val
            state[count_key] = 1
        if state[count_key] >= 30:
            state[key] = new_val


# ─────────────────────────────────────────────────────────────────────────────
# Formation lines (dashed)
# ─────────────────────────────────────────────────────────────────────────────

def _dashed_line(frame, pt1, pt2, color, thickness=1, dash=8, gap=5):
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    length = sqrt(dx * dx + dy * dy)
    if length == 0:
        return
    seg = dash + gap
    steps = int(length / seg)
    for i in range(steps):
        t0 = i * seg / length
        t1_ = min((i * seg + dash) / length, 1.0)
        s = (int(pt1[0] + dx * t0), int(pt1[1] + dy * t0))
        e = (int(pt1[0] + dx * t1_), int(pt1[1] + dy * t1_))
        cv2.line(frame, s, e, color, thickness)


def draw_formation_lines(zone, player_data, zone_w, zone_h, margin=20):
    sx, sy = _make_scale(zone_w, zone_h, margin)

    for team, color in [(1, T1_FORM), (2, T2_FORM)]:
        players = [
            (pid, info) for pid, info in player_data.items()
            if info.get('team') == team and info.get('position_transformed') is not None
        ]
        if len(players) < 2:
            continue

        # Sort by depth then group into lines
        players.sort(key=lambda p: p[1]['position_transformed'][0],
                     reverse=(team == 2))
        xs = [p[1]['position_transformed'][0] for p in players]

        line_id, line_of = 0, [0]
        for i in range(1, len(xs)):
            if abs(xs[i] - xs[i - 1]) > 2.5:
                line_id += 1
            line_of.append(line_id)

        by_line = defaultdict(list)
        for (pid, info), la in zip(players, line_of):
            by_line[la].append(info)

        for line_players in by_line.values():
            if len(line_players) < 2:
                continue
            sorted_p = sorted(line_players, key=lambda i: i['position_transformed'][1])
            pts = [_to_px(*p['position_transformed'], sx, sy, margin, zone_w, zone_h)
                   for p in sorted_p]
            ov = zone.copy()
            for i in range(len(pts) - 1):
                _dashed_line(ov, pts[i], pts[i + 1], color, thickness=2)
            cv2.addWeighted(ov, 0.75, zone, 0.25, 0, zone)


# ─────────────────────────────────────────────────────────────────────────────
# Players on tactical map
# ─────────────────────────────────────────────────────────────────────────────

def draw_players_on_map(zone, player_data, position_history, fatigue, zone_w, zone_h, margin=20):
    sx, sy = _make_scale(zone_w, zone_h, margin)

    for pid, info in player_data.items():
        pos = info.get('position_transformed')
        team = info.get('team')
        if pos is None or team not in (1, 2):
            continue

        px, py = _to_px(*pos, sx, sy, margin, zone_w, zone_h)
        speed_kmh = info.get('speed', 0)
        speed_ms = speed_kmh / 3.6
        dot_color = T1_DOT if team == 1 else T2_DOT

        # Velocity arrow
        if speed_kmh > 2:
            hist = list(position_history.get(pid, []))
            vx = vy = 0.0
            if len(hist) >= 2:
                ref = hist[max(0, len(hist) - 5)]
                ddx = pos[0] - ref[0]
                ddy = pos[1] - ref[1]
                norm = sqrt(ddx * ddx + ddy * ddy) or 1.0
                vx, vy = ddx / norm * speed_ms, ddy / norm * speed_ms
            arr_len = int(speed_ms * 3)
            ex = max(0, min(zone_w - 1, int(px + vy * sx * 0.35)))
            ey = max(0, min(zone_h - 1, int(py + vx * sy * 0.35)))
            if (ex, ey) != (px, py):
                cv2.arrowedLine(zone, (px, py), (ex, ey), dot_color, 1,
                                tipLength=0.35, line_type=cv2.LINE_AA)

        # Gassed pulse ring
        if fatigue.is_gassed(pid):
            pulse = True  # simple static ring on map
            cv2.circle(zone, (px, py), 14, (0, 0, 220), 2)

        # Player dot
        cv2.circle(zone, (px, py), 9, dot_color, -1)
        cv2.circle(zone, (px, py), 10, (255, 255, 255), 1)
        cv2.putText(zone, str(pid), (px - 5, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)


def draw_player_dots_only(zone, player_data, zone_w, zone_h, margin=20):
    sx, sy = _make_scale(zone_w, zone_h, margin)
    for pid, info in player_data.items():
        pos = info.get('position_transformed')
        team = info.get('team')
        if pos is None or team not in (1, 2):
            continue
        px, py = _to_px(*pos, sx, sy, margin, zone_w, zone_h)
        color = T1_DOT if team == 1 else T2_DOT
        cv2.circle(zone, (px, py), 3, color, -1)


def draw_ball_on_map(zone, ball_data, zone_w, zone_h, margin=20):
    sx, sy = _make_scale(zone_w, zone_h, margin)
    binfo = ball_data.get(1, {}) if ball_data else {}
    pos = binfo.get('position_transformed')
    if pos is None:
        return
    px, py = _to_px(*pos, sx, sy, margin, zone_w, zone_h)
    cv2.circle(zone, (px, py), 5, (255, 255, 255), -1)
    cv2.circle(zone, (px, py), 5, (120, 120, 120), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Zone A header + legend
# ─────────────────────────────────────────────────────────────────────────────

def draw_zone_a_header(zone, zone_w, formation_smooth, phase_t1, phase_t2, frame_num, fps):
    ov = zone.copy()
    cv2.rectangle(ov, (0, 0), (zone_w, 28), (10, 26, 10), -1)
    cv2.addWeighted(ov, 0.9, zone, 0.1, 0, zone)

    cv2.putText(zone, "TACTICAL MAP", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    f1 = formation_smooth.get('t1', 'N/A')
    f2 = formation_smooth.get('t2', 'N/A')
    mid_text = f"T1: {f1} [{phase_t1}]  |  T2: {f2} [{phase_t2}]"
    tw = cv2.getTextSize(mid_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    cx_ = zone_w // 2 - tw // 2
    cv2.putText(zone, f"T1: {f1} [{phase_t1}]", (cx_, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, T1_FORM, 1)
    t1_w = cv2.getTextSize(f"T1: {f1} [{phase_t1}]  |  ", cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    cv2.putText(zone, f"T2: {f2} [{phase_t2}]", (cx_ + t1_w, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, T2_FORM, 1)

    t_sec = frame_num / fps
    ts = f"{int(t_sec // 60):02d}:{int(t_sec % 60):02d}"
    tw2 = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    cv2.putText(zone, ts, (zone_w - tw2 - 8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)


def draw_zone_a_legend(zone, zone_h):
    x0, y0 = 8, zone_h - 60
    ov = zone.copy()
    cv2.rectangle(ov, (x0 - 2, y0 - 2), (x0 + 135, y0 + 58), (10, 10, 10), -1)
    cv2.addWeighted(ov, 0.6, zone, 0.4, 0, zone)

    items = [
        (T1_FORM, "T1 formation"),
        (T2_FORM, "T2 formation"),
        (T1_FILL, "T1 Voronoi"),
        (T2_FILL, "T2 Voronoi"),
    ]
    for i, (col, label) in enumerate(items):
        y = y0 + 12 + i * 13
        cv2.rectangle(zone, (x0, y - 4), (x0 + 14, y + 4), col, -1)
        cv2.putText(zone, label, (x0 + 18, y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (210, 210, 210), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Zone B header + PPDA
# ─────────────────────────────────────────────────────────────────────────────

def draw_zone_b_header(zone, zone_w, zone_b_h, ppda_t1, ppda_t2,
                       team_ball_control, frame_num):
    ov = zone.copy()
    cv2.rectangle(ov, (0, 0), (zone_w, 28), (10, 26, 10), -1)
    cv2.addWeighted(ov, 0.9, zone, 0.1, 0, zone)

    cv2.putText(zone, "PRESSURE MAP — last 30s", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    def ppda_color(v):
        return (0, 200, 0) if v < 8 else ((0, 165, 255) if v < 12 else (0, 0, 220))

    p1_str = f"T1 PPDA: {ppda_t1:.1f}" if ppda_t1 > 0 else "T1 PPDA: --"
    p2_str = f"T2 PPDA: {ppda_t2:.1f}" if ppda_t2 > 0 else "T2 PPDA: --"
    tw1 = cv2.getTextSize(p1_str + "  |  " + p2_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    rx = zone_w - tw1 - 8
    cv2.putText(zone, p1_str, (rx, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ppda_color(ppda_t1), 1)
    off = cv2.getTextSize(p1_str + "  |  ", cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    cv2.putText(zone, p2_str, (rx + off, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ppda_color(ppda_t2), 1)

    # Bottom ball-control bar
    tbc = team_ball_control[:frame_num + 1] if frame_num < len(team_ball_control) else team_ball_control
    total = int((tbc == 1).sum() + (tbc == 2).sum())
    t1_pct = 100 * int((tbc == 1).sum()) / total if total else 50
    t2_pct = 100 - t1_pct

    bar_h = 20
    bar_y = zone_b_h - bar_h
    t1_w = int(zone_w * t1_pct / 100)
    cv2.rectangle(zone, (0, bar_y), (t1_w, zone_b_h), T1_DOT, -1)
    cv2.rectangle(zone, (t1_w, bar_y), (zone_w, zone_b_h), T2_DOT, -1)

    label = f"Ball control:  T1 {t1_pct:.0f}%     T2 {t2_pct:.0f}%"
    cv2.putText(zone, label, (8, zone_b_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Fatigue sidebar
# ─────────────────────────────────────────────────────────────────────────────

def draw_fatigue_sidebar(canvas, fatigue, player_data, frame_num, left_w, canvas_h, sidebar_w=220):
    x0 = left_w
    # Dark background
    ov = canvas.copy()
    cv2.rectangle(ov, (x0, 0), (x0 + sidebar_w, canvas_h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.72, canvas, 0.28, 0, canvas)
    cv2.line(canvas, (x0, 0), (x0, canvas_h), (60, 60, 60), 1)

    # Header
    cv2.putText(canvas, "FITNESS", (x0 + sidebar_w // 2 - 28, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.line(canvas, (x0, 30), (x0 + sidebar_w, 30), (60, 60, 60), 1)

    # Collect and sort player IDs
    pids = sorted(player_data.keys())
    n = len(pids)
    if n == 0:
        return

    footer_h = 80
    available_h = canvas_h - 34 - footer_h
    row_h = max(38, min(60, available_h // max(n, 1)))

    y_cursor = 34

    t1_speeds, t2_speeds = [], []
    gassed_count = 0

    for pid in pids:
        if y_cursor + row_h > canvas_h - footer_h:
            break

        info = player_data.get(pid, {})
        team = info.get('team', 0)
        speed = info.get('speed', 0)
        reserve = fatigue.get_reserve_pct(pid)
        gassed = fatigue.is_gassed(pid)
        critical = fatigue.is_critical(pid)

        if team == 1:
            t1_speeds.append(speed)
        elif team == 2:
            t2_speeds.append(speed)
        if gassed:
            gassed_count += 1

        # ── Battery bar ──────────────────────────────────────────
        bar_x = x0 + 8
        bar_y = y_cursor + (row_h - 38) // 2
        bar_w, bar_h_px = 10, 38
        fill_level = reserve / 100.0

        if reserve >= 80:
            fill_col = (34, 170, 34)
        elif reserve >= 65:
            fill_col = (34, 165, 210)
        else:
            fill_col = (34, 34, 220)

        # Outer border
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_px),
                      (120, 120, 120), 1)
        # Battery tip
        tip_x = bar_x + (bar_w - 4) // 2
        cv2.rectangle(canvas, (tip_x, bar_y - 3), (tip_x + 4, bar_y), (120, 120, 120), 1)
        # Fill from bottom
        fill_px = int(fill_level * bar_h_px)
        if fill_px > 0:
            cv2.rectangle(canvas,
                          (bar_x + 1, bar_y + bar_h_px - fill_px),
                          (bar_x + bar_w - 1, bar_y + bar_h_px - 1),
                          fill_col, -1)
        # Gassed red glow
        if gassed:
            ov2 = canvas.copy()
            cv2.rectangle(ov2, (bar_x - 1, bar_y - 1),
                          (bar_x + bar_w + 1, bar_y + bar_h_px + 1), (0, 0, 180), 1)
            cv2.addWeighted(ov2, 0.5, canvas, 0.5, 0, canvas)

        # ── Text block ─────────────────────────────────────────────
        tx = x0 + 24
        ty = y_cursor + 12

        # Line 1: #ID + team dot
        cv2.putText(canvas, f"#{pid}", (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)
        dot_col = T1_DOT if team == 1 else T2_DOT if team == 2 else GK_DOT
        dot_x = tx + cv2.getTextSize(f"#{pid}", cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0] + 6
        cv2.circle(canvas, (dot_x, ty - 4), 4, dot_col, -1)

        # Line 2: reserve %
        ty += 13
        res_col = (34, 170, 34) if reserve >= 80 else ((34, 165, 210) if reserve >= 65 else (34, 34, 220))
        cv2.putText(canvas, f"{reserve}% reserve", (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, res_col, 1)

        # Line 3: current speed
        ty += 11
        cv2.putText(canvas, f"{speed:.1f} km/h", (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1)

        # Line 4: GASSED / CRITICAL
        ty += 11
        if critical:
            cv2.putText(canvas, "! SUB NOW", (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 220), 1)
        elif gassed:
            cv2.putText(canvas, "! GASSED", (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 165, 255), 1)

        # Line 5: sparkline
        spark = fatigue.get_sparkline(pid, 60)
        if len(spark) >= 2:
            sp_x = tx
            sp_y = y_cursor + row_h - 14
            sp_w, sp_h = min(sidebar_w - 30, 85), 10
            smax = max(max(spark), 1)
            pts_spark = []
            for si, sv in enumerate(spark):
                sx_ = sp_x + int(si * sp_w / len(spark))
                sy_ = sp_y + sp_h - int(sv / smax * sp_h)
                pts_spark.append((sx_, sy_))
            if len(pts_spark) >= 2:
                cv2.polylines(canvas, [np.array(pts_spark, dtype=np.int32)],
                              False, fill_col, 1)

        # Separator
        cv2.line(canvas, (x0, y_cursor + row_h - 1),
                 (x0 + sidebar_w, y_cursor + row_h - 1), (40, 40, 40), 1)
        y_cursor += row_h

    # ── Footer ──────────────────────────────────────────────────────
    fy = canvas_h - footer_h
    cv2.line(canvas, (x0, fy), (x0 + sidebar_w, fy), (60, 60, 60), 1)

    avg1 = f"{np.mean(t1_speeds):.1f} km/h" if t1_speeds else "--"
    avg2 = f"{np.mean(t2_speeds):.1f} km/h" if t2_speeds else "--"

    cv2.putText(canvas, "Team avg speed", (x0 + 6, fy + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1)
    cv2.putText(canvas, f"T1: {avg1}", (x0 + 6, fy + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, T1_FORM, 1)
    cv2.putText(canvas, f"T2: {avg2}", (x0 + 6, fy + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, T2_FORM, 1)
    cv2.putText(canvas, f"Gassed: {gassed_count}", (x0 + 6, fy + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 165, 255), 1)


# ─────────────────────────────────────────────────────────────────────────────
# PPDA tracker
# ─────────────────────────────────────────────────────────────────────────────

class PPDATracker:
    def __init__(self):
        self.def_actions_t1 = 0   # T1 defending (close to ball when T2 has it)
        self.def_actions_t2 = 0
        self.poss_changes = 0
        self.prev_control = 0

    def update(self, frame_num, player_data, ball_data, team_ball_control):
        if frame_num >= len(team_ball_control):
            return
        ctrl = int(team_ball_control[frame_num])
        if self.prev_control != 0 and ctrl != 0 and ctrl != self.prev_control:
            self.poss_changes += 1
        self.prev_control = ctrl

        ball_pos = None
        if ball_data and 1 in ball_data:
            ball_pos = ball_data[1].get('position_transformed')
        if ball_pos is None:
            return

        for info in player_data.values():
            team = info.get('team')
            pos = info.get('position_transformed')
            if pos is None or team not in (1, 2):
                continue
            dist = sqrt((pos[0] - ball_pos[0]) ** 2 + (pos[1] - ball_pos[1]) ** 2)
            if dist < 3.0:
                if team == 1 and ctrl == 2:
                    self.def_actions_t1 += 1
                elif team == 2 and ctrl == 1:
                    self.def_actions_t2 += 1

    def get_ppda(self):
        opp_passes = max(1, self.poss_changes // 2)
        ppda_t1 = self.def_actions_t1 / opp_passes
        ppda_t2 = self.def_actions_t2 / opp_passes
        return ppda_t1, ppda_t2


# ─────────────────────────────────────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────────────────────────────────────

def render_tactical_video(video_frames, tracks, team_ball_control, fps):
    h, w = video_frames[0].shape[:2]
    canvas_w = w + SIDEBAR_W
    canvas_h = h
    zone_a_h = h // 2
    zone_b_h = h - zone_a_h

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_videos/tactical_video.avi',
                          fourcc, fps, (canvas_w, canvas_h))

    fatigue = FatigueTracker(fps)
    ppda_tracker = PPDATracker()
    buf_t1 = deque(maxlen=fps * 30)
    buf_t2 = deque(maxlen=fps * 30)
    pos_hist = defaultdict(lambda: deque(maxlen=10))
    formation_smooth = {'t1': 'N/A', 't2': 'N/A',
                        't1_cand': '', 't2_cand': '',
                        't1_count': 0, 't2_count': 0}

    total = len(video_frames)

    for frame_num in range(total):
        if frame_num % 50 == 0:
            print(f"Rendering frame {frame_num}/{total}...")

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        player_data = tracks['players'][frame_num] if frame_num < len(tracks['players']) else {}
        ball_data   = tracks['ball'][frame_num]    if frame_num < len(tracks['ball'])    else {}

        # ── Update trackers ──────────────────────────────────────────
        for pid, pdata in player_data.items():
            spd = pdata.get('speed', 0)
            fatigue.update(frame_num, pid, spd)
            pos = pdata.get('position_transformed')
            if pos is not None:
                pos_hist[pid].append(pos)
                if pdata.get('team') == 1:
                    buf_t1.append(pos)
                elif pdata.get('team') == 2:
                    buf_t2.append(pos)

        ppda_tracker.update(frame_num, player_data, ball_data, team_ball_control)
        ppda_t1, ppda_t2 = ppda_tracker.get_ppda()

        formation_t1, phase_t1 = detect_formation_and_phase(
            player_data, ball_data, 1, frame_num, team_ball_control)
        formation_t2, phase_t2 = detect_formation_and_phase(
            player_data, ball_data, 2, frame_num, team_ball_control)
        smooth_formation(formation_smooth, formation_t1, formation_t2)

        # ── Zone A ───────────────────────────────────────────────────
        zone_a = np.full((zone_a_h, w, 3), (13, 43, 13), dtype=np.uint8)
        draw_pitch_markings(zone_a, w, zone_a_h, margin=20)
        draw_voronoi(zone_a, player_data, pos_hist, w, zone_a_h, margin=20)
        draw_pressure_wash(zone_a, buf_t1, buf_t2, w, zone_a_h, alpha=0.28, margin=20)
        draw_formation_lines(zone_a, player_data, w, zone_a_h, margin=20)
        draw_players_on_map(zone_a, player_data, pos_hist, fatigue, w, zone_a_h, margin=20)
        draw_ball_on_map(zone_a, ball_data, w, zone_a_h, margin=20)
        draw_zone_a_header(zone_a, w, formation_smooth,
                           phase_t1, phase_t2, frame_num, fps)
        draw_zone_a_legend(zone_a, zone_a_h)

        # ── Zone B ───────────────────────────────────────────────────
        zone_b = np.full((zone_b_h, w, 3), (13, 43, 13), dtype=np.uint8)
        draw_pitch_markings(zone_b, w, zone_b_h, margin=20)
        draw_pressure_heatmap_full(zone_b, buf_t1, buf_t2, w, zone_b_h, alpha=0.55, margin=20)
        draw_player_dots_only(zone_b, player_data, w, zone_b_h, margin=20)
        draw_zone_b_header(zone_b, w, zone_b_h, ppda_t1, ppda_t2,
                           team_ball_control, frame_num)

        # ── Compose canvas ───────────────────────────────────────────
        canvas[0:zone_a_h, 0:w] = zone_a
        canvas[zone_a_h:h, 0:w] = zone_b

        # ── Sidebar ──────────────────────────────────────────────────
        draw_fatigue_sidebar(canvas, fatigue, player_data,
                             frame_num, w, canvas_h, SIDEBAR_W)

        out.write(canvas)

    out.release()
    print(f"tactical_video.avi saved ({total} frames).")
