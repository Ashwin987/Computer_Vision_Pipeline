"""
analysis_utils.py — Shared infrastructure for all output renderers.
"""

import cv2
import numpy as np
from collections import defaultdict, deque

PITCH_D_M = 23.32
PITCH_W_M = 68.0


# ─────────────────────────────────────────────────────────────────────────────
# FatigueTracker
# ─────────────────────────────────────────────────────────────────────────────

class FatigueTracker:
    """Tracks per-player fatigue metrics throughout a video."""

    def __init__(self, fps):
        self.fps = fps
        self.speed_history = defaultdict(list)
        self.game_max = {}
        self.sprint_max_2min = {}
        self.meta_power_hist = defaultdict(list)
        self.prev_speed_ms = {}

    def update(self, frame_num, player_id, speed_kmh):
        speed_ms = speed_kmh / 3.6
        self.speed_history[player_id].append(speed_kmh)
        self.game_max[player_id] = max(self.game_max.get(player_id, 0), speed_kmh)
        window = self.fps * 120
        recent = self.speed_history[player_id][-window:]
        self.sprint_max_2min[player_id] = max(recent) if recent else 0
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
# HomographyProjector
# ─────────────────────────────────────────────────────────────────────────────

class HomographyProjector:
    """
    Wraps the ViewTransformer homography for real-world <-> pixel projection.

    Coordinate system (from view_transformer.py):
      pixel : (col, row) in camera frame
      real  : (x, y) where x = depth 0..23.32 m, y = width 0..68 m
    """

    def __init__(self, view_transformer):
        self.H_fwd = view_transformer.persepctive_trasnformer.copy()
        self.H_inv = np.linalg.inv(self.H_fwd)

    def real_to_pixel(self, x, y):
        """Map real-world (x=depth, y=width) → camera pixel (col, row)."""
        pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self.H_inv)
        col, row = result[0, 0]
        return int(col), int(row)

    def get_topdown_warp_matrix(self, pitch_left, pitch_top, pitch_w_px, pitch_d_px):
        """
        Returns M for cv2.warpPerspective(topdown_img, M, (cam_w, cam_h)).

        The 4 pitch corners in the top-down canvas:
          (pitch_left,          pitch_top)          -> real (x=0,     y=0)
          (pitch_left+w_px,     pitch_top)          -> real (x=0,     y=68)
          (pitch_left,          pitch_top+d_px)     -> real (x=23.32, y=0)
          (pitch_left+w_px,     pitch_top+d_px)     -> real (x=23.32, y=68)
        """
        src_pts = np.float32([
            [pitch_left,               pitch_top],
            [pitch_left + pitch_w_px,  pitch_top],
            [pitch_left,               pitch_top + pitch_d_px],
            [pitch_left + pitch_w_px,  pitch_top + pitch_d_px],
        ])
        real_corners = np.float32([
            [0.0,   0.0],
            [0.0,  68.0],
            [23.32, 0.0],
            [23.32, 68.0],
        ])
        dst_pts = cv2.perspectiveTransform(
            real_corners.reshape(1, -1, 2), self.H_inv
        ).reshape(-1, 2)
        return cv2.getPerspectiveTransform(src_pts, dst_pts)


# ─────────────────────────────────────────────────────────────────────────────
# FormationDetector
# ─────────────────────────────────────────────────────────────────────────────

class FormationDetector:
    """
    Gap-based formation detector with 30-frame smoothing per team.
    Returns formation string and line assignments.
    """

    LINE_GAP_M = 2.5

    def __init__(self):
        self._smooth = {}

    def detect(self, positions_x):
        """
        positions_x : list of float (pitch-x = depth, 0..23.32 m)
        Returns (formation_str, assignments) where assignments[i] = line_index.
        """
        if len(positions_x) < 2:
            return "N/A", list(range(len(positions_x)))

        order = np.argsort(positions_x)
        xs_sorted = np.array(positions_x)[order]

        line_id = 0
        sorted_line = [0]
        for i in range(1, len(xs_sorted)):
            if xs_sorted[i] - xs_sorted[i - 1] > self.LINE_GAP_M:
                line_id += 1
            sorted_line.append(line_id)

        n_lines = line_id + 1
        counts = [sorted_line.count(l) for l in range(n_lines)]
        formation_str = "-".join(str(c) for c in counts)

        assignments = [0] * len(positions_x)
        for sorted_idx, orig_idx in enumerate(order):
            assignments[orig_idx] = sorted_line[sorted_idx]

        return formation_str, assignments

    def smooth(self, team, new_val, threshold=30):
        """Returns stable (displayed) formation string for the team."""
        if team not in self._smooth:
            self._smooth[team] = {'label': new_val, 'candidate': new_val, 'count': 0}
        s = self._smooth[team]
        if s['candidate'] == new_val:
            s['count'] += 1
        else:
            s['candidate'] = new_val
            s['count'] = 1
        if s['count'] >= threshold:
            s['label'] = new_val
        return s['label']

    def get_phase(self, positions_x, midpoint=11.66):
        """Return 'ATK' if team's average depth > midpoint, else 'DEF'."""
        if not positions_x:
            return "N/A"
        return "ATK" if float(np.mean(positions_x)) > midpoint else "DEF"


# ─────────────────────────────────────────────────────────────────────────────
# draw_fatigue_sidebar
# ─────────────────────────────────────────────────────────────────────────────

_T1_DOT  = (180,  80,  20)
_T2_DOT  = ( 30,  30, 180)
_GK_DOT  = (200,  50, 150)   # purple, distinct from referee-yellow
_T1_FORM = (255, 180,  80)
_T2_FORM = ( 80, 130, 255)


def draw_fatigue_sidebar(canvas, fatigue, player_data, frame_num,
                         left_w, canvas_h, sidebar_w=220, team_filter=None):
    """
    Render a fitness sidebar on canvas starting at x=left_w, width=sidebar_w.
    team_filter: 1 or 2 to show only that team's players, or None for all.
    """
    x0 = left_w
    ov = canvas.copy()
    cv2.rectangle(ov, (x0, 0), (x0 + sidebar_w, canvas_h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.72, canvas, 0.28, 0, canvas)
    cv2.line(canvas, (x0, 0), (x0, canvas_h), (60, 60, 60), 1)

    label = f"FITNESS T{team_filter}" if team_filter else "FITNESS"
    tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    cv2.putText(canvas, label, (x0 + (sidebar_w - tw) // 2, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.line(canvas, (x0, 30), (x0 + sidebar_w, 30), (60, 60, 60), 1)

    pids = sorted(
        pid for pid, info in player_data.items()
        if team_filter is None or info.get('team') == team_filter
    )
    n = len(pids)
    if n == 0:
        return

    footer_h = 80
    available_h = canvas_h - 34 - footer_h
    row_h = max(38, min(60, available_h // max(n, 1)))
    y_cursor = 34
    speeds, gassed_count = [], 0

    for pid in pids:
        if y_cursor + row_h > canvas_h - footer_h:
            break
        info = player_data.get(pid, {})
        team = info.get('team', 0)
        speed = info.get('speed', 0)
        reserve = fatigue.get_reserve_pct(pid)
        gassed = fatigue.is_gassed(pid)
        critical = fatigue.is_critical(pid)
        speeds.append(speed)
        if gassed:
            gassed_count += 1

        # Battery bar
        bar_x, bar_y = x0 + 8, y_cursor + (row_h - 38) // 2
        bar_w, bar_h_px = 10, 38
        if reserve >= 80:   fill_col = (34, 170,  34)
        elif reserve >= 65: fill_col = (34, 165, 210)
        else:               fill_col = (34,  34, 220)

        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_px), (120, 120, 120), 1)
        tip_x = bar_x + (bar_w - 4) // 2
        cv2.rectangle(canvas, (tip_x, bar_y - 3), (tip_x + 4, bar_y), (120, 120, 120), 1)
        fill_px = int((reserve / 100.0) * bar_h_px)
        if fill_px > 0:
            cv2.rectangle(canvas,
                          (bar_x + 1, bar_y + bar_h_px - fill_px),
                          (bar_x + bar_w - 1, bar_y + bar_h_px - 1),
                          fill_col, -1)
        if gassed:
            ov2 = canvas.copy()
            cv2.rectangle(ov2, (bar_x - 1, bar_y - 1),
                          (bar_x + bar_w + 1, bar_y + bar_h_px + 1), (0, 0, 180), 1)
            cv2.addWeighted(ov2, 0.5, canvas, 0.5, 0, canvas)

        tx, ty = x0 + 24, y_cursor + 12
        dot_col = _T1_DOT if team == 1 else _T2_DOT if team == 2 else _GK_DOT
        cv2.putText(canvas, f"#{pid}", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)
        dot_x = tx + cv2.getTextSize(f"#{pid}", cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0] + 6
        cv2.circle(canvas, (dot_x, ty - 4), 4, dot_col, -1)

        ty += 13
        res_col = (34, 170, 34) if reserve >= 80 else ((34, 165, 210) if reserve >= 65 else (34, 34, 220))
        cv2.putText(canvas, f"{reserve}% reserve", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.30, res_col, 1)

        ty += 11
        cv2.putText(canvas, f"{speed:.1f} km/h", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1)

        ty += 11
        if critical:
            cv2.putText(canvas, "! SUB NOW", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 220), 1)
        elif gassed:
            cv2.putText(canvas, "! GASSED",  (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 165, 255), 1)

        spark = fatigue.get_sparkline(pid, 60)
        if len(spark) >= 2:
            sp_x, sp_y = tx, y_cursor + row_h - 14
            sp_w, sp_h = min(sidebar_w - 30, 85), 10
            smax = max(max(spark), 1)
            pts_spark = []
            for si, sv in enumerate(spark):
                sx_ = sp_x + int(si * sp_w / len(spark))
                sy_ = sp_y + sp_h - int(sv / smax * sp_h)
                pts_spark.append((sx_, sy_))
            if len(pts_spark) >= 2:
                cv2.polylines(canvas, [np.array(pts_spark, dtype=np.int32)], False, fill_col, 1)

        cv2.line(canvas, (x0, y_cursor + row_h - 1), (x0 + sidebar_w, y_cursor + row_h - 1), (40, 40, 40), 1)
        y_cursor += row_h

    # Footer
    fy = canvas_h - footer_h
    cv2.line(canvas, (x0, fy), (x0 + sidebar_w, fy), (60, 60, 60), 1)
    avg = f"{np.mean(speeds):.1f} km/h" if speeds else "--"
    t_col = _T1_FORM if team_filter == 1 else _T2_FORM if team_filter == 2 else (200, 200, 200)
    cv2.putText(canvas, "Avg speed", (x0 + 6, fy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 180, 180), 1)
    cv2.putText(canvas, avg,         (x0 + 6, fy + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.36, t_col, 1)
    cv2.putText(canvas, f"Gassed: {gassed_count}", (x0 + 6, fy + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 165, 255), 1)
