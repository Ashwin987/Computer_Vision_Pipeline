import cv2
import numpy as np
from tqdm import tqdm

from .formation_detector import FormationDetector
from .voronoi_control import VoronoiPitchControl

# ── Canvas / pitch layout ────────────────────────────────────────────────────
CANVAS_W  = 1100
CANVAS_H  = 600
HEADER_H  = 70
FOOTER_H  = 70
PITCH_PAD = 25        # horizontal gap between canvas edge and pitch edge

PITCH_W_M = 68.0      # pitch width in metres (full width)
PITCH_D_M = 23.32     # visible depth in metres (view-transformer section)

SCALE      = (CANVAS_W - 2 * PITCH_PAD) / PITCH_W_M   # ≈ 15.44 px/m
PITCH_W_PX = CANVAS_W - 2 * PITCH_PAD                  # 1050
PITCH_D_PX = int(PITCH_D_M * SCALE)                    # ≈ 360
PITCH_LEFT = PITCH_PAD
PITCH_TOP  = HEADER_H + ((CANVAS_H - HEADER_H - FOOTER_H - PITCH_D_PX) // 2)

# Voronoi blend alpha (0 = no Voronoi, 1 = full Voronoi)
VORONOI_ALPHA = 0.42

# Velocity arrow time horizon (seconds shown as arrow length)
ARROW_T = 0.35

# Velocity smoothing window (frames)
VEL_SMOOTH = 3


def _pitch_to_canvas(pitch_x, pitch_y):
    """Pitch (x=depth 0..23.32, y=width 0..68) → canvas (cx, cy)."""
    cx = int(PITCH_LEFT + pitch_y * SCALE)
    cy = int(PITCH_TOP  + pitch_x * SCALE)
    return cx, cy


def _build_pitch_bg():
    """Draw the static pitch background (green surface + white lines)."""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), (22, 22, 22), dtype=np.uint8)

    pr = (PITCH_LEFT, PITCH_TOP,
          PITCH_LEFT + PITCH_W_PX, PITCH_TOP + PITCH_D_PX)

    # Alternating stripe pattern
    stripe_w = int(PITCH_W_PX / 8)
    for i in range(8):
        x1 = PITCH_LEFT + i * stripe_w
        x2 = x1 + stripe_w
        col = (45, 118, 45) if i % 2 == 0 else (38, 100, 38)
        cv2.rectangle(canvas, (x1, PITCH_TOP), (x2, PITCH_TOP + PITCH_D_PX), col, -1)

    # Pitch border
    cv2.rectangle(canvas, (pr[0], pr[1]), (pr[2], pr[3]), (220, 220, 220), 2)

    # Halfway line (depth midpoint)
    mid_y = int(PITCH_TOP + (PITCH_D_M / 2) * SCALE)
    cv2.line(canvas, (PITCH_LEFT, mid_y),
             (PITCH_LEFT + PITCH_W_PX, mid_y), (180, 180, 180), 1)

    # Centre circle (approximate)
    cx_pitch = PITCH_LEFT + PITCH_W_PX // 2
    r_m = 9.15   # standard centre-circle radius
    cv2.circle(canvas, (cx_pitch, mid_y), int(r_m * SCALE), (160, 160, 160), 1)
    cv2.circle(canvas, (cx_pitch, mid_y), 3, (200, 200, 200), -1)

    return canvas


class TacticalMapGenerator:

    def __init__(self, fps=24):
        self.fps = fps
        self.formation_detector = FormationDetector()
        self._pitch_bg = _build_pitch_bg()

    # ------------------------------------------------------------------
    # Velocity pre-computation
    # ------------------------------------------------------------------

    def _compute_velocities(self, tracks):
        """Returns {player_id: {frame_num: np.array([vx, vy]) in m/s}}."""
        n = len(tracks['players'])
        w = VEL_SMOOTH

        # Collect position sequences
        pos_seq = {}
        for f in range(n):
            for pid, info in tracks['players'][f].items():
                p = info.get('position_transformed')
                if p is not None:
                    pos_seq.setdefault(pid, {})[f] = np.array(p, dtype=np.float32)

        vels = {}
        for pid, pos_by_frame in pos_seq.items():
            frames = sorted(pos_by_frame)
            vels[pid] = {}
            for i, f in enumerate(frames):
                f0 = frames[max(0, i - w)]
                f1 = frames[min(len(frames) - 1, i + w)]
                if f0 == f1:
                    vels[pid][f] = np.zeros(2, dtype=np.float32)
                else:
                    dt = (f1 - f0) / self.fps
                    vels[pid][f] = (pos_by_frame[f1] - pos_by_frame[f0]) / dt

        return vels

    # ------------------------------------------------------------------
    # Single-frame render
    # ------------------------------------------------------------------

    def _render_frame(self, frame_num, tracks, vels,
                      team_bgr, voronoi_ctrl,
                      possession_pct, transitions_so_far):

        canvas = self._pitch_bg.copy()

        # ── Collect player data ──────────────────────────────────────
        players_by_team = {1: [], 2: []}
        voronoi_input = []

        for pid, info in tracks['players'][frame_num].items():
            pos = info.get('position_transformed')
            team = info.get('team')
            if pos is None or team not in (1, 2):
                continue
            vel = vels.get(pid, {}).get(frame_num, np.zeros(2))
            players_by_team[team].append({'id': pid, 'pos': pos, 'vel': vel})
            voronoi_input.append((team, pos, vel))

        # ── Voronoi pitch control ────────────────────────────────────
        # Build softened team colours for Voronoi regions
        vor_bgr = {
            t: tuple(int(c * 0.55) for c in bgr)
            for t, bgr in team_bgr.items()
        }
        vor_patch = voronoi_ctrl.compute(
            voronoi_input, SCALE, PITCH_LEFT, PITCH_TOP, vor_bgr
        )
        # Blend only inside pitch bounds
        py1, py2 = PITCH_TOP, PITCH_TOP + PITCH_D_PX
        px1, px2 = PITCH_LEFT, PITCH_LEFT + PITCH_W_PX
        pitch_patch = canvas[py1:py2, px1:px2].astype(np.float32)
        vor_f = vor_patch.astype(np.float32)
        mask = vor_patch.sum(axis=2) > 0
        blended = pitch_patch.copy()
        blended[mask] = (pitch_patch[mask] * (1 - VORONOI_ALPHA)
                         + vor_f[mask] * VORONOI_ALPHA)
        canvas[py1:py2, px1:px2] = blended.astype(np.uint8)

        # ── Formation lines & player dots ────────────────────────────
        formations = {}
        for team in (1, 2):
            plist = players_by_team[team]
            if not plist:
                formations[team] = ("N/A", "N/A")
                continue

            color = team_bgr[team]
            xs = [p['pos'][0] for p in plist]
            f_str, assignments, _ = self.formation_detector.detect(xs)
            phase = self.formation_detector.get_phase(xs)
            formations[team] = (f_str, phase)

            # Group players by line
            by_line = {}
            for p, la in zip(plist, assignments):
                by_line.setdefault(la, []).append(p)

            # Draw formation line connectors
            for line_players in by_line.values():
                sorted_p = sorted(line_players, key=lambda p: p['pos'][1])
                pts = [_pitch_to_canvas(*p['pos']) for p in sorted_p]
                for i in range(len(pts) - 1):
                    cv2.line(canvas, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)

            # Draw velocity arrows and player dots
            for p in plist:
                cx, cy = _pitch_to_canvas(*p['pos'])
                vx, vy = float(p['vel'][0]), float(p['vel'][1])

                # Arrow tip (vel in pitch coords → canvas offsets)
                ex = int(cx + vy * SCALE * ARROW_T)
                ey = int(cy + vx * SCALE * ARROW_T)
                if (ex, ey) != (cx, cy):
                    cv2.arrowedLine(canvas, (cx, cy), (ex, ey),
                                    color, 1, tipLength=0.45, line_type=cv2.LINE_AA)

                # Player dot
                cv2.circle(canvas, (cx, cy), 9, color, -1)
                cv2.circle(canvas, (cx, cy), 9, (230, 230, 230), 1)
                cv2.putText(canvas, str(p['id']),
                            (cx - 5, cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Draw ball
        ball_info = tracks['ball'][frame_num].get(1, {})
        ball_pos = ball_info.get('position_transformed')
        if ball_pos is not None:
            bx, by_ = _pitch_to_canvas(*ball_pos)
            cv2.circle(canvas, (bx, by_), 6, (255, 255, 255), -1)
            cv2.circle(canvas, (bx, by_), 6, (50, 50, 50), 1)

        # ── Header bar ───────────────────────────────────────────────
        cv2.rectangle(canvas, (0, 0), (CANVAS_W, HEADER_H), (14, 14, 14), -1)
        cv2.putText(canvas, "TACTICAL MAP",
                    (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (210, 210, 210), 2)

        for team in (1, 2):
            f_str, phase = formations.get(team, ("N/A", "N/A"))
            color = team_bgr[team]
            x_off = 280 if team == 1 else 660
            cv2.putText(canvas, f"T{team}: {f_str}  [{phase}]",
                        (x_off, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        t_sec = frame_num / self.fps
        m, s = int(t_sec // 60), int(t_sec % 60)
        cv2.putText(canvas, f"{m:02d}:{s:02d}",
                    (CANVAS_W - 78, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (190, 190, 190), 2)

        # ── Footer bar ───────────────────────────────────────────────
        fy = CANVAS_H - FOOTER_H
        cv2.rectangle(canvas, (0, fy), (CANVAS_W, CANVAS_H), (14, 14, 14), -1)
        cv2.line(canvas, (0, fy), (CANVAS_W, fy), (55, 55, 55), 1)

        t1_pct = possession_pct.get(1, 0.0)
        t2_pct = possession_pct.get(2, 0.0)
        t1_gained = sum(1 for t in transitions_so_far if t['to_team'] == 1)
        t2_gained = sum(1 for t in transitions_so_far if t['to_team'] == 2)

        # Possession bar
        bar_x1, bar_y = PITCH_LEFT, fy + 18
        bar_total = PITCH_W_PX
        bar_h = 14
        t1_bar = int(bar_total * t1_pct / 100) if (t1_pct + t2_pct) > 0 else bar_total // 2
        cv2.rectangle(canvas, (bar_x1, bar_y),
                      (bar_x1 + t1_bar, bar_y + bar_h), team_bgr[1], -1)
        cv2.rectangle(canvas, (bar_x1 + t1_bar, bar_y),
                      (bar_x1 + bar_total, bar_y + bar_h), team_bgr[2], -1)

        cv2.putText(canvas, f"T1  {t1_pct:.1f}%  |  Transitions gained: {t1_gained}",
                    (PITCH_LEFT, fy + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, team_bgr[1], 1)
        cv2.putText(canvas, f"T2  {t2_pct:.1f}%  |  Transitions gained: {t2_gained}",
                    (PITCH_LEFT + 420, fy + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, team_bgr[2], 1)

        # Legend dots
        for i, (team, label) in enumerate([(1, "Team 1"), (2, "Team 2")]):
            lx = CANVAS_W - 200 + i * 100
            cv2.circle(canvas, (lx, fy + 40), 7, team_bgr[team], -1)
            cv2.putText(canvas, label, (lx + 12, fy + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, team_bgr[team], 1)

        return canvas

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(self, tracks, team_ball_control, team_colors_rgb,
                 transitions=None):
        """
        Parameters
        ----------
        tracks            : the full tracks dict (from main pipeline)
        team_ball_control : np.array of ints (0/1/2 per frame)
        team_colors_rgb   : dict {1: np.array([R,G,B]), 2: np.array([R,G,B])}
                            from team_assigner.team_colors
        transitions       : list of transition dicts from TransitionDetector

        Returns
        -------
        list of BGR frames (CANVAS_H × CANVAS_W)
        """
        if transitions is None:
            transitions = []

        # team_colors_rgb values are float64 in BGR order (from OpenCV KMeans)
        team_bgr = {
            t: tuple(int(c) for c in col)
            for t, col in team_colors_rgb.items()
        }

        # Pre-compute velocities once
        print("Pre-computing player velocities...")
        vels = self._compute_velocities(tracks)

        # Build Voronoi controller (grids computed once)
        voronoi_ctrl = VoronoiPitchControl(
            PITCH_W_PX, PITCH_D_PX, PITCH_LEFT, PITCH_TOP, time_horizon=0.5
        )

        n = len(tracks['players'])
        frames = []

        for frame_num in tqdm(range(n), desc="Generating tactical map"):
            # Running possession %
            tbc = team_ball_control[:frame_num + 1]
            total = int((tbc == 1).sum() + (tbc == 2).sum())
            pct = {
                1: 100.0 * int((tbc == 1).sum()) / total if total else 0.0,
                2: 100.0 * int((tbc == 2).sum()) / total if total else 0.0,
            }
            trans_so_far = [t for t in transitions if t['frame'] <= frame_num]
            frames.append(
                self._render_frame(frame_num, tracks, vels,
                                   team_bgr, voronoi_ctrl,
                                   pct, trans_so_far)
            )

        return frames
