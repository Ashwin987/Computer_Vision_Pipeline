import numpy as np


class VoronoiPitchControl:
    """
    Velocity-aware Voronoi pitch control.

    Each player's "influence centre" is projected forward by
    `time_horizon` seconds in the direction they are moving.
    The nearest projected centre wins each pitch pixel.

    The result is a patch image covering only the pitch sub-region
    (pitch_d_px × pitch_w_px), ready to blend onto the canvas.
    """

    def __init__(self, pitch_w_px, pitch_d_px, pitch_left, pitch_top,
                 time_horizon=0.5):
        self.pitch_w_px = pitch_w_px
        self.pitch_d_px = pitch_d_px
        self.time_horizon = time_horizon

        # Pre-compute pixel coordinate grids for the pitch sub-region.
        # xg[r,c] = absolute canvas-x of column c  (maps to pitch-y / width)
        # yg[r,c] = absolute canvas-y of row r      (maps to pitch-x / depth)
        canvas_xs = np.arange(pitch_left, pitch_left + pitch_w_px, dtype=np.float32)
        canvas_ys = np.arange(pitch_top,  pitch_top  + pitch_d_px, dtype=np.float32)
        self.xg, self.yg = np.meshgrid(canvas_xs, canvas_ys)  # (D, W)

    def compute(self, players_with_team, scale, pitch_left, pitch_top,
                team_bgr):
        """
        Parameters
        ----------
        players_with_team : list of (team, (px_m, py_m), (vx_ms, vy_ms))
            team    : 1 or 2
            px_m    : pitch-x in metres (depth)
            py_m    : pitch-y in metres (width)
            vx_ms   : velocity in pitch-x direction (m/s)
            vy_ms   : velocity in pitch-y direction (m/s)
        scale        : px / metre
        pitch_left   : canvas x of pitch left edge
        pitch_top    : canvas y of pitch top edge
        team_bgr     : dict {1: (B,G,R), 2: (B,G,R)}  — team colours

        Returns
        -------
        BGR patch (pitch_d_px, pitch_w_px, 3)
        """
        patch = np.zeros((self.pitch_d_px, self.pitch_w_px, 3), dtype=np.uint8)
        if not players_with_team:
            return patch

        min_d2 = np.full((self.pitch_d_px, self.pitch_w_px), np.inf,
                         dtype=np.float32)
        nearest_team = np.zeros((self.pitch_d_px, self.pitch_w_px),
                                dtype=np.uint8)

        for team, (px_m, py_m), (vx, vy) in players_with_team:
            # Project position forward
            proj_x = px_m + vx * self.time_horizon   # depth (pitch-x)
            proj_y = py_m + vy * self.time_horizon   # width (pitch-y)

            # Convert to canvas coords
            cx = pitch_left + proj_y * scale   # horizontal
            cy = pitch_top  + proj_x * scale   # vertical

            d2 = (self.xg - cx) ** 2 + (self.yg - cy) ** 2
            mask = d2 < min_d2
            min_d2[mask] = d2[mask]
            nearest_team[mask] = team

        for team, bgr in team_bgr.items():
            patch[nearest_team == team] = bgr

        return patch
