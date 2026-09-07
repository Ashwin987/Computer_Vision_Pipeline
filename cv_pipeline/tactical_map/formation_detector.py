import numpy as np


class FormationDetector:
    # Minimum gap in metres between player x-positions to consider a new line
    LINE_GAP_M = 2.5

    def detect(self, positions_x):
        """
        Cluster players into formation lines using natural gaps in depth (x).

        positions_x : list of float  (pitch-x = depth, 0..23.32 m)

        Returns
        -------
        formation_str  : e.g. "4-3-3"
        assignments    : list[int] — line index for each input position
                         (0 = deepest / most defensive line)
        line_centers   : list[float] — average x of each line
        """
        if len(positions_x) < 2:
            return "N/A", [], []

        order = np.argsort(positions_x)
        xs_sorted = np.array(positions_x)[order]

        # Split wherever the gap between consecutive players exceeds LINE_GAP_M
        line_id = 0
        sorted_line = [0]
        for i in range(1, len(xs_sorted)):
            if xs_sorted[i] - xs_sorted[i - 1] > self.LINE_GAP_M:
                line_id += 1
            sorted_line.append(line_id)

        n_lines = line_id + 1
        counts = [sorted_line.count(l) for l in range(n_lines)]
        formation_str = "-".join(str(c) for c in counts)

        # Map back to original order
        assignments = [0] * len(positions_x)
        for sorted_idx, orig_idx in enumerate(order):
            assignments[orig_idx] = sorted_line[sorted_idx]

        # Average x per line
        line_xs = [[] for _ in range(n_lines)]
        for x, la in zip(positions_x, assignments):
            line_xs[la].append(x)
        line_centers = [float(np.mean(lx)) for lx in line_xs]

        return formation_str, assignments, line_centers

    def get_phase(self, positions_x, midpoint=11.66):
        """Return 'ATK' if the team's average depth > midpoint, else 'DEF'."""
        if not positions_x:
            return "N/A"
        return "ATK" if float(np.mean(positions_x)) > midpoint else "DEF"
