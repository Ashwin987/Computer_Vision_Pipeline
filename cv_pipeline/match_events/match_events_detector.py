import numpy as np
import cv2
from tqdm import tqdm


class MatchEventsDetector:
    # Minimum ball speed (km/h) to classify a kicked ball as a shot
    SHOT_SPEED_THRESHOLD = 25
    # Frames of no possession after a kick before we call it a shot
    SHOT_NO_POSSESSION_FRAMES = 15
    # Frames used to sample ball speed immediately after possession is lost
    SPEED_SAMPLE_WINDOW = 5

    def __init__(self, fps=24):
        self.fps = fps
        self.events = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ball_speed_kmh(self, pos1, pos2, n_frames):
        """Return ball speed in km/h between two transformed (meter) positions."""
        if pos1 is None or pos2 is None or n_frames == 0:
            return 0.0
        dist_m = float(np.linalg.norm(np.array(pos1) - np.array(pos2)))
        return (dist_m / (n_frames / self.fps)) * 3.6

    def _build_possession_sequence(self, tracks):
        """Return list of (player_id, team) or None for each frame."""
        possession = []
        for frame_num in range(len(tracks['players'])):
            poss = None
            for player_id, info in tracks['players'][frame_num].items():
                if info.get('has_ball', False):
                    poss = (player_id, info.get('team', 0))
                    break
            possession.append(poss)
        return possession

    # ------------------------------------------------------------------
    # Event detection
    # ------------------------------------------------------------------

    def detect_events(self, tracks):
        """
        Scan possession transitions and ball speed to produce a list of
        PASS, SHOT, and INTERCEPTION events.
        """
        events = []
        n_frames = len(tracks['players'])
        possession = self._build_possession_sequence(tracks)

        i = 0
        while i < n_frames:
            if possession[i] is None:
                i += 1
                continue

            # Find end of current possession run
            j = i + 1
            while j < n_frames and possession[j] == possession[i]:
                j += 1
            # possession[i..j-1] all belong to the same player

            # Find next possessor after the gap
            k = j
            while k < n_frames and possession[k] is None:
                k += 1
            gap_frames = k - j  # frames the ball was in the air / loose

            from_player, from_team = possession[i]

            # Compute ball speed over the first SPEED_SAMPLE_WINDOW frames of the gap
            b1 = tracks['ball'][j].get(1, {}).get('position_transformed') if j < n_frames else None
            sample_end = min(j + self.SPEED_SAMPLE_WINDOW, n_frames - 1)
            b2 = tracks['ball'][sample_end].get(1, {}).get('position_transformed')
            ball_speed = self._ball_speed_kmh(b1, b2, sample_end - j)

            if k < n_frames:
                to_player, to_team = possession[k]

                if from_player != to_player:
                    if from_team == to_team:
                        events.append({
                            'frame': k,
                            'type': 'PASS',
                            'from_player': from_player,
                            'to_player': to_player,
                            'team': from_team,
                            'speed': ball_speed,
                            'text': f"P{from_player} -> P{to_player}  PASS  {ball_speed:.1f} km/h",
                        })
                    else:
                        events.append({
                            'frame': k,
                            'type': 'INTERCEPTION',
                            'from_player': from_player,
                            'to_player': to_player,
                            'from_team': from_team,
                            'to_team': to_team,
                            'speed': ball_speed,
                            'text': f"P{to_player} (T{to_team}) INTERCEPTED P{from_player} (T{from_team})",
                        })

                # Long airball at high speed before reception → also a shot
                if gap_frames >= self.SHOT_NO_POSSESSION_FRAMES and ball_speed >= self.SHOT_SPEED_THRESHOLD:
                    events.append({
                        'frame': j,
                        'type': 'SHOT',
                        'from_player': from_player,
                        'team': from_team,
                        'speed': ball_speed,
                        'text': f"P{from_player} (T{from_team}) SHOT  {ball_speed:.1f} km/h",
                    })
            else:
                # Ball never re-possessed — likely a shot or clearance out of frame
                if ball_speed >= self.SHOT_SPEED_THRESHOLD:
                    events.append({
                        'frame': j,
                        'type': 'SHOT',
                        'from_player': from_player,
                        'team': from_team,
                        'speed': ball_speed,
                        'text': f"P{from_player} (T{from_team}) SHOT  {ball_speed:.1f} km/h",
                    })

            i = j

        events.sort(key=lambda e: e['frame'])
        self.events = events
        return events

    # ------------------------------------------------------------------
    # Video annotation
    # ------------------------------------------------------------------

    def draw_events(self, frames, events, panel_width=420):
        """
        Stitch a separate event-log panel to the right of each tracking frame.
        The tracking video is left completely untouched.
        """
        PANEL_BG      = (20, 20, 20)
        DIVIDER       = (70, 70, 70)
        COLOR_HEADER  = (220, 220, 220)
        COLOR_TIME    = (150, 150, 150)
        COLOR_CLOCK   = (200, 200, 200)
        EVENT_COLORS  = {
            'PASS':          ( 50, 205,  50),   # green
            'SHOT':          (  0,  60, 255),   # red
            'INTERCEPTION':  (  0, 165, 255),   # orange
        }
        TYPE_LABEL = {
            'PASS': 'PASS', 'SHOT': 'SHOT', 'INTERCEPTION': 'INT',
        }

        output = []
        for frame_num, frame in enumerate(tqdm(frames, desc="Building events panel")):
            h, w = frame.shape[:2]
            panel = np.full((h, panel_width, 3), PANEL_BG, dtype=np.uint8)

            # ---- header ----
            cv2.putText(panel, "MATCH  EVENTS",
                        (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_HEADER, 2)
            cv2.line(panel, (10, 50), (panel_width - 10, 50), DIVIDER, 1)

            # ---- running clock ----
            t = frame_num / self.fps
            m, s = int(t // 60), int(t % 60)
            cv2.putText(panel, f"{m:02d}:{s:02d}",
                        (panel_width - 70, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_CLOCK, 2)

            # ---- event log (all events up to now, newest first) ----
            past = [e for e in events if e['frame'] <= frame_num]
            recent = past[-16:][::-1]   # newest at top, max 16 rows

            y = 80
            row_h = (h - 90) // 16     # dynamic row height

            passes      = sum(1 for e in past if e['type'] == 'PASS')
            shots       = sum(1 for e in past if e['type'] == 'SHOT')
            intercepts  = sum(1 for e in past if e['type'] == 'INTERCEPTION')

            for evt in recent:
                color = EVENT_COLORS.get(evt['type'], COLOR_TIME)
                label = TYPE_LABEL.get(evt['type'], evt['type'])

                et = evt['frame'] / self.fps
                em, es = int(et // 60), int(et % 60)

                # timestamp chip
                cv2.putText(panel, f"{em:02d}:{es:02d}",
                            (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TIME, 1)
                # type badge
                cv2.rectangle(panel, (68, y - 14), (108, y + 2), color, -1)
                cv2.putText(panel, label,
                            (72, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
                # detail text — split into two lines if needed
                detail = evt['text']
                cv2.putText(panel, detail,
                            (114, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

                y += row_h
                if y > h - 60:
                    break

            # ---- stats footer ----
            cv2.line(panel, (10, h - 50), (panel_width - 10, h - 50), DIVIDER, 1)
            cv2.putText(panel, f"Passes: {passes}",
                        (12, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, EVENT_COLORS['PASS'], 1)
            cv2.putText(panel, f"Shots: {shots}",
                        (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, EVENT_COLORS['SHOT'], 1)
            cv2.putText(panel, f"Int: {intercepts}",
                        (panel_width - 100, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, EVENT_COLORS['INTERCEPTION'], 1)

            output.append(frame)

        return output

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    def print_summary(self):
        if not self.events:
            print("No match events detected.")
            return

        passes = [e for e in self.events if e['type'] == 'PASS']
        shots = [e for e in self.events if e['type'] == 'SHOT']
        intercepts = [e for e in self.events if e['type'] == 'INTERCEPTION']

        print(f"\n{'=' * 52}")
        print(f"  MATCH EVENTS  ({len(self.events)} total)")
        print(f"{'=' * 52}")
        for evt in self.events:
            t = evt['frame'] / self.fps
            m, s = int(t // 60), int(t % 60)
            print(f"  [{m:02d}:{s:02d}]  {evt['text']}")
        print(f"\n  Passes: {len(passes)}  |  Shots: {len(shots)}  |  Interceptions: {len(intercepts)}")
        print(f"{'=' * 52}\n")
