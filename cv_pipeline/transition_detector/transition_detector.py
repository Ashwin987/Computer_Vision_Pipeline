import cv2
import numpy as np
from tqdm import tqdm


class TransitionDetector:
    # How long the vulnerability window lasts after a team loses the ball
    VULNERABILITY_FRAMES = 60   # 2.5 s at 24 fps
    # Minimum position shift (metres) to consider the new team "counter-pressing"
    ADVANCE_THRESHOLD_M = 2.0

    def __init__(self, fps=24):
        self.fps = fps
        self.transitions = []

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_transitions(self, team_ball_control, tracks):
        transitions = []
        n = len(team_ball_control)

        for i in range(1, n):
            prev = int(team_ball_control[i - 1])
            curr = int(team_ball_control[i])
            if prev in (1, 2) and curr in (1, 2) and prev != curr:
                speed_frames = self._transition_speed(tracks, i, curr)
                transitions.append({
                    'frame': i,
                    'from_team': prev,
                    'to_team': curr,
                    'speed_frames': speed_frames,
                    'speed_seconds': round(speed_frames / self.fps, 2),
                })

        self.transitions = transitions
        return transitions

    def _transition_speed(self, tracks, start_frame, attacking_team, window=48):
        """Return how many frames until the newly possessing team advances 2 m."""
        n = len(tracks['players'])
        x0 = self._team_avg_x(tracks, start_frame, attacking_team)
        if x0 is None:
            return window
        for f in range(start_frame + 1, min(start_frame + window, n)):
            x1 = self._team_avg_x(tracks, f, attacking_team)
            if x1 is not None and abs(x1 - x0) >= self.ADVANCE_THRESHOLD_M:
                return f - start_frame
        return window

    def _team_avg_x(self, tracks, frame_num, team):
        xs = [
            info['position_transformed'][0]
            for info in tracks['players'][frame_num].values()
            if info.get('team') == team
            and info.get('position_transformed') is not None
        ]
        return float(np.mean(xs)) if xs else None

    # ------------------------------------------------------------------
    # Drawing (overlaid on Output 1 frames)
    # ------------------------------------------------------------------

    def draw_transitions(self, frames, transitions):
        """Overlay a dark banner at the top during each vulnerability window."""
        active = {}
        for t in transitions:
            for df in range(self.VULNERABILITY_FRAMES):
                f = t['frame'] + df
                if f < len(frames):
                    active.setdefault(f, []).append(t)

        output = []
        for frame_num, frame in enumerate(tqdm(frames, desc="Drawing transitions")):
            frame = frame.copy()
            evts = active.get(frame_num, [])
            if evts:
                t = evts[-1]
                remaining = (self.VULNERABILITY_FRAMES - (frame_num - t['frame'])) / self.fps
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], 68), (0, 18, 65), -1)
                cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
                cv2.putText(
                    frame,
                    f"TRANSITION  T{t['from_team']} -> T{t['to_team']}"
                    f"   |   Vulnerability window: {remaining:.1f}s"
                    f"   |   Transition speed: {t['speed_seconds']:.1f}s",
                    (18, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 210, 255), 2,
                )
            output.append(frame)
        return output

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_summary(self):
        if not self.transitions:
            print("No transitions detected.")
            return
        print(f"\n{'=' * 54}")
        print(f"  TRANSITIONS  ({len(self.transitions)} total)")
        print(f"{'=' * 54}")
        for t in self.transitions:
            s = t['frame'] / self.fps
            m, sc = int(s // 60), int(s % 60)
            print(f"  [{m:02d}:{sc:02d}]  T{t['from_team']} -> T{t['to_team']}"
                  f"  (transition speed: {t['speed_seconds']:.1f}s)")
        print(f"{'=' * 54}\n")
