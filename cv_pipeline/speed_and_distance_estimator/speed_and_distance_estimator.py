import cv2
import sys
sys.path.append('../')
from collections import defaultdict, deque
from utils import measure_distance, get_foot_position

BUFFER_SIZE   = 3       # rolling window (3 frames ≈ 125 ms at 24 fps)
MAX_SPEED_KMH = 36.0    # fastest credible sprint
MAX_JUMP_M    = 5.0     # max credible single-frame displacement in metres

# Valid in-pitch world coordinate bounds
PITCH_X_MAX = 105.0
PITCH_Y_MAX = 68.0

# Calibration-confidence tiers, applied to the per-frame-pair calibration
# proxy (view_transformer.compute_frame_confidence). Thresholds match the
# ones already validated against real data: frames below LOW_CONF_THRESH
# are exactly where moderate, cap/teleport-filter-evading speed noise was
# confirmed to cluster; HIGH_CONF_THRESH splits out the clearly-solid
# majority (median frame confidence in this pipeline runs ~95%).
HIGH_CONF_THRESH = 0.90
LOW_CONF_THRESH  = 0.70


def _confidence_tier(frac):
    if frac >= HIGH_CONF_THRESH:
        return "high"
    if frac >= LOW_CONF_THRESH:
        return "medium"
    return "low"

# Diagnostic player / frame range
DIAG_PID    = 6
DIAG_FRAMES = range(50, 66)


def _in_bounds(pos):
    """True iff pos is a plausible in-pitch world coordinate."""
    try:
        x, y = float(pos[0]), float(pos[1])
        return 0.0 <= x <= PITCH_X_MAX and 0.0 <= y <= PITCH_Y_MAX
    except (TypeError, IndexError, ValueError):
        return False


class SpeedAndDistance_Estimator():
    def __init__(self, fps):
        self.frame_rate = fps

    def add_speed_and_distance_to_tracks(self, tracks, calibration_confidence_per_frame=None):
        """
        Parameters
        ----------
        calibration_confidence_per_frame : dict, optional
            frame_num -> fraction in [0, 1], from
            ViewTransformer.compute_frame_confidence. When supplied, every
            genuinely-updated ("IN") speed/distance reading is tagged with
            a confidence tier derived from the weaker of the two frames
            the displacement was computed from — a shaky calibration frame
            can produce a plausible-looking (15-36 km/h) but unreliable
            reading that the CAP/TELEPORT filters alone can't tell apart
            from a real sprint. When omitted, every reading is tagged
            "high" (unknown calibration quality is not the same claim as
            good calibration quality, but this keeps behavior unchanged
            for any caller that doesn't pass it).
        """
        fps = self.frame_rate
        print(f"SpeedAndDistance_Estimator: fps={fps}  buffer={BUFFER_SIZE}  "
              f"cap={MAX_SPEED_KMH} km/h  bounds=[0-{PITCH_X_MAX}m x 0-{PITCH_Y_MAX}m]")

        total_distance     = {}
        speed_buffers      = defaultdict(lambda: deque(maxlen=BUFFER_SIZE))
        last_display_speed = {}   # (object, track_id) -> last displayed speed
        last_confidence    = {}   # (object, track_id) -> (tier, frac) of the last genuine update

        # ── Diagnostic header ─────────────────────────────────────────────────
        print(f"\n[speed_diag] PID {DIAG_PID}  frames {DIAG_FRAMES.start}-"
              f"{DIAG_FRAMES.stop - 1}")
        print(f"{'frame':>6}  {'pos_x':>9} {'pos_y':>8}  {'bounds':>9}  "
              f"{'inst_kmh':>10}  {'display':>10}")
        print("-" * 65)

        for object, object_tracks in tracks.items():
            if object in ("ball", "referees"):
                continue
            n = len(object_tracks)
            total_distance.setdefault(object, {})

            for frame_num in range(1, n):
                prev_frame = object_tracks[frame_num - 1]
                curr_frame = object_tracks[frame_num]

                for track_id, info in curr_frame.items():
                    key      = (object, track_id)
                    buf      = speed_buffers[key]
                    last_spd = last_display_speed.get(key, 0.0)
                    total_distance[object].setdefault(track_id, 0.0)

                    display_speed = last_spd   # hold last good by default
                    inst_kmh      = None
                    bounds_str    = "n/a"

                    if track_id in prev_frame:
                        pos_prev = prev_frame[track_id].get('position_transformed')
                        pos_curr = info.get('position_transformed')

                        if pos_prev is not None and pos_curr is not None:
                            prev_ok = _in_bounds(pos_prev)
                            curr_ok = _in_bounds(pos_curr)

                            if prev_ok and curr_ok:
                                dist = measure_distance(pos_prev, pos_curr)

                                if dist > MAX_JUMP_M:
                                    # Teleport: clear stale speed history but
                                    # keep last displayed value (no hard reset)
                                    buf.clear()
                                    bounds_str = "TELEPORT"
                                else:
                                    inst_kmh = dist * fps * 3.6

                                    if inst_kmh > MAX_SPEED_KMH:
                                        # Physically impossible — skip, hold last
                                        bounds_str = "CAP"
                                    else:
                                        buf.append(inst_kmh)
                                        total_distance[object][track_id] += dist
                                        display_speed = sum(buf) / len(buf)
                                        bounds_str = "IN"

                                        if calibration_confidence_per_frame is not None:
                                            pair_frac = min(
                                                calibration_confidence_per_frame.get(frame_num - 1, 1.0),
                                                calibration_confidence_per_frame.get(frame_num, 1.0))
                                        else:
                                            pair_frac = 1.0
                                        last_confidence[key] = (_confidence_tier(pair_frac), pair_frac)
                            else:
                                bounds_str = "OUT"

                    last_display_speed[key] = display_speed
                    conf_tier, conf_frac = last_confidence.get(key, ("high", 1.0))
                    info['speed']               = display_speed
                    info['distance']            = total_distance[object][track_id]
                    info['speed_confidence']      = conf_tier
                    info['speed_confidence_frac'] = conf_frac

                    # ── Diagnostic for PID 6 ──────────────────────────────────
                    if track_id == DIAG_PID and frame_num in DIAG_FRAMES:
                        pos = info.get('position_transformed')
                        if pos is not None:
                            px = f"{float(pos[0]):.2f}"
                            py = f"{float(pos[1]):.2f}"
                        else:
                            px = py = "None"
                        ik = f"{inst_kmh:.2f}" if inst_kmh is not None else "skipped"
                        print(f"{frame_num:>6}  {px:>9} {py:>8}  {bounds_str:>9}  "
                              f"{ik:>10}  {display_speed:>10.2f}")

        print("-" * 65)
        print()

    # Normal reading text color vs a dimmed gray for "low"-confidence
    # readings (built on a poorly-calibrated frame pair — see
    # add_speed_and_distance_to_tracks). Medium tier is left at normal
    # color: dimming both medium and low would gray out ~40% of all
    # on-screen readings, including plenty that are probably fine — low
    # alone (~17% of readings) is where cap/teleport-evading calibration
    # noise was actually confirmed to cluster.
    _NORMAL_COLOR = (0, 0, 0)
    _LOW_CONF_COLOR = (170, 170, 170)

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object in ("ball", "referees"):
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" not in track_info:
                        continue
                    speed    = track_info.get('speed')
                    distance = track_info.get('distance')
                    if speed is None or distance is None:
                        continue
                    color = (self._LOW_CONF_COLOR
                             if track_info.get('speed_confidence') == 'low'
                             else self._NORMAL_COLOR)
                    bbox     = track_info['bbox']
                    position = list(get_foot_position(bbox))
                    position[1] += 40
                    position = tuple(map(int, position))
                    cv2.putText(frame, f"{speed:.1f} km/h", position,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, f"{distance:.0f} m",
                                (position[0], position[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            output_frames.append(frame)
        return output_frames
