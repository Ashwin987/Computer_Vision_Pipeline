import numpy as np
import cv2

# Plausible in-pitch world coordinate bounds (same as
# speed_and_distance_estimator's PITCH_X_MAX/PITCH_Y_MAX) — kept here too
# since this is where calibration confidence is assessed, independent of
# whether the speed estimator is even used downstream.
PITCH_X_MAX = 105.0
PITCH_Y_MAX = 68.0

class ViewTransformer():
    def __init__(self):
        court_width = 68
        court_length = 23.32

        self.pixel_vertices = np.array([[0, 1080],
                               [0, 0],
                               [1920, 0],
                               [1920, 1080]])

        self.target_vertices = np.array([
            [0,court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = (int(point[0]),int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices,p,False) >= 0
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        tranform_point = cv2.perspectiveTransform(reshaped_point,self.persepctive_trasnformer)
        return tranform_point.reshape(-1,2)

    def add_transformed_position_to_tracks(self, tracks, homography_per_frame=None):
        """
        Compute position_transformed for every tracked object.

        Parameters
        ----------
        tracks : dict  — the full tracking dictionary
        homography_per_frame : dict (optional)
            Mapping frame_num -> (H, H_inv) where H is a 3x3 float64 ndarray
            (camera pixels -> real-world metres) produced by NaryaCalibrator.
            When supplied, each frame uses its own calibrated H instead of the
            single fixed persepctive_trasnformer.  When None or when a frame is
            missing from the dict, the fixed matrix is used as fallback.
        """
        for object_, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                # Select H for this frame
                if (homography_per_frame is not None
                        and frame_num in homography_per_frame):
                    H = homography_per_frame[frame_num][0].astype(np.float32)
                else:
                    H = self.persepctive_trasnformer

                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)

                    # is_inside check uses the fixed pixel_vertices polygon
                    # (the visible-pitch boundary is stable across frames for
                    # a static camera shot)
                    p = (int(position[0]), int(position[1]))
                    is_inside = cv2.pointPolygonTest(
                        self.pixel_vertices, p, False) >= 0
                    if not is_inside:
                        tracks[object_][frame_num][track_id][
                            'position_transformed'] = None
                        continue

                    reshaped = position.reshape(-1, 1, 2).astype(np.float32)
                    transformed = cv2.perspectiveTransform(reshaped, H)
                    tracks[object_][frame_num][track_id][
                        'position_transformed'] = transformed.reshape(-1, 2).squeeze().tolist()

    def compute_frame_confidence(self, tracks):
        """Per-frame calibration-confidence proxy: the fraction of tracked
        players whose position_transformed lands inside plausible pitch
        bounds (0-105m x 0-68m). Must be called after
        add_transformed_position_to_tracks.

        A poorly-calibrated per-frame homography can still pass the
        pixel-space is-inside check above (so position_transformed isn't
        None) while landing a player hundreds of metres off a real pitch —
        this doesn't diagnose WHY calibration is bad for that frame, only
        how much its output can be trusted. Confirmed against real data:
        frames with a low fraction here are exactly the frames where
        moderate (15-36 km/h), cap/teleport-filter-evading speed noise was
        found to cluster.

        Returns
        -------
        dict  frame_num -> fraction in [0, 1] (1.0 for frames with no
        tracked players, since there's nothing to be unreliable about).
        """
        player_tracks = tracks.get('players', [])
        confidence = {}
        for frame_num, frame in enumerate(player_tracks):
            total = len(frame)
            if total == 0:
                confidence[frame_num] = 1.0
                continue
            good = 0
            for info in frame.values():
                pos = info.get('position_transformed')
                if pos is None:
                    continue
                try:
                    x, y = float(pos[0]), float(pos[1])
                except (TypeError, IndexError, ValueError):
                    continue
                if 0.0 <= x <= PITCH_X_MAX and 0.0 <= y <= PITCH_Y_MAX:
                    good += 1
            confidence[frame_num] = good / total
        return confidence
