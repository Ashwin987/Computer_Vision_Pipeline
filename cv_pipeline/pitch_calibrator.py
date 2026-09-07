"""
pitch_calibrator.py  —  SoccerNet-based pitch homography calibrator.

Uses a YOLOv8-pose model trained on the Roboflow football-pitch-keypoints
dataset (48 pitch keypoints) to detect landmark positions, then solves for
the homography H : pixel (col, row) -> real-world (x_m, y_m).

Real-world coordinate system (standard FIFA 105 x 68 m pitch):
  x : 0 m (left goal line) → 105 m (right goal line)
  y : 0 m (far touchline)  →  68 m (near touchline)

API
---
  cal = SoccerNetCalibrator(model_path)
  result = cal.calibrate_frame(bgr_frame)   # (H, H_inv) or None
  hpf    = cal.calibrate_video(video_path)  # {frame_num: (H, H_inv)}
"""

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Pitch keypoint world coordinates  (105 x 68 m pitch)
#
# Index → (x_m, y_m) derived from visual analysis of the
# football-pitch-keypoints-1pfv8 (v15) Roboflow dataset and standard FIFA
# pitch dimensions.  Uncertain assignments are marked with a comment;
# verify against the Roboflow project page if higher accuracy is needed.
#
# Pitch measurements used:
#   penalty area depth : 16.5 m   penalty area width : 40.32 m (y 13.84–54.16)
#   goal area depth    :  5.5 m   goal area width    : 18.32 m (y 24.84–43.16)
#   penalty spot       : 11.0 m   centre circle r    :  9.15 m
#   goal width         :  7.32 m  posts at y 30.34 and 37.66
# ---------------------------------------------------------------------------
PITCH_KP_WORLD = {
     0: ( 26.18,  0.56),
     1: (  0.00,  0.00),
     2: ( 35.42,  6.87),
     3: (  0.00, 68.00),
     4: (  5.50, 68.00),
     5: ( 12.32, 50.75),
     6: ( 26.74, 15.70),
     7: ( 30.94, 27.34),
     8: ( 16.50,  0.00),
     9: ( 16.50, 13.84),
    10: ( 16.50, 54.16),
    11: ( 16.50, 68.00),
    12: (  0.00, 24.84),
    13: ( 53.90,  0.42),
    14: ( 54.74, 23.98),
    15: ( 52.50, 43.15),
    16: ( 54.88, 39.82),
    17: ( 37.94, 28.32),
    18: ( 28.42, 66.88),
    19: ( 32.48, 41.64),
    20: ( 48.16, 30.99),
    21: ( 25.76, 22.85),
    22: ( 33.04, 10.09),
    23: ( 22.68,  7.29),
    24: ( 31.92,  0.56),
    25: ( 88.50, 54.16),
    26: ( 88.50, 68.00),
    27: ( 46.00,  0.00),
    28: ( 42.28, 23.13),
    29: ( 47.00, 43.15),
    30: ( 35.84, 36.87),
    31: ( 88.50, 34.00),
    32: ( 16.50,  0.00),
    33: ( 58.00,  0.00),
    34: ( 40.60,  1.26),
    35: ( 88.50, 13.84),
    36: ( 30.38, 53.28),
    37: ( 33.60, 64.07),
    38: ( 36.40, 61.97),
    39: ( 31.36, 53.84),
    40: ( 64.00,  0.00),
    41: ( 88.50,  0.00),
    42: ( 38.08, 57.91),
    43: ( 35.84, 59.73),
    44: (  0.00, 13.84),
    45: (  0.00, 54.16),
    46: (105.00, 13.84),
    47: (105.00, 54.16),
}

# Pitch bounds used for sanity-checking the computed homography
_PITCH_X_MAX = 105.0
_PITCH_Y_MAX =  68.0


class SoccerNetCalibrator:
    """
    Per-frame homography estimator using the YOLOv8-pose pitch keypoint model.

    Parameters
    ----------
    model_path : str
        Path to the trained YOLOv8-pose weights (.pt file).
        Default: 'pose/pitch_keypoints_model2/weights/best.pt'
    conf_threshold : float
        Minimum keypoint confidence to include in the homography solve.
    min_keypoints : int
        Minimum visible keypoints required; returns None if fewer detected.
    ransac_reproj_threshold : float
        RANSAC reprojection threshold in pixels.
    imgsz : int
        Inference image size (longer side).
    """

    DEFAULT_MODEL = 'pose/pitch_keypoints_v3/weights/best.pt'

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        conf_threshold: float = 0.15,
        min_keypoints: int = 4,
        ransac_reproj_threshold: float = 8.0,
        imgsz: int = 1280,
    ):
        self.conf_threshold = conf_threshold
        self.min_keypoints = min_keypoints
        self.ransac_reproj = ransac_reproj_threshold
        self.imgsz = imgsz

        print(f"SoccerNetCalibrator: loading model from {model_path} ...")
        self._model = YOLO(model_path)
        print("SoccerNetCalibrator: model loaded.")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def calibrate_frame(self, frame: np.ndarray):
        """
        Detect pitch keypoints and compute the homography for one frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3).

        Returns
        -------
        (H, H_inv) : tuple of (3,3) float32 ndarrays, or None
            H     maps pixel (x, y) → world (x_m, y_m).
            H_inv maps world (x_m, y_m) → pixel (x, y).
            Returns None if fewer than `min_keypoints` are visible or the
            resulting homography fails the sanity check.
        """
        kp_pixel, kp_world = self._detect_keypoints(frame)
        if kp_pixel is None or len(kp_pixel) < self.min_keypoints:
            return None
        return self._solve_homography(kp_pixel, kp_world, frame.shape)

    def calibrate_video(
        self,
        video_path: str,
        sample_every_n_frames: int = 3,
        progress_callback=None,
    ) -> dict:
        """
        Calibrate a full video by sampling frames and interpolating.

        Parameters
        ----------
        video_path : str
        sample_every_n_frames : int
            Calibrate every Nth frame; fill gaps from nearest calibrated frame.
        progress_callback : callable, optional
            Called as progress_callback(sampled_done, sampled_total) right
            after every SAMPLED frame (i.e. every sample_every_n_frames-th
            raw frame) is actually run through the pose model. Sampled
            counts, not raw frame counts, are what's reported -- this
            stage's entire real cost sits in these calls (historically
              ~2-5s each), while the raw decode-loop frames in between are
            near-instant, so a caller polling for "is this genuinely
            progressing" needs the sampled count, not the raw one.

        Returns
        -------
        dict : {frame_num: (H, H_inv)}  for every frame in the video.
            Frames where calibration fails map to None (caller should fall
            back to the fixed ViewTransformer homography).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampled_total = len(range(0, total, sample_every_n_frames))
        sampled: dict[int, object] = {}   # frame_num → (H, H_inv) or None

        frame_num = 0
        sampled_done = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % sample_every_n_frames == 0:
                sampled[frame_num] = self.calibrate_frame(frame)
                sampled_done += 1
                print(f"  frame {frame_num}/{total}  H={'ok' if sampled[frame_num] else 'None'}",
                      flush=True)
                if progress_callback is not None:
                    progress_callback(sampled_done, sampled_total)
            frame_num += 1
        cap.release()

        total_frames = frame_num
        return self._interpolate(sampled, total_frames)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _detect_keypoints(self, frame: np.ndarray):
        """
        Run YOLOv8-pose inference and return matched pixel / world arrays.

        Returns (kp_pixel, kp_world) each shape (N, 2) float32, or (None, None).
        """
        results = self._model.predict(
            source=frame,
            conf=0.1,          # low conf here; we filter by keypoint conf below
            imgsz=self.imgsz,
            save=False,
            verbose=False,
        )
        if not results or results[0].keypoints is None:
            return None, None

        kp_data = results[0].keypoints
        if len(kp_data.xy) == 0:
            return None, None

        xy   = kp_data.xy[0].cpu().numpy()    # (48, 2) pixel coords
        conf = kp_data.conf[0].cpu().numpy()  # (48,)   confidence

        pix_pts   = []
        world_pts = []
        seen_world = set()

        for i, (pt, c) in enumerate(zip(xy, conf)):
            if c < self.conf_threshold:
                continue
            if i >= len(PITCH_KP_WORLD):
                continue
            wx, wy = PITCH_KP_WORLD[i]
            if (wx, wy) in seen_world:
                continue
            seen_world.add((wx, wy))
            pix_pts.append([pt[0], pt[1]])
            world_pts.append([wx, wy])

        if len(pix_pts) < self.min_keypoints:
            return None, None

        return (np.array(pix_pts,   dtype=np.float32),
                np.array(world_pts, dtype=np.float32))

    def _solve_homography(
        self,
        kp_pixel: np.ndarray,
        kp_world: np.ndarray,
        frame_shape: tuple,
    ):
        """
        Compute H (pixel→world) via RANSAC and validate it.
        Returns (H, H_inv) or None.
        """
        H, mask = cv2.findHomography(
            kp_pixel, kp_world,
            cv2.RANSAC, self.ransac_reproj,
        )
        if H is None:
            return None

        inliers = int(mask.sum()) if mask is not None else 0
        if inliers < self.min_keypoints:
            return None

        if not self._validate(H, frame_shape):
            return None

        H     = H.astype(np.float32)
        H_inv = np.linalg.inv(H).astype(np.float32)
        return H, H_inv

    def _validate(self, H: np.ndarray, frame_shape: tuple) -> bool:
        """
        Check that the frame centre maps inside the pitch domain.
        """
        fh, fw = frame_shape[:2]
        cx, cy = fw / 2.0, fh / 2.0
        pt  = np.array([[[cx, cy]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, H.astype(np.float32))
        rx, ry = float(res[0, 0, 0]), float(res[0, 0, 1])
        return (-10 <= rx <= _PITCH_X_MAX + 10) and (-10 <= ry <= _PITCH_Y_MAX + 10)

    @staticmethod
    def _interpolate(sampled: dict, total_frames: int) -> dict:
        """
        Fill every frame index with the homography from the nearest
        sampled frame.  Frames that couldn't be calibrated at all
        propagate None.
        """
        if not sampled:
            return {i: None for i in range(total_frames)}

        sorted_keys = sorted(sampled.keys())
        result: dict = {}

        for fn in range(total_frames):
            # Find nearest sampled frame
            nearest = min(sorted_keys, key=lambda k: abs(k - fn))
            result[fn] = sampled[nearest]

        return result
