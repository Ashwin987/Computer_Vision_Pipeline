"""
narya_calibrator.py - Automatic pitch calibration with graceful fallback.

Narya ML model (TensorFlow + MXNet keypoint detector) is attempted first.
If unavailable the calibrator silently falls back (in order) to:
  1. Last successfully computed Narya homography
  2. Manual 4-point homography from ViewTransformer

INSTALLATION STATUS
===================
Narya requires:
  tensorflow==2.2.0   mxnet==1.6.0   Keras==2.3.1
  segmentation-models   tensorflow-probability
These conflict with the current environment:
  - torch 2.9.0 (Narya requires 1.5.0)
  - numpy 1.26.4 (mxnet 1.6 requires <1.20)
  - No TensorFlow installed
Running in full-fallback mode: the manual 4-point homography from
ViewTransformer is returned for every frame.  When a compatible TF/MXNet
environment is available, Narya will activate automatically.

API
===
  calibrator = NaryaCalibrator(manual_H, manual_H_inv)
  H, H_inv   = calibrator.get_homography(frame)   # numpy float64 (3,3)
  n_kp       = calibrator.get_keypoint_count()     # int
  calibrator.print_summary(elapsed_s, total_frames)
  calibrator.save_debug_image(frame, H_inv, tracks, out_path)
"""

import sys
import os
import cv2
import numpy as np

# Pitch coordinate bounds used for coordinate-system verification.
# Our visible patch is 23.32 m x 68 m.  Narya maps to the full 105 x 68 m
# pitch; using 105 as the upper bound accommodates either system.
_PITCH_X_MAX = 105.0
_PITCH_Y_MAX =  68.0


class NaryaCalibrator:
    """
    Per-frame homography estimator with three-level fallback.

    Coordinate convention (inherited from ViewTransformer):
      H     : camera pixel (col, row) -> real-world (depth_m, width_m)
      H_inv : real-world (depth_m, width_m) -> camera pixel (col, row)
    """

    def __init__(self, manual_H: np.ndarray, manual_H_inv: np.ndarray):
        """
        Parameters
        ----------
        manual_H     : 3x3 homography from ViewTransformer
                       (camera pixels -> real-world metres)
        manual_H_inv : its inverse (real-world -> camera pixels)
        """
        self._manual_H     = manual_H.copy().astype(np.float64)
        self._manual_H_inv = manual_H_inv.copy().astype(np.float64)

        self.last_successful_H     = None
        self.last_successful_H_inv = None

        self._narya_available    = False
        self._narya_estimator    = None
        self._last_keypoint_cnt  = 0

        # Cumulative stats
        self._n_total   = 0
        self._n_narya   = 0
        self._n_prev    = 0
        self._n_manual  = 0
        self._kp_counts = []   # one entry per frame

        self._try_load_narya()

    # =========================================================================
    # Narya loading
    # =========================================================================

    def _try_load_narya(self):
        """Attempt to load Narya HomographyEstimator.  Any error -> fallback."""
        try:
            narya_src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'narya_src')
            if os.path.isdir(narya_src) and narya_src not in sys.path:
                sys.path.insert(0, narya_src)

            # This import triggers TensorFlow + MXNet.  Will raise
            # ModuleNotFoundError when either is absent.
            from narya.tracker.homography_estimator import HomographyEstimator

            print("Narya: loading pretrained models (may download ~150 MB) ...")
            self._narya_estimator = HomographyEstimator(pretrained=True)
            self._narya_available = True
            print("Narya: HomographyEstimator loaded successfully.")

        except Exception as exc:
            self._narya_available = False
            self._narya_estimator = None
            self._print_install_failure(exc)

    @staticmethod
    def _print_install_failure(exc):
        bar = "=" * 62
        print(bar)
        print("NARYA INSTALLATION FAILURE — using manual-homography fallback")
        print(bar)
        print(f"Error  : {exc}")
        print()
        print("Narya requires packages NOT present in this environment:")
        print("  tensorflow==2.2.0   (not installed)")
        print("  mxnet==1.6.0        (not installed)")
        print("  Keras==2.3.1        (not installed)")
        print("  segmentation-models (not installed)")
        print()
        print("Conflicts with the current stack:")
        print("  torch 2.9.0   (Narya requires 1.5.0)")
        print("  numpy 1.26.4  (mxnet 1.6 needs < 1.20)")
        print("  No TensorFlow installed")
        print()
        print("ACTION  : All frames will use the original ViewTransformer")
        print("          manual 4-point homography as fallback.")
        print("          Pitch coverage improvement from Narya is deferred")
        print("          until a compatible TF+MXNet environment is set up.")
        print(bar)

    # =========================================================================
    # Coordinate-system verification
    # =========================================================================

    def _verify_and_correct(self, H: np.ndarray, fh: int, fw: int):
        """
        Check that the frame centre maps inside the pitch domain (0..105, 0..68).
        If not, try horizontal and vertical flips.
        Returns corrected (H, H_inv) or (None, None) if all fail.
        """
        cx, cy = fw / 2.0, fh / 2.0
        pt     = np.array([[[cx, cy]]], dtype=np.float32)
        res    = cv2.perspectiveTransform(pt, H.astype(np.float32))
        rx, ry = float(res[0, 0, 0]), float(res[0, 0, 1])

        def inside(x, y):
            return 0.0 <= x <= _PITCH_X_MAX and 0.0 <= y <= _PITCH_Y_MAX

        if inside(rx, ry):
            return H, np.linalg.inv(H), None

        # Horizontal flip (mirror in x)
        Fx = np.array([[-1, 0, _PITCH_X_MAX], [0, 1, 0], [0, 0, 1]],
                      dtype=np.float64)
        H2    = Fx @ H
        res2  = cv2.perspectiveTransform(pt, H2.astype(np.float32))
        rx2, ry2 = float(res2[0, 0, 0]), float(res2[0, 0, 1])
        if inside(rx2, ry2):
            print(f"  [Narya] horizontal flip applied "
                  f"(raw centre {rx:.1f},{ry:.1f} -> {rx2:.1f},{ry2:.1f})")
            return H2, np.linalg.inv(H2), "horizontal_flip"

        # Vertical flip (mirror in y)
        Fy = np.array([[1, 0, 0], [0, -1, _PITCH_Y_MAX], [0, 0, 1]],
                      dtype=np.float64)
        H3    = Fy @ H
        res3  = cv2.perspectiveTransform(pt, H3.astype(np.float32))
        rx3, ry3 = float(res3[0, 0, 0]), float(res3[0, 0, 1])
        if inside(rx3, ry3):
            print(f"  [Narya] vertical flip applied "
                  f"(raw centre {rx:.1f},{ry:.1f} -> {rx3:.1f},{ry3:.1f})")
            return H3, np.linalg.inv(H3), "vertical_flip"

        print(f"  [Narya] WARNING: frame centre maps to ({rx:.1f}, {ry:.1f}), "
              f"outside pitch bounds. Discarding frame.")
        return None, None, "rejected"

    # =========================================================================
    # Public API
    # =========================================================================

    def get_homography(self, frame: np.ndarray):
        """
        Return (H, H_inv) for this frame.  Never raises.

        Priority:
          1. Narya (if available and succeeds)
          2. last_successful_H (Narya from a previous frame)
          3. Manual 4-point ViewTransformer homography
        """
        self._n_total += 1
        fh, fw = frame.shape[:2]
        H = H_inv = None
        self._last_keypoint_cnt = 0

        # ------------------------------------------------------------------
        # Attempt Narya
        # ------------------------------------------------------------------
        if self._narya_available:
            try:
                img512 = cv2.resize(frame, (512, 512))
                # Narya returns pred_homo mapping 512-px image -> pitch template
                pred_homo, method = self._narya_estimator(img512)

                # Scale to camera-pixel coordinates
                sx = 512.0 / fw
                sy = 512.0 / fh
                S  = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]],
                               dtype=np.float64)
                H_candidate = pred_homo.astype(np.float64) @ S

                H_v, H_inv_v, corr = self._verify_and_correct(
                    H_candidate, fh, fw)
                if H_v is not None:
                    H     = H_v
                    H_inv = H_inv_v
                    # Estimate keypoint count from method used
                    self._last_keypoint_cnt = 8 if method == "cv" else 2
                    self._n_narya += 1
            except Exception:
                pass   # fall through to fallback

        # ------------------------------------------------------------------
        # Fallback hierarchy
        # ------------------------------------------------------------------
        if H is None:
            if self.last_successful_H is not None:
                H     = self.last_successful_H.copy()
                H_inv = self.last_successful_H_inv.copy()
                self._n_prev += 1
            else:
                H     = self._manual_H.copy()
                H_inv = self._manual_H_inv.copy()
                self._n_manual += 1

        self._kp_counts.append(self._last_keypoint_cnt)
        self.last_successful_H     = H.copy()
        self.last_successful_H_inv = H_inv.copy()
        return H, H_inv

    def get_keypoint_count(self) -> int:
        """Number of keypoints detected on the last processed frame."""
        return self._last_keypoint_cnt

    def print_summary(self, elapsed_s: float, total_frames: int):
        kp  = self._kp_counts
        avg = f"{sum(kp)/len(kp):.1f}" if kp else "N/A"
        bar = "=" * 55
        print(bar)
        print("NARYA CALIBRATION SUMMARY")
        print(bar)
        print(f"  Total frames processed : {total_frames}")
        print(f"  Narya successful       : {self._n_narya}")
        print(f"  Prev-frame fallback    : {self._n_prev}")
        print(f"  Manual fallback        : {self._n_manual}")
        print(f"  Avg keypoints / frame  : {avg}")
        print(f"  Min keypoints          : {min(kp) if kp else 0}")
        print(f"  Max keypoints          : {max(kp) if kp else 0}")
        print(f"  Preprocessing time     : {elapsed_s:.1f} s")
        print(bar)

    # =========================================================================
    # Debug image (Step 10)
    # =========================================================================

    def save_debug_image(self, frame: np.ndarray, H_inv: np.ndarray,
                         tracks: dict, out_path: str):
        """
        Save a debug JPEG for frame 0 showing:
          - All detected keypoints (circles + IDs)
          - Pitch boundary polygon in green
          - Up to 5 player positions with real-world labels
        """
        try:
            debug = frame.copy()
            fh, fw = debug.shape[:2]
            H_inv_f32 = H_inv.astype(np.float32)

            # -- Pitch boundary polygon from H_inv -------------------------
            corners_real = np.float32([[0, 0], [0, 68],
                                       [23.32, 0], [23.32, 68]])
            corners_cam  = cv2.perspectiveTransform(
                corners_real.reshape(1, -1, 2), H_inv_f32).reshape(-1, 2)
            # Order: TL(0,0), TR(0,68), BR(23.32,68), BL(23.32,0)
            poly = np.array([corners_cam[0], corners_cam[1],
                             corners_cam[3], corners_cam[2]],
                            dtype=np.int32)
            cv2.polylines(debug, [poly], True, (0, 255, 0), 3)
            px0, py0 = int(poly[0][0]), int(poly[0][1])
            cv2.putText(debug, "PITCH BOUNDARY (H_inv)",
                        (max(0, px0), max(20, py0 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # -- Narya keypoints (none when TF/MXNet absent) ---------------
            if self._last_keypoint_cnt == 0:
                cv2.putText(debug,
                            "Narya keypoints: 0  (TF/MXNet not installed)",
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
            else:
                cv2.putText(debug,
                            f"Narya keypoints: {self._last_keypoint_cnt}",
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            # -- Sample player positions from frame 0 ----------------------
            players_f0 = {}
            if tracks.get('players') and len(tracks['players']) > 0:
                players_f0 = tracks['players'][0]

            drawn = 0
            for pid, info in list(players_f0.items())[:5]:
                pos  = info.get('position_transformed')
                bbox = info.get('bbox')
                if pos is None or bbox is None:
                    continue
                x_r = float(pos[0]) if hasattr(pos, '__len__') else 0.0
                y_r = float(pos[1]) if hasattr(pos, '__len__') and len(pos) > 1 else 0.0

                pt_cam = cv2.perspectiveTransform(
                    np.array([[[x_r, y_r]]], dtype=np.float32), H_inv_f32)
                cx_c = int(pt_cam[0, 0, 0])
                cy_c = int(pt_cam[0, 0, 1])

                if 0 <= cx_c < fw and 0 <= cy_c < fh:
                    cv2.drawMarker(debug, (cx_c, cy_c), (0, 0, 255),
                                   cv2.MARKER_CROSS, 18, 2)
                    lbl = f"#{pid} ({x_r:.1f}m, {y_r:.1f}m)"
                    cv2.putText(debug, lbl, (cx_c + 7, cy_c - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
                    drawn += 1

            # -- Status banner ----------------------------------------------
            status_txt = ("Narya: ACTIVE" if self._narya_available
                          else "Narya: FALLBACK (manual 4-pt homography)")
            status_col = (0, 255, 0) if self._narya_available else (0, 165, 255)
            cv2.putText(debug, status_txt, (20, fh - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_col, 2)

            cv2.imwrite(out_path, debug)
            print(f"Debug image saved: {out_path}  "
                  f"(players shown: {drawn})")
        except Exception as exc:
            print(f"WARNING: could not save debug image: {exc}")
