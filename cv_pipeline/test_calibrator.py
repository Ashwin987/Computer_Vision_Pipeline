"""
test_calibrator.py  —  smoke-test for SoccerNetCalibrator.

Usage:
  python test_calibrator.py                      # uses default video
  python test_calibrator.py path/to/video.mp4
"""

import sys
import time
import cv2
import numpy as np

from pitch_calibrator import SoccerNetCalibrator

VIDEO_PATH  = sys.argv[1] if len(sys.argv) > 1 else "Match_videos/121364_0.mp4"
MODEL_PATH  = sys.argv[2] if len(sys.argv) > 2 else None  # None → use DEFAULT_MODEL
SAMPLE_EVERY = 3      # calibrate every 3rd frame
CONF_THRESHOLD = 0.15 # keypoint confidence cutoff — RANSAC filters bad points


def main():
    print(f"\n{'='*60}")
    print("SoccerNet Pitch Calibrator — Test")
    print(f"Video : {VIDEO_PATH}")
    model_display = MODEL_PATH or SoccerNetCalibrator.DEFAULT_MODEL
    print(f"Model : {model_display}")
    print(f"Sampling every {SAMPLE_EVERY} frames, conf≥{CONF_THRESHOLD}")
    print('='*60)
    sys.stdout.flush()

    kwargs = {"conf_threshold": CONF_THRESHOLD}
    if MODEL_PATH:
        kwargs["model_path"] = MODEL_PATH
    cal = SoccerNetCalibrator(**kwargs)

    t0 = time.time()
    print(f"\nCalibrating video ...")
    homography_per_frame = cal.calibrate_video(VIDEO_PATH, SAMPLE_EVERY)
    elapsed = time.time() - t0

    # ── Statistics ────────────────────────────────────────────────────────
    total   = len(homography_per_frame)
    success = sum(1 for v in homography_per_frame.values() if v is not None)
    fallback = total - success

    print(f"\nResults")
    print(f"  Total frames          : {total}")
    print(f"  Calibrated (H found)  : {success}  ({100*success/max(total,1):.1f}%)")
    print(f"  Fallback (H=None)     : {fallback}  ({100*fallback/max(total,1):.1f}%)")
    print(f"  Wall time             : {elapsed:.1f} s  "
          f"({elapsed/max(total,1)*1000:.1f} ms/frame)")

    # ── Spot-check first successful frame ────────────────────────────────
    first_ok = next(
        ((fn, h) for fn, h in sorted(homography_per_frame.items()) if h is not None),
        None,
    )
    if first_ok is None:
        print("\nWARNING: No frames were successfully calibrated.")
        print("  Possible reasons:")
        print("  - Keypoint confidence too high (try lowering CONF_THRESHOLD)")
        print("  - Model not finding pitch features in this video")
        print("  - PITCH_KP_WORLD mapping may need adjustment")
        return

    fn, (H, H_inv) = first_ok
    print(f"\nSpot-check frame {fn}:")

    # Open video and read that frame
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    cap.release()

    if ret:
        fh, fw = frame.shape[:2]
        cx, cy = fw / 2.0, fh / 2.0
        pt  = np.array([[[cx, cy]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, H)
        rx, ry = float(res[0, 0, 0]), float(res[0, 0, 1])
        print(f"  Frame centre pixel  : ({cx:.0f}, {cy:.0f})")
        print(f"  → world coords      : ({rx:.1f} m, {ry:.1f} m)")
        print(f"  (pitch is 0–105 m × 0–68 m; centre ≈ 52.5, 34)")

        # Save a debug image with the projected pitch corners
        debug = frame.copy()
        corners_world = np.array([
            [[0, 0]], [[105, 0]], [[105, 68]], [[0, 68]]
        ], dtype=np.float32)
        corners_px = cv2.perspectiveTransform(corners_world, H_inv)
        pts = corners_px.reshape(-1, 2).astype(np.int32)
        cv2.polylines(debug, [pts], True, (0, 255, 0), 3)
        cv2.putText(debug, "Projected pitch boundary",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        out_path = "calibrator_debug.jpg"
        cv2.imwrite(out_path, debug)
        print(f"\nDebug image saved: {out_path}")

    print(f"\nNote: ViewTransformer integration")
    print(f"  Pass homography_per_frame to:")
    print(f"  view_transformer.add_transformed_position_to_tracks(tracks, homography_per_frame)")
    print(f"  Frames where H=None will fall back to the fixed 4-point homography.")
    print('='*60)


if __name__ == '__main__':
    main()
