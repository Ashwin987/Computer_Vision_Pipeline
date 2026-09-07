"""
fast_setup.py — Optimized foundation pipeline for long videos.

Completely SEPARATE from main.py. Does not import, modify, or otherwise
touch main.py, or any of render_output1.py/2/3/4/6.py. It reuses the
existing Tracker / SoccerNetCalibrator / CameraMovementEstimator classes
as library code (constructing instances, calling their existing public
and private methods as-is) rather than duplicating their internals, but
composes a NEW, faster detection/calibration scheme around them:

  1. HALF-RESOLUTION INFERENCE: every frame handed to the player/ball/
     referee detector or the pitch-keypoint model is resized to 50% width
     and height first. All resulting box/keypoint coordinates are scaled
     back up by 2x immediately after inference, before anything else
     touches them (tracking, homography solving, storage) — so every
     downstream consumer only ever sees full-resolution coordinates.

  2. FRAME-SKIP DETECTION: the player/ball/referee detector only runs
     every 3rd frame. ByteTrack tracks that reduced-rate stream (it only
     ever sees the frames we feed it). The 2 skipped frames in each gap
     get each track's bbox linearly interpolated between the previous and
     next detected frame — but ONLY for a track present in BOTH endpoints;
     a track that only appears in the next detected frame (new arrival)
     is left absent until that real detection, per the "wait for the next
     real detection" rule.

  3. CALIBRATION: pitch keypoints run every 10th frame (also half-res,
     scaled back), homography for the gaps is filled with
     SoccerNetCalibrator's own existing nearest-sampled-frame scheme
     (its `_interpolate` staticmethod, reused as-is).

  4. Camera movement (optical flow, not a YOLO model) is NOT downscaled —
     only the two model-based steps above are. Its per-frame math is a
     lightweight re-implementation using CameraMovementEstimator's own
     constructor-derived config (features/lk_params/minimum_distance),
     driven by our own streaming loop instead of its list-based one.

STREAMING, NOT LIST-BASED: unlike utils.video_utils.read_video (used by
main.py), which loads the ENTIRE video into a Python list, this file
never holds more than the current frame (plus, for interpolation, only
small per-track bbox dicts — never raw pixel data — between the two
nearest detected/calibrated frames). This matters: a long video like the
42,475-frame Liverpool/PSG clip at 1920x1080 would need ~250GB to hold
as a full in-memory frame list, far beyond this machine's 16.8GB RAM.
Each stage (detection, camera movement, calibration) opens its own fresh
cv2.VideoCapture and reads forward once, sequentially.

Saves tracking / camera movement / calibration to stubs/fastpipe_*.pkl —
prefixed so they can never collide with main.py's stubs/*_121364.pkl or
any other stub naming.

Usage:
    python fast_setup.py
    python fast_setup.py "path\\to\\video.mp4"
"""

import os
import sys
import time
import pickle

import cv2
import numpy as np
import supervision as sv

from trackers import Tracker
from pitch_calibrator import SoccerNetCalibrator
from camera_movement_estimator import CameraMovementEstimator

DEFAULT_VIDEO  = r'C:\UCLA\yolo model 2\Sample_Videos\LiverpoolPSG_short.mp4'
SCALE          = 0.5     # inference downscale factor (both axes)
DETECT_EVERY   = 3       # player/ball/referee detection cadence
CALIB_EVERY    = 10      # pitch-keypoint calibration cadence

PLAYER_MODEL   = 'models/best.pt'
KEYPOINT_MODEL = 'pose/pitch_keypoints_v3/weights/best.pt'

STUB_DIR     = 'stubs'
CACHE_TRACK  = os.path.join(STUB_DIR, 'fastpipe_track_stubs.pkl')
CACHE_CAM    = os.path.join(STUB_DIR, 'fastpipe_camera_movement_stub.pkl')
CACHE_CAL    = os.path.join(STUB_DIR, 'fastpipe_homography_stub.pkl')
CACHE_META   = os.path.join(STUB_DIR, 'fastpipe_meta.pkl')

# Lowered from 500: at 500, a stalled or pathologically-slow frame anywhere
# in a 250-750 frame stretch between print boundaries was invisible until
# the whole stage finished (or hung) — no way to tell "slow under load" from
# "stuck on one frame" without waiting for the next boundary. 25 keeps the
# print volume reasonable while giving much finer live visibility.
PROGRESS_EVERY = 25


def _video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _video_fps(video_path):
    """Real fps from the video file's own metadata (cv2.CAP_PROP_FPS) —
    NOT a hardcoded assumption. A previous hardcoded FPS = 24 here didn't
    match either sample video's real 25fps, which silently skewed every
    downstream frame-index -> seconds conversion (analyze_danger_score.py,
    analyze_formation_window.py) by a factor of 25/24 without any of them
    knowing it was wrong."""
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps


def _resize_half(frame):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * SCALE), int(h * SCALE)), interpolation=cv2.INTER_LINEAR)


def _progress(stage, fn, n_frames, start_time, last_boundary):
    """Print elapsed/ETA whenever `fn` crosses a new PROGRESS_EVERY boundary."""
    boundary = fn // PROGRESS_EVERY
    if boundary == last_boundary:
        return last_boundary
    elapsed = time.time() - start_time
    pace = elapsed / max(fn, 1)
    eta = pace * max(n_frames - fn, 0)
    print(f"  [{stage}] frame {fn}/{n_frames}  elapsed={elapsed:7.1f}s  "
          f"eta={eta:7.1f}s  ({fn/max(elapsed,1e-6):.2f} frame/s)", flush=True)
    return boundary


def _timing_stats(times):
    """mean/median/p95/max/total over a list of per-inference-attempt wall
    times (seconds) — surfaces slow OUTLIER frames that a stage-total print
    hides completely (a single 30s frame buried in a 250-frame, 90s-total
    stage never shows up in "done in Xs")."""
    if not times:
        return {'n': 0, 'mean_sec': 0.0, 'median_sec': 0.0, 'p95_sec': 0.0,
                'max_sec': 0.0, 'total_sec': 0.0}
    arr = np.array(times, dtype=np.float64)
    return {
        'n': int(len(arr)),
        'mean_sec': float(np.mean(arr)),
        'median_sec': float(np.median(arr)),
        'p95_sec': float(np.percentile(arr, 95)),
        'max_sec': float(np.max(arr)),
        'total_sec': float(np.sum(arr)),
    }


def detect_and_track(video_path, n_frames, use_ball_fallback=False):
    """Stream the video ONCE. Half-res detection every DETECT_EVERY-th
    frame + full-res ByteTrack tracking on that subsampled stream, then
    linear bbox interpolation for the skipped frames. Never holds more
    than the current frame in memory.

    use_ball_fallback: opt-in targeted fallback (see ball_fallback.py) —
    only ever invoked on a frame where detection was actually attempted
    (fn in detected_set) AND the primary model missed the ball on it.
    Off by default: 10-25x slower per triggered frame, so it must be
    asked for explicitly rather than silently slowing down every run.

    Returns (tracks, n_detected_frames, n_interp_frames, fallback_summary,
    timing_stats). fallback_summary is None when use_ball_fallback is False.
    timing_stats is the mean/median/p95/max/total (seconds) of the per-
    detection-attempt-frame cost (resize+predict+track, not decode) — see
    _timing_stats.
    """
    tracker_obj = Tracker(PLAYER_MODEL)   # fresh model + fresh ByteTrack instance
    ball_fallback = None
    if use_ball_fallback:
        from ball_fallback import BallFallback
        ball_fallback = BallFallback()
    detected_indices = list(range(0, n_frames, DETECT_EVERY))
    detected_set = set(detected_indices)

    per_frame = {}
    frame_times = []
    cap = cv2.VideoCapture(video_path)
    start = time.time()
    last_boundary = -1
    fn = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fn >= n_frames:
            break
        if fn in detected_set:
            _t0 = time.time()
            small = _resize_half(frame)
            result = tracker_obj.model.predict(small, conf=0.1, verbose=False)[0]
            cls_names = result.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detections = sv.Detections.from_ultralytics(result)
            if len(detections) > 0:
                detections.xyxy = detections.xyxy / SCALE   # scale back to full-res

            # Capture the model's own goalkeeper/player distinction before it's
            # collapsed below, same as trackers/tracker.py — carried through
            # ByteTrack via Detections.data.
            detections.data['is_goalkeeper'] = np.array([
                cls_names[cid] == "goalkeeper" for cid in detections.class_id
            ])

            for object_ind, class_id in enumerate(detections.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detections.class_id[object_ind] = cls_names_inv["player"]

            tracked = tracker_obj.tracker.update_with_detections(detections)

            frame_out = {'players': {}, 'referees': {}, 'ball': {}}
            for fd in tracked:
                bbox, cls_id, track_id = fd[0].tolist(), fd[3], fd[4]
                is_gk = bool(fd[5].get('is_goalkeeper', False))
                if cls_id == cls_names_inv['player']:
                    frame_out['players'][track_id] = {'bbox': bbox, 'is_goalkeeper': is_gk}
                if cls_id == cls_names_inv['referee']:
                    frame_out['referees'][track_id] = {'bbox': bbox}
            for fd in detections:
                bbox, cls_id = fd[0].tolist(), fd[3]
                if cls_id == cls_names_inv['ball']:
                    frame_out['ball'][1] = {'bbox': bbox}

            # Targeted fallback: only on frames detection was actually
            # attempted on AND the primary model just missed the ball.
            # Runs on the full-res `frame`, not the half-res `small` the
            # primary model saw. Does not change gap-handling for frames
            # where the fallback also misses (frame_out['ball'] just stays
            # empty, same as before, and downstream interpolation is
            # untouched).
            if ball_fallback is not None and 1 not in frame_out['ball']:
                recovered_bbox = ball_fallback.try_recover(frame)
                if recovered_bbox is not None:
                    frame_out['ball'][1] = {'bbox': recovered_bbox, 'fallback': True}

            per_frame[fn] = frame_out
            frame_times.append(time.time() - _t0)
            last_boundary = _progress("detect", fn, n_frames, start, last_boundary)
        fn += 1
    cap.release()

    timing_stats = _timing_stats(frame_times)
    print(f"  [detect] done: {len(per_frame)} frames detected "
          f"in {time.time() - start:.1f}s")
    print(f"  [detect] per-frame timing: mean={timing_stats['mean_sec']:.3f}s "
          f"median={timing_stats['median_sec']:.3f}s p95={timing_stats['p95_sec']:.3f}s "
          f"max={timing_stats['max_sec']:.3f}s (n={timing_stats['n']})")

    # ── Fill the gaps: linear bbox interpolation, common-track-id only ────────
    tracks = {
        'players':  [{} for _ in range(n_frames)],
        'referees': [{} for _ in range(n_frames)],
        'ball':     [{} for _ in range(n_frames)],
    }
    for idx in detected_indices:
        if idx not in per_frame:
            continue
        for obj in ('players', 'referees', 'ball'):
            tracks[obj][idx] = per_frame[idx][obj]

    n_interp_frames = 0
    present_indices = [i for i in detected_indices if i in per_frame]
    for i in range(len(present_indices) - 1):
        prev_idx, next_idx = present_indices[i], present_indices[i + 1]
        gap = next_idx - prev_idx
        if gap <= 1:
            continue
        any_interp_this_gap = False
        for obj in ('players', 'referees', 'ball'):
            prev_tracks = per_frame[prev_idx][obj]
            next_tracks = per_frame[next_idx][obj]
            common_ids = set(prev_tracks.keys()) & set(next_tracks.keys())
            for g in range(prev_idx + 1, next_idx):
                frac = (g - prev_idx) / gap
                for tid in common_ids:
                    b0, b1 = prev_tracks[tid]['bbox'], next_tracks[tid]['bbox']
                    bbox = [b0[k] + (b1[k] - b0[k]) * frac for k in range(4)]
                    interp_track = {'bbox': bbox}
                    if 'is_goalkeeper' in prev_tracks[tid]:
                        interp_track['is_goalkeeper'] = prev_tracks[tid]['is_goalkeeper']
                    tracks[obj][g][tid] = interp_track
                    any_interp_this_gap = True
        if any_interp_this_gap:
            n_interp_frames += (gap - 1)

    fallback_summary = ball_fallback.summary() if ball_fallback is not None else None
    return tracks, len(per_frame), n_interp_frames, fallback_summary, timing_stats


def camera_movement(video_path, n_frames):
    """Stream the video ONCE, full resolution (not one of the two YOLO
    models, so not downscaled). Calls CameraMovementEstimator's own
    estimate_frame_movement() per frame pair — the same fixed,
    median-based / status-flag-aware math get_camera_movement() uses —
    instead of keeping a separate duplicate of the algorithm here. Only
    the streaming loop itself (reading one frame at a time, never holding
    the whole video) is specific to this file; get_camera_movement()'s own
    loop is list-based and can't be called directly on a long video for
    the same memory reason detection is streamed above."""
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError(f"Cannot read first frame of {video_path}")

    estimator = CameraMovementEstimator(first_frame)
    movement = [[0.0, 0.0] for _ in range(n_frames)]

    old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    old_features = cv2.goodFeaturesToTrack(old_gray, **estimator.features)

    start = time.time()
    last_boundary = -1
    fn = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fn >= n_frames:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cam_x, cam_y, magnitude, old_features = estimator.estimate_frame_movement(
            old_gray, old_features, frame_gray)

        if magnitude > estimator.minimum_distance:
            movement[fn] = [cam_x, cam_y]

        old_gray = frame_gray
        last_boundary = _progress("camera_movement", fn, n_frames, start, last_boundary)
        fn += 1
    cap.release()

    print(f"  [camera_movement] done in {time.time() - start:.1f}s")
    return movement


def calibrate(video_path, n_frames):
    """Stream the video ONCE. Half-res pitch-keypoint calibration every
    CALIB_EVERY-th frame, keypoints scaled back to full-res before
    homography solving, gaps filled via SoccerNetCalibrator's own
    nearest-sampled-frame scheme."""
    calibrator = SoccerNetCalibrator(KEYPOINT_MODEL)
    calib_indices = list(range(0, n_frames, CALIB_EVERY))
    calib_set = set(calib_indices)

    sampled = {}
    frame_times = []
    cap = cv2.VideoCapture(video_path)
    start = time.time()
    last_boundary = -1
    fn = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fn >= n_frames:
            break
        if fn in calib_set:
            _t0 = time.time()
            full_h, full_w = frame.shape[:2]
            small = _resize_half(frame)
            kp_pixel, kp_world = calibrator._detect_keypoints(small)
            if kp_pixel is None:
                sampled[fn] = None
            else:
                kp_pixel_full = kp_pixel / SCALE   # scale keypoints back to full-res
                sampled[fn] = calibrator._solve_homography(kp_pixel_full, kp_world, (full_h, full_w))
            frame_times.append(time.time() - _t0)
            last_boundary = _progress("calibrate", fn, n_frames, start, last_boundary)
        fn += 1
    cap.release()

    timing_stats = _timing_stats(frame_times)
    print(f"  [calibrate] done: {len(sampled)} frames calibrated "
          f"in {time.time() - start:.1f}s")
    print(f"  [calibrate] per-frame timing: mean={timing_stats['mean_sec']:.3f}s "
          f"median={timing_stats['median_sec']:.3f}s p95={timing_stats['p95_sec']:.3f}s "
          f"max={timing_stats['max_sec']:.3f}s (n={timing_stats['n']})")

    homography_per_frame = SoccerNetCalibrator._interpolate(sampled, n_frames)
    return homography_per_frame, len(sampled), n_frames - len(sampled), timing_stats


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    print(f"fast_setup: video = {video_path}")

    os.makedirs(STUB_DIR, exist_ok=True)

    t0 = time.time()
    n_frames = _video_frame_count(video_path)
    fps = _video_fps(video_path)
    print(f"  {n_frames} frames @ {fps}fps (streaming — never loaded into memory as a full list).")

    print("\n== Stage 1/3: player/ball/referee detection (half-res, every "
          f"{DETECT_EVERY}rd frame) ==")
    tracks, n_detected, n_interp, _fallback_summary, _detect_timing = detect_and_track(video_path, n_frames)
    with open(CACHE_TRACK, 'wb') as f:
        pickle.dump(tracks, f)
    print(f"  Saved {CACHE_TRACK}")

    print("\n== Stage 2/3: camera movement (full-res, unmodified math) ==")
    camera_movement_per_frame = camera_movement(video_path, n_frames)
    with open(CACHE_CAM, 'wb') as f:
        pickle.dump(camera_movement_per_frame, f)
    print(f"  Saved {CACHE_CAM}")

    print(f"\n== Stage 3/3: pitch calibration (half-res, every {CALIB_EVERY}th frame) ==")
    homography_per_frame, n_calibrated, n_calib_interp, _calib_timing = calibrate(video_path, n_frames)
    with open(CACHE_CAL, 'wb') as f:
        pickle.dump(homography_per_frame, f)
    print(f"  Saved {CACHE_CAL}")

    meta = {'video_path': video_path, 'n_frames': n_frames, 'fps': fps,
            'detect_every': DETECT_EVERY}
    with open(CACHE_META, 'wb') as f:
        pickle.dump(meta, f)
    print(f"  Saved {CACHE_META}")

    total_time = time.time() - t0
    print("\n" + "=" * 60)
    print("fast_setup.py SUMMARY")
    print("=" * 60)
    print(f"  Total setup time                 : {total_time:.1f}s  ({total_time/60:.1f} min)")
    print(f"  Frames total                     : {n_frames}")
    print(f"  Frames fully detected             : {n_detected}")
    print(f"  Frames interpolated (detection)   : {n_interp}")
    print(f"  Frames calibrated                 : {n_calibrated}")
    print(f"  Frames interpolated (calibration) : {n_calib_interp}")
    print("=" * 60)


if __name__ == '__main__':
    main()
