"""
test_roboflow_ball.py — standalone evaluation of Roboflow's pre-trained
"football-ball-detection-rejhg/4" model against our own known ball-
detection problem cases, to decide whether it's worth integrating.

STANDALONE TEST ONLY. Does not import, modify, or touch main.py,
fast_setup.py, trackers/tracker.py, or any render_output*.py file.
Reads video frames directly with cv2 and reads our existing cached
track stubs (read-only) purely to know which frames our own model
already missed, for an apples-to-apples comparison.

Approach (per the Roboflow soccer-ball-tracking tutorial):
  1. InferenceSlicer: split each full frame into 2x2 tiles (each tile
     independently resized/inferred by the model), merge per-tile boxes
     with NMS. Small/fast objects like a ball are relatively larger
     (in pixels) within a quarter-frame tile than in the full downscaled
     frame, which is the whole point of testing this.
  2. BallTracker: keeps only the single detection closest to the recent
     rolling-average ball position, to reject spurious ball-shaped false
     positives (shin pads, advertising boards, etc.) elsewhere on the
     pitch when the slicer returns more than one candidate.

Requires: pip install inference supervision  (already installed here)
Requires: ROBOFLOW_API_KEY in a local .env file (gitignored) or the
environment. Get a free key at https://roboflow.com if none is set.

Usage:
    python test_roboflow_ball.py --phase gap47
    python test_roboflow_ball.py --phase fastpipe --stride 30
    python test_roboflow_ball.py --phase main --stride 3
    python test_roboflow_ball.py --phase all
"""
import argparse
import json
import os
import time
from collections import deque

import cv2
import numpy as np
import supervision as sv
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get('ROBOFLOW_API_KEY')
MODEL_ID = 'football-ball-detection-rejhg/4'

FAST_VIDEO = 'Sample_Videos/LiverpoolPSG_short.mp4'
MAIN_VIDEO = 'Match_videos/121364_0.mp4'
FAST_STUB  = 'stubs/fastpipe_track_stubs.pkl'
MAIN_STUB  = 'stubs/track_stubs_121364.pkl'

CONF = 0.1   # match our own pipeline's accept threshold, for a fair comparison

RESULTS_PATH = 'test_roboflow_ball_results.json'


def _require_api_key():
    if not API_KEY:
        raise SystemExit(
            "No ROBOFLOW_API_KEY found (checked .env and the environment).\n"
            "Get a free key at https://roboflow.com (Settings -> API Keys) "
            "and put it in a local .env file as:\n"
            "    ROBOFLOW_API_KEY=your_key_here"
        )


def load_model():
    _require_api_key()
    from inference import get_model
    return get_model(model_id=MODEL_ID, api_key=API_KEY)


def build_slicer(model, frame_shape, conf=CONF):
    """2x2 tiling: each tile is exactly half the frame's width/height,
    no overlap needed since the ball is never larger than a quarter-frame
    tile and NMS just needs to merge any box that happens to straddle a
    tile boundary."""
    h, w = frame_shape[:2]

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = model.infer(image_slice, confidence=conf)[0]
        return sv.Detections.from_inference(result)

    return sv.InferenceSlicer(
        callback=callback,
        slice_wh=(w // 2, h // 2),
        overlap_wh=(64, 64),
        overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
        iou_threshold=0.5,
        thread_workers=1,
    )


class BallTracker:
    """Roboflow soccer-ball-tutorial anomaly filter: keeps only the
    detection closest to the recent rolling-average ball position."""

    def __init__(self, buffer_size: int = 10):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        if len(detections) == 0:
            return detections
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)
        centroid = np.mean(np.concatenate(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = int(np.argmin(distances))
        return detections[[index]]


def stream_frames(video_path, wanted_indices):
    """Sequential single pass over the video (never cv2.set-seeks, which
    are imprecise on many codecs) — yields (frame_index, frame) only for
    indices in `wanted_indices`. `wanted_indices` must be sorted."""
    wanted = sorted(set(wanted_indices))
    if not wanted:
        return
    cap = cv2.VideoCapture(video_path)
    target_iter = iter(wanted)
    next_target = next(target_iter)
    fn = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fn == next_target:
            yield fn, frame
            try:
                next_target = next(target_iter)
            except StopIteration:
                break
        fn += 1
    cap.release()


def raw_detection_mask_from_stub(stub_path, indices):
    """Our own model's real (non-interpolated) ball detections at these
    exact frame indices, straight from the cached stub — read-only, no
    re-inference. Returns {frame_idx: bool}."""
    import pickle
    with open(stub_path, 'rb') as f:
        tracks = pickle.load(f)
    ball = tracks['ball']
    return {i: (i < len(ball) and ball[i].get(1) is not None) for i in indices}


def run_test(video_path, indices, baseline_stub=None, label=''):
    model = load_model()
    slicer = None
    tracker = BallTracker(buffer_size=10)

    baseline = raw_detection_mask_from_stub(baseline_stub, indices) if baseline_stub else {}

    per_frame = []
    t_slicer_total = 0.0
    n_done = 0
    t0 = time.time()

    for fn, frame in stream_frames(video_path, indices):
        if slicer is None:
            slicer = build_slicer(model, frame.shape)

        ts = time.time()
        dets = slicer(frame)
        t_slicer_total += time.time() - ts

        tracked = tracker.update(dets)

        per_frame.append({
            'frame': fn,
            'n_raw_detections': int(len(dets)),
            'raw_max_conf': float(dets.confidence.max()) if len(dets) else None,
            'tracked_hit': bool(len(tracked) > 0),
            'baseline_hit': baseline.get(fn),
        })
        n_done += 1
        if n_done % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] {n_done}/{len(indices)} frames  "
                  f"elapsed={elapsed:.0f}s  avg={elapsed/n_done:.2f}s/frame", flush=True)

    n = len(per_frame)
    n_new_hits = sum(1 for r in per_frame if r['tracked_hit'])
    n_baseline_hits = sum(1 for r in per_frame if r['baseline_hit'])
    n_recovered = sum(1 for r in per_frame
                       if r['tracked_hit'] and baseline_stub and not r['baseline_hit'])

    summary = {
        'label': label,
        'video_path': video_path,
        'n_frames_tested': n,
        'new_model_hits': n_new_hits,
        'new_model_hit_rate': n_new_hits / n if n else None,
        'baseline_hits': n_baseline_hits if baseline_stub else None,
        'baseline_hit_rate': n_baseline_hits / n if (baseline_stub and n) else None,
        'frames_recovered_by_new_model': n_recovered if baseline_stub else None,
        'avg_sec_per_frame_slicer_only': t_slicer_total / n if n else None,
        'total_wall_sec': time.time() - t0,
    }
    return summary, per_frame


def bench_original_model(video_path, n_frames=10, start_frame=0):
    """Time our OWN current model (models/best.pt), single full-res
    frame, no slicing — for a clean speed comparison inside this same
    script/session (not relying on a previous session's numbers)."""
    from ultralytics import YOLO
    model = YOLO('models/best.pt')
    indices = list(range(start_frame, start_frame + n_frames))
    frames = [f for _, f in stream_frames(video_path, indices)]
    if not frames:
        return None
    model.predict(frames[0], conf=0.1, verbose=False)  # warmup
    t0 = time.time()
    for f in frames:
        model.predict(f, conf=0.1, verbose=False)
    return (time.time() - t0) / len(frames)


def phase_gap47():
    print("\n=== PHASE: 47-frame shot gap, frames 39301-39347 (LiverpoolPSG_short.mp4) ===")
    indices = list(range(39301, 39348))
    summary, per_frame = run_test(FAST_VIDEO, indices, baseline_stub=FAST_STUB, label='gap47')
    print(json.dumps(summary, indent=2))
    for r in per_frame:
        print(f"    frame {r['frame']}: raw_n={r['n_raw_detections']} "
              f"max_conf={r['raw_max_conf']}  tracked_hit={r['tracked_hit']}  "
              f"our_model_baseline_hit={r['baseline_hit']}")
    return summary, per_frame


def phase_fastpipe(stride):
    print(f"\n=== PHASE: fastpipe 9000-frame window [33475:42475), stride={stride} ===")
    indices = list(range(33475, 42475, stride))
    summary, per_frame = run_test(FAST_VIDEO, indices, baseline_stub=FAST_STUB, label='fastpipe_sample')
    print(json.dumps(summary, indent=2))
    return summary, per_frame


def phase_main(stride):
    print(f"\n=== PHASE: main.py clip [0:750), stride={stride} ===")
    indices = list(range(0, 750, stride))
    summary, per_frame = run_test(MAIN_VIDEO, indices, baseline_stub=MAIN_STUB, label='main_sample')
    print(json.dumps(summary, indent=2))
    return summary, per_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['gap47', 'fastpipe', 'main', 'speed', 'all'], default='all')
    p.add_argument('--stride', type=int, default=None,
                    help="Frame sampling stride for fastpipe/main phases (default: 30 for fastpipe, 3 for main)")
    args = p.parse_args()

    all_results = {}

    if args.phase in ('gap47', 'all'):
        s, pf = phase_gap47()
        all_results['gap47'] = {'summary': s, 'per_frame': pf}

    if args.phase in ('fastpipe', 'all'):
        stride = args.stride or 30
        s, pf = phase_fastpipe(stride)
        all_results['fastpipe'] = {'summary': s, 'per_frame': pf}

    if args.phase in ('main', 'all'):
        stride = args.stride or 3
        s, pf = phase_main(stride)
        all_results['main'] = {'summary': s, 'per_frame': pf}

    if args.phase in ('speed', 'all'):
        print("\n=== PHASE: speed comparison (our current model vs new tiled model) ===")
        orig_speed = bench_original_model(FAST_VIDEO, n_frames=8, start_frame=39301)
        # reuse the gap47 slicer timing already measured if available
        slicer_speed = None
        if 'gap47' in all_results:
            slicer_speed = all_results['gap47']['summary']['avg_sec_per_frame_slicer_only']
        speed_summary = {
            'our_model_sec_per_frame_single_full_res': orig_speed,
            'new_model_sec_per_frame_2x2_tiled': slicer_speed,
            'slowdown_factor': (slicer_speed / orig_speed) if (orig_speed and slicer_speed) else None,
        }
        print(json.dumps(speed_summary, indent=2))
        all_results['speed'] = speed_summary

    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote full results to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
