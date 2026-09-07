"""
test_fastpipe_position_data.py — proves the full-match tracking/position
data can be loaded and used WITHOUT loading any video frames or
calibration images, and without hitting a memory error.

Only reads:
  stubs/fastpipe_meta.pkl         (video_path, fps, n_frames)
  stubs/fastpipe_track_stubs.pkl  (per-frame player/referee/ball bboxes)

Never opens the video file, never touches fastpipe_homography_stub.pkl
or fastpipe_camera_movement_stub.pkl (calibration/camera data), and
never constructs a Tracker (which would load the YOLO model weights).
"""

import gc
import os
import pickle
import time

import psutil

from utils import get_center_of_bbox, get_foot_position

STUB_DIR    = 'stubs'
CACHE_TRACK = os.path.join(STUB_DIR, 'fastpipe_track_stubs.pkl')
CACHE_META  = os.path.join(STUB_DIR, 'fastpipe_meta.pkl')

process = psutil.Process(os.getpid())


def rss_mb():
    return process.memory_info().rss / (1024 ** 2)


def main():
    gc.collect()
    baseline_mb = rss_mb()
    print(f"Baseline RSS before loading anything: {baseline_mb:.1f} MB")

    # ---- 1. Load only the numeric tracking data (no video, no images) ----
    t0 = time.time()

    with open(CACHE_META, 'rb') as f:
        meta = pickle.load(f)
    video_path, fps, n_frames = meta['video_path'], meta['fps'], meta['n_frames']

    with open(CACHE_TRACK, 'rb') as f:
        tracks = pickle.load(f)

    load_s = time.time() - t0
    after_load_mb = rss_mb()

    print(f"\nLoaded stubs/fastpipe_track_stubs.pkl + fastpipe_meta.pkl in {load_s:.2f}s")
    print(f"  video_path (metadata only, file never opened): {video_path}")
    print(f"  fps   : {fps}")
    print(f"  frames: {n_frames}")
    for obj in ('players', 'referees', 'ball'):
        print(f"  tracks['{obj}']: {len(tracks[obj])} frames")

    assert len(tracks['players']) == n_frames, "player track list doesn't cover the full match"

    # ---- 2. Derive position from bbox (pure math, no video needed) ----
    for object_name, object_tracks in tracks.items():
        for frame_num, frame_tracks in enumerate(object_tracks):
            for track_id, track_info in frame_tracks.items():
                bbox = track_info['bbox']
                if object_name == 'ball':
                    track_info['position'] = get_center_of_bbox(bbox)
                else:
                    track_info['position'] = get_foot_position(bbox)

    after_position_mb = rss_mb()
    print(f"\nRSS after loading + computing positions for all {n_frames} frames: "
          f"{after_position_mb:.1f} MB")
    print(f"  (+{after_position_mb - baseline_mb:.1f} MB over baseline, "
          f"+{after_position_mb - after_load_mb:.1f} MB from position computation)")

    # ---- 3. Trivial sanity computation across the FULL match ----
    # Count how many frames each player track ID appears in.
    t1 = time.time()
    frame_counts = {}
    for frame_tracks in tracks['players']:
        for track_id in frame_tracks:
            frame_counts[track_id] = frame_counts.get(track_id, 0) + 1
    compute_s = time.time() - t1

    print(f"\nSanity computation (frames-per-player-ID) over all {n_frames} frames "
          f"took {compute_s:.2f}s")
    print(f"  Distinct player track IDs seen across the full match: {len(frame_counts)}")
    top5 = sorted(frame_counts.items(), key=lambda kv: -kv[1])[:5]
    print("  Top 5 most-present player IDs (track_id: frame_count):")
    for tid, cnt in top5:
        pct = 100.0 * cnt / n_frames
        print(f"    player {tid}: {cnt} frames ({pct:.1f}% of match)")

    final_mb = rss_mb()
    peak_mb = getattr(process.memory_info(), 'peak_wset', None)
    print(f"\nFinal process RSS: {final_mb:.1f} MB")
    if peak_mb:
        print(f"Peak working set (Windows): {peak_mb / (1024 ** 2):.1f} MB")

    print("\nRESULT: full-match position/tracking data loaded and analyzed "
          "successfully, no video frames or calibration images touched, "
          "no memory error.")


if __name__ == '__main__':
    main()
