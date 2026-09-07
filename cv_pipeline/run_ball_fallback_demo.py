"""
run_ball_fallback_demo.py — ONE-TIME run of the real integrated ball
fallback (trackers/tracker.py's use_ball_fallback=True) over main.py's
short demo clip (Match_videos/121364_0.mp4, 750 frames).

Standalone: does not modify or import main.py. Uses the same Tracker
class and video path main.py uses, but writes to its OWN stub path so
the existing stubs/track_stubs_121364.pkl (used by main.py's normal
runs) is never touched. Forces a fresh detection pass (read_from_stub=
False) so the fallback call site in get_object_tracks() actually runs
on every frame the primary model misses.

Usage:
    python run_ball_fallback_demo.py
"""
import pickle
import time

from utils import read_video
from trackers import Tracker

VIDEO_PATH = 'Match_videos/121364_0.mp4'
NEW_STUB_PATH = 'stubs/track_stubs_121364_ball_fallback.pkl'
BASELINE_STUB_PATH = 'stubs/track_stubs_121364.pkl'
RESULTS_PATH = 'ball_fallback_demo_results.json'


def main():
    print(f"Reading {VIDEO_PATH} ...")
    video_frames = read_video(VIDEO_PATH)
    n_frames = len(video_frames)
    print(f"  {n_frames} frames loaded")

    tracker = Tracker('models/best.pt', use_ball_fallback=True)

    t0 = time.time()
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=False,            # force a fresh run so the fallback call site executes
        stub_path=NEW_STUB_PATH,          # saved here, existing main stub is never touched
        video_path=VIDEO_PATH,
    )
    total_time = time.time() - t0

    ball = tracks['ball']
    primary_hits = sum(1 for b in ball if b.get(1) is not None and not b[1].get('fallback'))
    fallback_hits = sum(1 for b in ball if b.get(1) is not None and b[1].get('fallback'))
    final_hits = primary_hits + fallback_hits
    still_missed = n_frames - final_hits

    fb_summary = tracker.ball_fallback.summary()

    result = {
        'video_path': VIDEO_PATH,
        'n_frames': n_frames,
        'total_wall_sec': total_time,
        'total_wall_min': total_time / 60,
        'primary_hits': primary_hits,
        'fallback_triggered': fb_summary['n_triggered'],
        'fallback_recovered': fb_summary['n_recovered'],
        'fallback_recovery_rate': fb_summary['recovery_rate'],
        'fallback_total_sec': fb_summary['total_fallback_sec'],
        'fallback_avg_sec_per_trigger': fb_summary['avg_sec_per_trigger'],
        'final_hits_combined': final_hits,
        'final_detection_rate': final_hits / n_frames,
        'still_missed_frames_count': still_missed,
        'primary_only_detection_rate': primary_hits / n_frames,
        'new_stub_path': NEW_STUB_PATH,
        'baseline_stub_path_untouched': BASELINE_STUB_PATH,
    }

    print("\n" + "=" * 60)
    print("BALL FALLBACK DEMO RUN — SUMMARY")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k:32s}: {v}")
    print("=" * 60)

    import json
    with open(RESULTS_PATH, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nWrote {RESULTS_PATH}")
    print(f"New ball-tracking cache saved to: {NEW_STUB_PATH}")


if __name__ == '__main__':
    main()
