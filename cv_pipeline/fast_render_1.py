"""
fast_render_1.py — fast-pipeline counterpart to render_output1.py.

Loads from the fastpipe_ cache built by fast_setup.py (does not re-run
detection/calibration), for a bounded [--start, --end) frame window (the
render_output*.py functions need a full in-memory frame list, so a window
small enough to fit in RAM must be chosen — see fast_common.py). Computes
the tactical-events precompute (same shared tactical_events package
main.py uses) and calls the real, unmodified render_output1() function,
then renames the result to output_videos/longer_output1.avi
(render_output1.py always saves to the fixed output_videos/output1.avi
internally, which would otherwise collide with main.py's own output of
the same run).

Usage:
    python fast_render_1.py
    python fast_render_1.py --start 1000 --end 4000
"""
import argparse
import sys
import time

from fast_common import load_fast_pipeline, NO_CACHE_MESSAGE, DEFAULT_WINDOW_FRAMES, rename_longer_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=DEFAULT_WINDOW_FRAMES)
    args = parser.parse_args()

    data = load_fast_pipeline(start_frame=args.start, end_frame=args.end)
    if data is None:
        print(NO_CACHE_MESSAGE)
        sys.exit(1)

    from render_output3 import STEP, _FALLBACK_PITCH_VERTS
    from tactical_events.space_control import (
        build_pitch_mask, build_sampling_grid, compute_space_control_per_frame,
        compute_pitch_verts_from_tracks,
    )
    from tactical_events.tactical_events_detector import TacticalEventsDetector
    from tactical_events.event_ranking import rank_events_by_window
    from render_output1 import render_output1

    tracks = data['tracks']
    fps = data['fps']

    h, w = data['video_frames'][0].shape[:2]
    # Derived from this clip's own observed player positions, not a fixed
    # constant tuned to a different camera framing — see
    # compute_pitch_verts_from_tracks and render_output3.py's identical use.
    pitch_verts = compute_pitch_verts_from_tracks(tracks, h, w, fallback_verts=_FALLBACK_PITCH_VERTS)
    pitch_mask = build_pitch_mask(pitch_verts, h, w)
    grid, *_ = build_sampling_grid(pitch_mask, STEP)
    space_control_per_frame = compute_space_control_per_frame(tracks, grid, top_frac=0.10)

    detector = TacticalEventsDetector(fps=fps)
    events_by_frame = detector.detect(tracks, space_control_per_frame)
    ranked_windows, window_frames = rank_events_by_window(
        events_by_frame, tracks, space_control_per_frame, fps=fps)
    print(f"fast_render_1: {sum(detector.event_counts.values())} tactical events detected "
          f"across {len(ranked_windows)} windows")

    t0 = time.time()
    render_output1(tracks, fps=fps, annotated_frames=data['annotated_frames_with_ball'],
                   ranked_windows=ranked_windows, window_frames=window_frames)
    print(f"fast_render_1: render took {time.time() - t0:.1f}s "
          f"({data['end_frame'] - data['start_frame']} frames)")
    rename_longer_output(1)


if __name__ == '__main__':
    main()
