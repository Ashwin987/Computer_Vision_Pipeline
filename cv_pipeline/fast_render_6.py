"""
fast_render_6.py — fast-pipeline counterpart to render_output6.py.

Loads from the fastpipe_ cache built by fast_setup.py for a bounded
[--start, --end) frame window and calls the real, unmodified
render_output6() function, then renames the result to
output_videos/longer_output6.avi (render_output6.py always saves to
the fixed output_videos/output6.avi internally, which would otherwise
collide with main.py's own output of the same run).

Usage:
    python fast_render_6.py
    python fast_render_6.py --start 1000 --end 4000
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

    import render_output6 as ro6

    # Identity-linking's PROXIMITY_PX was implicitly calibrated for
    # full-frame-rate (every-frame) detection. On a stride>1 fastpipe run,
    # a real player's measured position can drift further between the
    # last frame of a lost fragment and the first frame of its
    # replacement, purely because detection itself samples more coarsely
    # — verified against this clip's own relink near-misses: scaling by
    # the actual detect_every stride recovered most of them (33/53) while
    # correctly leaving the rest (400-800px away — physically impossible
    # for a real player to cover in the relevant window) unmatched. This
    # is a runtime override of the module constant, not an edit to
    # render_output6.py, so main.py's full-frame-rate use (stride=1, no
    # change) is untouched.
    stride = data.get('detect_every', 1)
    if stride > 1:
        base_proximity = ro6.PROXIMITY_PX
        ro6.PROXIMITY_PX = base_proximity * stride
        print(f"fast_render_6: scaled PROXIMITY_PX {base_proximity}px -> "
              f"{ro6.PROXIMITY_PX}px for detect_every={stride}")

    t0 = time.time()
    ro6.render_output6(data['video_frames'], data['tracks'], data['team_ball_control'],
                       data['view_transformer'], fps=data['fps'],
                       annotated_frames=data['annotated_frames'],
                       homography_per_frame=data['homography_per_frame'],
                       camera_movement_per_frame=data['camera_movement_per_frame'])
    print(f"fast_render_6: render took {time.time() - t0:.1f}s "
          f"({data['end_frame'] - data['start_frame']} frames)")
    rename_longer_output(6)


if __name__ == '__main__':
    main()
