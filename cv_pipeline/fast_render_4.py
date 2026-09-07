"""
fast_render_4.py — fast-pipeline counterpart to render_output4.py.

Loads from the fastpipe_ cache built by fast_setup.py for a bounded
[--start, --end) frame window and calls the real, unmodified
render_output4() function, then renames the result to
output_videos/longer_output4.avi (render_output4.py always saves to
the fixed output_videos/output4.avi internally, which would otherwise
collide with main.py's own output of the same run).

Usage:
    python fast_render_4.py
    python fast_render_4.py --start 1000 --end 4000
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

    from render_output4 import render_output4

    t0 = time.time()
    render_output4(data['video_frames'], data['tracks'], data['team_ball_control'],
                   data['view_transformer'], fps=data['fps'],
                   annotated_frames=data['annotated_frames'],
                   homography_per_frame=data['homography_per_frame'])
    print(f"fast_render_4: render took {time.time() - t0:.1f}s "
          f"({data['end_frame'] - data['start_frame']} frames)")
    rename_longer_output(4)


if __name__ == '__main__':
    main()
