"""
fast_render_3.py — fast-pipeline counterpart to render_output3.py.

Loads from the fastpipe_ cache built by fast_setup.py for a bounded
[--start, --end) frame window and calls the real, unmodified
render_output3() function, then renames the result to
output_videos/longer_output3.avi (render_output3.py always saves to
the fixed output_videos/output3.avi internally, which would otherwise
collide with main.py's own output of the same run).

render_output3() now takes video_frames directly (it used to read its
own background video from a hardcoded RAW_VIDEO_PATH constant, always
Match_videos/121364_0.mp4 regardless of what was actually being
processed — fixed after that path independence turned out to mean it
silently rendered the wrong match's footage the first time this pipeline
was pointed at different source video). This script previously worked
around the old hardcoding by writing the bounded window out to a
temporary clip on disk and monkeypatching the module's path constant to
point at it; with a direct parameter, that round-trip (encode a temp
video, then have render_output3 immediately decode it back) is no longer
needed — the already-loaded window frames are passed straight through.

Usage:
    python fast_render_3.py
    python fast_render_3.py --start 1000 --end 4000
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

    import render_output3 as ro3

    t0 = time.time()
    # `transitions` (4th positional arg) is unused inside render_output3, so
    # it's safe to pass None here.
    ro3.render_output3(data['video_frames'], data['tracks'], data['team_ball_control'], None,
                       fps=data['fps'], homography_per_frame=data['homography_per_frame'])
    print(f"fast_render_3: render took {time.time() - t0:.1f}s "
          f"({data['end_frame'] - data['start_frame']} frames)")
    rename_longer_output(3)


if __name__ == '__main__':
    main()
