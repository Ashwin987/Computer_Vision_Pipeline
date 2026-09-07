"""
archive_outputs.py — copy the current output_videos/*.avi files into a
timestamped, video-named archive subfolder before the next pipeline run
overwrites them.

Purely additive: does not touch main.py, fast_setup.py, any
render_output*.py, or any fast_render_*.py script. It only COPIES files
(never moves or deletes the originals), so the next `python main.py` /
`python fast_setup.py` + `fast_render_*.py` run still writes to the same
fixed filenames as always — this script just saves a snapshot first.

Workflow:
    python main.py                               (or the fast pipeline)
    python archive_outputs.py "121364_0.mp4"      (immediately after, to
                                                    save this run's outputs
                                                    before the next run
                                                    overwrites them)

If no filename is given, falls back to the video path recorded in
stubs/fastpipe_meta.pkl (written by fast_setup.py), if that cache exists.
main.py itself doesn't persist which video it used anywhere on disk, so
running this with no argument after a plain `python main.py` run (with
no fastpipe_ cache present) isn't supported — pass the filename explicitly.

Usage:
    python archive_outputs.py "121364_0.mp4"
    python archive_outputs.py
"""

import os
import sys
import glob
import shutil
import pickle
from datetime import datetime

OUTPUT_DIR    = 'output_videos'
ARCHIVE_ROOT  = os.path.join(OUTPUT_DIR, 'archive')
FASTPIPE_META = os.path.join('stubs', 'fastpipe_meta.pkl')


def _infer_video_name():
    """Fall back to the video path recorded by fast_setup.py's cache, if any."""
    if os.path.exists(FASTPIPE_META):
        try:
            with open(FASTPIPE_META, 'rb') as f:
                meta = pickle.load(f)
            return os.path.basename(meta['video_path'])
        except Exception:
            pass
    return None


def main():
    if len(sys.argv) > 1:
        video_name = sys.argv[1]
    else:
        video_name = _infer_video_name()
        if video_name is None:
            print("No video filename given, and no stubs/fastpipe_meta.pkl to infer one from.")
            print('Usage: python archive_outputs.py "video_name.mp4"')
            sys.exit(1)
        print(f"No argument given -- inferred video name from {FASTPIPE_META}: {video_name}")

    video_stem = os.path.splitext(os.path.basename(video_name))[0]
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    dest_dir = os.path.join(ARCHIVE_ROOT, f"{video_stem}_{timestamp}")

    video_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, '*.avi')))
    if not video_files:
        print(f"No .avi files found directly in {OUTPUT_DIR}/ -- nothing to archive.")
        sys.exit(0)

    os.makedirs(dest_dir, exist_ok=True)

    copied = []
    for src in video_files:
        name = os.path.basename(src)
        dest = os.path.join(dest_dir, name)
        shutil.copy2(src, dest)
        copied.append(name)

    print(f"Archived {len(copied)} file(s) from {OUTPUT_DIR}/ -> {dest_dir}/")
    for name in copied:
        size_mb = os.path.getsize(os.path.join(dest_dir, name)) / 1e6
        print(f"  {name}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
