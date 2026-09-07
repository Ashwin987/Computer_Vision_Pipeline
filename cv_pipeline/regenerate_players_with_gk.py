"""
regenerate_players_with_gk.py — ONE-TIME: add the is_goalkeeper flag to
stubs/track_stubs_121364_ball_fallback.pkl WITHOUT re-running the
117-minute ball-fallback pass.

'players', 'referees', and 'ball' are fully independent keys in the
tracks dict (verified: no cross-referencing). The expensive part of
that stub was entirely in 'ball' (the targeted Roboflow fallback);
'players'/'referees' come from a cheap (~4.5 min) primary detection
pass that ran BEFORE the fallback and didn't yet carry the
is_goalkeeper flag (added to trackers/tracker.py after that stub was
built).

This re-runs ONLY that cheap primary detection (fresh, so the new
is_goalkeeper flag gets captured), keeps the existing 'ball' key
completely untouched, and overwrites the stub with the merged result.
"""
import pickle
import time

from utils import read_video
from trackers import Tracker

VIDEO_PATH = 'Match_videos/121364_0.mp4'
STUB_PATH = 'stubs/track_stubs_121364_ball_fallback.pkl'


def main():
    print(f"Loading existing stub: {STUB_PATH}")
    with open(STUB_PATH, 'rb') as f:
        old_tracks = pickle.load(f)
    old_ball = old_tracks['ball']   # the expensive part — reused verbatim
    n_ball_recovered = sum(1 for b in old_ball if b.get(1, {}).get('fallback'))
    print(f"  existing ball data: {len(old_ball)} frames, "
          f"{n_ball_recovered} fallback-recovered — will be preserved as-is")

    print(f"\nReading {VIDEO_PATH} ...")
    video_frames = read_video(VIDEO_PATH)
    print(f"  {len(video_frames)} frames loaded")

    tracker = Tracker('models/best.pt')   # no use_ball_fallback — don't need it here
    t0 = time.time()
    fresh_tracks = tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path=None,
                                             video_path=VIDEO_PATH)
    detect_time = time.time() - t0
    print(f"\nFresh primary detection done in {detect_time:.1f}s "
          f"({detect_time/60:.1f} min) — this is the ONLY stage re-run")

    n_gk_flagged = sum(
        1 for frame in fresh_tracks['players']
        for det in frame.values() if det.get('is_goalkeeper')
    )
    print(f"  {n_gk_flagged} goalkeeper-flagged player detections across "
          f"{len(fresh_tracks['players'])} frames")

    merged_tracks = {
        'players': fresh_tracks['players'],
        'referees': fresh_tracks['referees'],
        'ball': old_ball,   # untouched — the 117-minute fallback result
    }

    with open(STUB_PATH, 'wb') as f:
        pickle.dump(merged_tracks, f)
    print(f"\nWrote merged stub to {STUB_PATH}")
    print(f"  players: fresh (with is_goalkeeper flag)")
    print(f"  referees: fresh")
    print(f"  ball: unchanged from prior run ({n_ball_recovered} fallback-recovered frames preserved)")


if __name__ == '__main__':
    main()
