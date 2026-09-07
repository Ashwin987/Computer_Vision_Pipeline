"""
fast_common.py — shared loader for the fast_render_*.py scripts.

Completely separate from main.py (not imported, not referenced). This
module only loads the foundation caches produced by fast_setup.py
(stubs/fastpipe_*.pkl) and runs the same downstream enrichment every
render needs (positions, camera-compensated positions, world positions,
speed/distance, team assignment, ball possession, and the base
annotated-frames layer) so each fast_render_N.py script can just call
the real, untouched render_outputN() function from render_outputN.py.

This duplicates the *shape* of main.py's post-tracking pipeline (it has
to, to produce compatible data), but is independent, hand-written code —
main.py itself is never imported or read by this file.

BOUNDED WINDOW, NOT THE WHOLE VIDEO: the render_output*.py functions this
feeds all require a full in-memory Python list of frames (either passed
in directly, or in render_output3's case loaded internally) — that's
inherent to their existing, untouched implementation. For a long video
(e.g. 42,475 frames at 1920x1080 ≈ 250GB as a raw frame list) that is not
possible on a machine with 16.8GB RAM, regardless of how fast detection
was. So load_fast_pipeline() takes a [start_frame, end_frame) window and
only ever materializes that slice of frames into memory — the caller
(each fast_render_N.py) picks a window small enough to fit.
"""

import os
import pickle

import cv2
import numpy as np
from tqdm import tqdm

from trackers import Tracker
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

STUB_DIR    = 'stubs'
CACHE_TRACK = os.path.join(STUB_DIR, 'fastpipe_track_stubs.pkl')
CACHE_CAM   = os.path.join(STUB_DIR, 'fastpipe_camera_movement_stub.pkl')
CACHE_CAL   = os.path.join(STUB_DIR, 'fastpipe_homography_stub.pkl')
CACHE_META  = os.path.join(STUB_DIR, 'fastpipe_meta.pkl')

PLAYER_MODEL = 'models/best.pt'

DEFAULT_WINDOW_FRAMES = 3000   # ~2 minutes at 25fps — safe default chunk size


def fastpipe_cache_exists():
    return all(os.path.exists(p) for p in (CACHE_TRACK, CACHE_CAM, CACHE_CAL, CACHE_META))


def _read_frame_window(video_path, start_frame, end_frame):
    """Stream just [start_frame, end_frame) into a list — never the whole video."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def load_fast_pipeline(start_frame=0, end_frame=None):
    """Load fastpipe_ caches for the [start_frame, end_frame) window and run
    the shared enrichment pass over just that window.

    Returns a dict of everything a fast_render_N.py script needs, or
    None if the fastpipe_ cache doesn't exist yet (caller should print
    the "run fast_setup.py first" message and exit).
    """
    if not fastpipe_cache_exists():
        return None

    with open(CACHE_META, 'rb') as f:
        meta = pickle.load(f)
    video_path, fps, n_frames = meta['video_path'], meta['fps'], meta['n_frames']
    detect_every = meta.get('detect_every', 1)   # 1 = old caches predating this key (no skip)

    end_frame = n_frames if end_frame is None else min(end_frame, n_frames)
    start_frame = max(0, start_frame)
    if start_frame >= end_frame:
        raise ValueError(f"start_frame ({start_frame}) must be < end_frame ({end_frame})")

    print(f"fast_common: window [{start_frame}:{end_frame}) of {n_frames} total frames "
          f"from {video_path}")
    video_frames = _read_frame_window(video_path, start_frame, end_frame)

    with open(CACHE_TRACK, 'rb') as f:
        full_tracks = pickle.load(f)
    with open(CACHE_CAM, 'rb') as f:
        full_camera_movement = pickle.load(f)
    with open(CACHE_CAL, 'rb') as f:
        full_homography = pickle.load(f)

    tracks = {obj: full_tracks[obj][start_frame:end_frame]
              for obj in ('players', 'referees', 'ball')}
    camera_movement_per_frame = full_camera_movement[start_frame:end_frame]
    homography_per_frame = {i: full_homography.get(start_frame + i)
                            for i in range(end_frame - start_frame)}

    tracker = Tracker(PLAYER_MODEL)
    tracker.add_position_to_tracks(tracks)

    view_transformer = ViewTransformer()
    fb_H = view_transformer.persepctive_trasnformer.astype(np.float32)
    fb_H_inv = np.linalg.inv(fb_H).astype(np.float32)
    homography_per_frame = {
        fn: h if h is not None else (fb_H, fb_H_inv)
        for fn, h in homography_per_frame.items()
    }

    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    view_transformer.add_transformed_position_to_tracks(tracks, homography_per_frame)

    # Per-frame calibration-confidence proxy — see main.py's identical block
    # and ViewTransformer.compute_frame_confidence for the full rationale.
    # Without this, every speed/distance reading silently defaulted to
    # "high" confidence regardless of actual calibration quality, so a
    # shaky-calibration frame's reading looked identical to a solid one.
    calibration_confidence_per_frame = view_transformer.compute_frame_confidence(tracks)
    _conf_vals = list(calibration_confidence_per_frame.values())
    _low  = sum(1 for v in _conf_vals if v < 0.70)
    _med  = sum(1 for v in _conf_vals if 0.70 <= v < 0.90)
    _high = sum(1 for v in _conf_vals if v >= 0.90)
    print(f"[diag] calibration confidence: high={_high} medium={_med} low={_low} "
          f"(of {len(_conf_vals)} frames)  mean={sum(_conf_vals)/len(_conf_vals):.2%}")

    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    speed_and_distance_estimator = SpeedAndDistance_Estimator(fps=fps)
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(
        tracks, calibration_confidence_per_frame=calibration_confidence_per_frame)

    # Two-pass batch design — see TeamAssigner.resolve_all_teams and main.py's
    # identical block for the full rationale (offline video has the whole
    # clip available up front, so a player's early frames shouldn't render
    # gray just because their lock hadn't happened yet in sequential order).
    team_assigner = TeamAssigner()
    team_assigner.resolve_all_teams(video_frames, tracks['players'], tracks['referees'])

    # Conservative end-of-clip fallback — see main.py's identical block and
    # TeamAssigner.finalize_fallback_assignments for the full rationale.
    fallback_assignments = team_assigner.finalize_fallback_assignments()
    if fallback_assignments:
        print(f"[team_assigner] fallback-assigned {len(fallback_assignments)} "
              f"player(s) who never passed the reliability gate: {fallback_assignments}")

    # Track fragmentation merging — see main.py's identical block and
    # TeamAssigner.merge_fragmented_tracks for the full rationale.
    merged_pairs, rejected_pairs = team_assigner.merge_fragmented_tracks(
        video_frames, tracks['players'])
    if merged_pairs:
        print(f"[team_assigner] merged {len(merged_pairs)} fragmented track pair(s):")
        for m in merged_pairs:
            print(f"    {m}")
    if rejected_pairs:
        print(f"[team_assigner] rejected {len(rejected_pairs)} candidate pair(s):")
        for r in rejected_pairs:
            print(f"    {r}")

    # Pass 2: write each player's FINAL team decision across every frame
    # they appear in.
    team_assigner.apply_final_teams(tracks['players'])

    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tqdm(tracks['players'], desc="fast_common: assigning ball possession")):
        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox')
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox) if ball_bbox else -1
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
    team_ball_control = np.array(team_ball_control)

    # Ball detection is noticeably unreliable in this pipeline (half-res
    # inference + every-3rd-frame detection hits the ball hardest — a tiny,
    # fast-moving object). Hide the ball marker in the SHARED background
    # (used by outputs 3/4/6) by passing draw_annotations() an empty 'ball'
    # track list for THIS call only — the real tracks['ball'] (used for
    # has_ball assignment, tactical events, etc.) is untouched. This
    # doesn't touch trackers/tracker.py, main.py, or any render_output*.py —
    # draw_team_ball_control() (the "Team 1/2 Ball Control: X%" text) is
    # defined in tracker.py but is never actually called by either pipeline
    # today, so there's nothing active to suppress for that one.
    tracks_for_drawing = dict(tracks)
    tracks_for_drawing['ball'] = [{} for _ in tracks['ball']]
    output_video_frames = tracker.draw_annotations(video_frames, tracks_for_drawing, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame)
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(
        output_video_frames, tracks)

    original_w = video_frames[0].shape[1]
    annotated_frames = [f[:, :original_w] for f in output_video_frames]

    # Outputs 1 & 2 (unlike 3/4/6) want the ball marker: gap structure for
    # a typical fastpipe window is short enough to render sensibly (median
    # gap ~8 frames, 90% under ~1s, no gap over 3s in the audited window) —
    # already-interpolated tracks['ball'] (from interpolate_ball_positions
    # above) is dense enough to just draw, no new detection needed. Drawn
    # as a second pass on top of the already-built ball-less frames, using
    # the same tracker.draw_traingle() main.py's own ball marker uses, so
    # it looks identical rather than reimplementing the marker style.
    output_video_frames_with_ball = [f.copy() for f in output_video_frames]
    for frame_num, frame in enumerate(output_video_frames_with_ball):
        ball_dict = tracks['ball'][frame_num] if frame_num < len(tracks['ball']) else {}
        for _, ball in ball_dict.items():
            bbox = ball.get('bbox')
            if bbox:
                frame = tracker.draw_traingle(frame, bbox, (0, 255, 0))
        output_video_frames_with_ball[frame_num] = frame
    annotated_frames_with_ball = [f[:, :original_w] for f in output_video_frames_with_ball]

    return {
        'tracks': tracks,
        'video_frames': video_frames,
        'output_video_frames': output_video_frames,
        'annotated_frames': annotated_frames,
        'output_video_frames_with_ball': output_video_frames_with_ball,
        'annotated_frames_with_ball': annotated_frames_with_ball,
        'team_ball_control': team_ball_control,
        'view_transformer': view_transformer,
        'homography_per_frame': homography_per_frame,
        'camera_movement_per_frame': camera_movement_per_frame,
        'detect_every': detect_every,
        'fps': fps,
        'video_path': video_path,
        'start_frame': start_frame,
        'end_frame': end_frame,
    }


def rename_longer_output(n):
    """render_output{n}.py always saves to the fixed 'output_videos/output{n}.avi'
    path internally, regardless of what pipeline produced it — colliding with
    main.py's own output of the same run. Immediately after a fast_render_N.py
    call finishes, rename that file to 'output_videos/longer_output{n}.avi'
    so it doesn't sit at the same path main.py would write to next time.
    No render_output*.py file is touched — this only renames the file that
    was just written, from the calling fast_render_N.py script."""
    src = os.path.join('output_videos', f'output{n}.avi')
    dest = os.path.join('output_videos', f'longer_output{n}.avi')
    if os.path.exists(src):
        os.replace(src, dest)
        print(f"fast_common: renamed {src} -> {dest} (avoids colliding with main.py's output)")
    return dest


NO_CACHE_MESSAGE = (
    "No fastpipe_ cache found in 'stubs/'. Run the fast setup pipeline first:\n"
    "    python fast_setup.py\n"
    "(or  python fast_setup.py \"path\\to\\video.mp4\"  for a different video)"
)
