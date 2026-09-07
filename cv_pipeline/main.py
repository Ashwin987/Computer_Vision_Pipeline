from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
import time as _time
import os
import pickle
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from match_events import MatchEventsDetector
from transition_detector import TransitionDetector
from tactical_map import TacticalMapGenerator
from tqdm import tqdm

def _smooth_positions(tracks, window=5):
    """Moving-average over position_transformed to dampen per-frame H-matrix jitter."""
    from collections import defaultdict
    half = window // 2
    for obj in tracks:
        if obj in ('ball', 'referees'):
            continue
        pos_by_id = defaultdict(dict)
        for fn, frame in enumerate(tracks[obj]):
            for tid, info in frame.items():
                p = info.get('position_transformed')
                if p is not None:
                    pos_by_id[tid][fn] = p
        for tid, pos_dict in pos_by_id.items():
            fn_set = set(pos_dict)
            for fn in list(fn_set):
                nbrs = [pos_dict[f] for f in range(fn - half, fn + half + 1)
                        if f in fn_set]
                if nbrs:
                    tracks[obj][fn][tid]['position_transformed'] = [
                        sum(p[0] for p in nbrs) / len(nbrs),
                        sum(p[1] for p in nbrs) / len(nbrs),
                    ]

VIDEO_PATH   = 'Match_videos/121364_0.mp4'
CAL_STUB     = 'stubs/homography_stub_121364.pkl'
# Pointed at the ball-fallback-augmented cache (93.6% ball detection vs the
# original 17.5%) for demo rendering. The original stub is untouched — this
# is a read-only pointer swap, not a regeneration of stubs/track_stubs_121364.pkl.
TRACK_STUB   = 'stubs/track_stubs_121364_ball_fallback.pkl'
CAM_STUB     = 'stubs/camera_movement_stub_121364.pkl'


def main():
    # Read Video
    video_frames = read_video(VIDEO_PATH)

    # Real fps from the video file's own metadata, not a hardcoded assumption
    # (same pattern as fast_setup.py's _video_fps / analyze_formation_window.py —
    # a previously hardcoded fps=24 here didn't match this clip's real 25fps).
    _cap = cv2.VideoCapture(VIDEO_PATH)
    fps = float(_cap.get(cv2.CAP_PROP_FPS))
    _cap.release()

    view_transformer = ViewTransformer()

    # ── Calibration (stub after first run) ───────────────────────────────────
    if os.path.exists(CAL_STUB):
        print(f"Loading calibration stub: {CAL_STUB}")
        with open(CAL_STUB, 'rb') as f:
            homography_per_frame = pickle.load(f)
    else:
        from pitch_calibrator import SoccerNetCalibrator
        sc = SoccerNetCalibrator('pose/pitch_keypoints_v3/weights/best.pt')
        homography_per_frame = sc.calibrate_video(VIDEO_PATH)
        with open(CAL_STUB, 'wb') as f:
            pickle.dump(homography_per_frame, f)
        print(f"Calibration stub saved: {CAL_STUB}")

    # Replace None entries with the fixed fallback so ViewTransformer never
    # receives None when it tries to index homography_per_frame[frame_num][0]
    _fb_H     = view_transformer.persepctive_trasnformer.astype(np.float32)
    _fb_H_inv = np.linalg.inv(_fb_H).astype(np.float32)
    homography_per_frame = {
        fn: h if h is not None else (_fb_H, _fb_H_inv)
        for fn, h in homography_per_frame.items()
    }

    # ── Tracking (stub after first run) ──────────────────────────────────────
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=os.path.exists(TRACK_STUB),
                                       stub_path=TRACK_STUB, video_path=VIDEO_PATH)
    tracker.add_position_to_tracks(tracks)

    # ── Camera movement (stub after first run) ────────────────────────────────
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=os.path.exists(CAM_STUB),
        stub_path=CAM_STUB)
    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks, camera_movement_per_frame)

    # View Transformer
    view_transformer.add_transformed_position_to_tracks(
        tracks, homography_per_frame)

    # No smoothing — raw per-frame positions so speed estimation sees real movement
    _smooth_positions(tracks, window=1)

    # Diagnostic: print coordinate range of position_transformed
    xs, ys = [], []
    for frame in tracks['players']:
        for info in frame.values():
            p = info.get('position_transformed')
            if p is not None:
                try:
                    xs.append(float(p[0])); ys.append(float(p[1]))
                except Exception:
                    pass
    if xs:
        print(f"[diag] position_transformed x: {min(xs):.1f}..{max(xs):.1f}m  "
              f"y: {min(ys):.1f}..{max(ys):.1f}m  (expect 0-105m x 0-68m)")

    # Per-frame calibration-confidence proxy (fraction of tracked players
    # landing in plausible pitch bounds after the homography transform).
    # Confirmed against real data: frames where this drops low are exactly
    # where moderate, cap/teleport-filter-evading speed noise clusters —
    # feeding it to the speed estimator lets readings built on a shaky
    # frame carry that caveat instead of looking identical to a solid one.
    calibration_confidence_per_frame = view_transformer.compute_frame_confidence(tracks)
    _conf_vals = list(calibration_confidence_per_frame.values())
    _low  = sum(1 for v in _conf_vals if v < 0.70)
    _med  = sum(1 for v in _conf_vals if 0.70 <= v < 0.90)
    _high = sum(1 for v in _conf_vals if v >= 0.90)
    print(f"[diag] calibration confidence: high={_high} medium={_med} low={_low} "
          f"(of {len(_conf_vals)} frames)  mean={sum(_conf_vals)/len(_conf_vals):.2%}")

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator(fps=fps)
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(
        tracks, calibration_confidence_per_frame=calibration_confidence_per_frame)

    # Assign Player Teams — two-pass batch design (see TeamAssigner.resolve_all_teams
    # for why): Pass 1 gathers each player's evidence across the WHOLE clip and
    # decides one final team per player; only after fallback + merge have also
    # had their say does Pass 2 (apply_final_teams) write that single decision
    # across every frame the player appears in, so a player's colour is
    # consistent for their entire screen time instead of gray-then-coloured
    # partway through.
    team_assigner = TeamAssigner()
    _team_t0 = _time.time()
    team_assigner.resolve_all_teams(video_frames, tracks['players'], tracks['referees'])
    print(f"[team_assigner] Pass 1 (evidence gathering, whole clip): "
          f"{_time.time() - _team_t0:.1f}s")

    # Conservative end-of-clip fallback for players who never passed the
    # reliability gate on any single frame but had an excellent best-ever
    # color match (see TeamAssigner.finalize_fallback_assignments).
    fallback_assignments = team_assigner.finalize_fallback_assignments()
    if fallback_assignments:
        print(f"[team_assigner] fallback-assigned {len(fallback_assignments)} "
              f"player(s) who never passed the reliability gate: {fallback_assignments}")

    # Track fragmentation merging: an outfield player who briefly drops out
    # of tracking (occlusion) and gets re-acquired under a new ByteTrack ID
    # can leave two individually-unresolved fragments where a combined view
    # would resolve cleanly. Requires both spatial/timing proximity AND
    # appearance agreement between the two segments (position alone isn't
    # reliable enough — verified some proximity-only candidates have
    # contradictory team evidence). See TeamAssigner.merge_fragmented_tracks.
    #
    # exclude_pairs: a track just split by tracker.applied_splits (see
    # KNOWN_ID_SPLITS in trackers/tracker.py) is immediately adjacent
    # (zero gap) to the fragment it was split from, which would otherwise
    # look like a perfect merge candidate on proximity alone — merging it
    # back would silently undo the split.
    merged_pairs, rejected_pairs = team_assigner.merge_fragmented_tracks(
        video_frames, tracks['players'],
        exclude_pairs=list(tracker.applied_splits.items()))
    if merged_pairs:
        print(f"[team_assigner] merged {len(merged_pairs)} fragmented track pair(s):")
        for m in merged_pairs:
            print(f"    {m}")
    if rejected_pairs:
        print(f"[team_assigner] rejected {len(rejected_pairs)} candidate pair(s) "
              f"(proximity but not appearance, or contradictory evidence):")
        for r in rejected_pairs:
            print(f"    {r}")

    # Pass 2: write each player's FINAL team decision across every frame
    # they appear in.
    team_assigner.apply_final_teams(tracks['players'])

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tqdm(tracks['players'],
                                                  desc="Assigning ball possession")):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(
            player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(
                tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(
                team_ball_control[-1] if team_ball_control else 0)
    team_ball_control = np.array(team_ball_control)

    # Detect match events (passes, shots, interceptions)
    events_detector = MatchEventsDetector(fps=fps)
    match_events = events_detector.detect_events(tracks)
    events_detector.print_summary()

    # ── Tactical Events: computed ONCE, before any rendering ──────────────────
    # Position/movement-based events (SPRINT, BURST, PRESS, RECOVERY, OVERLAP,
    # SPACE, LATERAL_RUN, DROP, BREAK, ISOLATED) for render_output1.py's panel.
    # Reuses render_output3's exact pitch-clip polygon (derived from this
    # clip's own observed player positions, not a fixed constant — see
    # compute_pitch_verts_from_tracks) + the shared grid/nearest-player
    # algorithm (tactical_events.space_control) so the coarse per-frame
    # "space control" grid is computed a single time here rather than
    # recomputed per render pass.
    from render_output3 import STEP as _tac_step, _FALLBACK_PITCH_VERTS as _tac_fallback_verts
    from tactical_events.space_control import (
        build_pitch_mask, build_sampling_grid, compute_space_control_per_frame,
        compute_pitch_verts_from_tracks,
    )
    from tactical_events.tactical_events_detector import TacticalEventsDetector

    _tac_h, _tac_w = video_frames[0].shape[:2]
    _tac_pitch_verts = compute_pitch_verts_from_tracks(
        tracks, _tac_h, _tac_w, fallback_verts=_tac_fallback_verts)
    _tac_pitch_mask = build_pitch_mask(_tac_pitch_verts, _tac_h, _tac_w)
    _tac_grid, _tac_grid_rows, _tac_grid_cols, _tac_row_idx, _tac_col_idx = \
        build_sampling_grid(_tac_pitch_mask, _tac_step)
    space_control_per_frame = compute_space_control_per_frame(tracks, _tac_grid, top_frac=0.10)

    tactical_events_detector = TacticalEventsDetector(fps=fps)
    events_by_frame = tactical_events_detector.detect(tracks, space_control_per_frame)

    print("\nTactical events detected (single precompute pass):")
    for etype, cnt in sorted(tactical_events_detector.event_counts.items()):
        print(f"  {etype:10s}: {cnt}")
    print(f"  {'TOTAL':10s}: {sum(tactical_events_detector.event_counts.values())}\n")

    # ── Selection layer: rank events into 20s windows, top 10 per window ──────
    # Base weight + intensity (0-5, how far the trigger exceeded its threshold)
    # per event, computed ONCE. render_output1.py's carousel just looks up the
    # current window's already-ranked batch.
    from tactical_events.event_ranking import rank_events_by_window
    ranked_windows, window_frames = rank_events_by_window(
        events_by_frame, tracks, space_control_per_frame, fps=fps)

    print("Ranked tactical events by 20-second window:")
    for widx in sorted(ranked_windows.keys()):
        t0, t1 = widx * 20, widx * 20 + 20
        evs = ranked_windows[widx]
        print(f"\n  Window {widx}  [{t0//60:02d}:{t0%60:02d}-{t1//60:02d}:{t1%60:02d}]  "
              f"({len(evs)} event(s)):")
        for ev in evs:
            print(f"    {ev['type']:10s} P{ev['player_id']:<4d}  "
                  f"base={ev['base_weight']:2d}  intensity={ev['intensity']:.2f}  "
                  f"score={ev['score']:5.2f}   ({ev['metric']})")
    print()

    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame)

    ## Draw Speed and Distance
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(
        output_video_frames, tracks)

    ## Draw Match Events
    output_video_frames = events_detector.draw_events(
        output_video_frames, match_events)

    # Detect transitions and overlay on Output 1
    transition_detector = TransitionDetector(fps=fps)
    transitions = transition_detector.detect_transitions(
        team_ball_control, tracks)
    transition_detector.print_summary()
    output_video_frames = transition_detector.draw_transitions(
        output_video_frames, transitions)

    # Save Output 1 (tracking + transition alerts)
    save_video(output_video_frames, 'output_videos/output_video.avi')

    original_w = video_frames[0].shape[1]
    annotated_frames = [frame[:, :original_w] for frame in output_video_frames]

    # Output 2 — fitness sidebars flanking the full output_video frames
    from render_output2 import render_output2
    render_output2(output_video_frames, tracks, team_ball_control, fps=fps,
                   homography_per_frame=homography_per_frame)

    # Output 3 — tactical map (replaces tactical_video.avi)
    from render_output3 import render_output3
    render_output3(video_frames, tracks, team_ball_control, transitions, fps=fps,
                   homography_per_frame=homography_per_frame)

    # Output 4 — pressure heatmap warped to camera perspective
    from render_output4 import render_output4
    render_output4(video_frames, tracks, team_ball_control, view_transformer,
                   fps=fps, annotated_frames=annotated_frames,
                   homography_per_frame=homography_per_frame)

    # Output 6 — off-ball movement trails with camera compensation
    from render_output6 import render_output6
    render_output6(video_frames, tracks, team_ball_control, view_transformer,
                   fps=fps, annotated_frames=annotated_frames,
                   homography_per_frame=homography_per_frame,
                   camera_movement_per_frame=camera_movement_per_frame)

    # Output 1 — tracking video with precomputed tactical-events carousel panel
    from render_output1 import render_output1
    render_output1(tracks, fps=fps, annotated_frames=annotated_frames,
                   ranked_windows=ranked_windows, window_frames=window_frames)

    # ── Rename outputs to a "shorter_" prefix (automatic, rename-only) ────────
    # Runs only after every render above has already completed and saved.
    # Does not touch any rendering/processing logic above — it only renames
    # the files this run just wrote, so this pipeline's outputs can never
    # collide with the fast pipeline's "longer_" outputs.
    _rename_map = {
        'output_videos/output_video.avi': 'output_videos/shorter_output_video.avi',
        'output_videos/output1.avi':      'output_videos/shorter_output1.avi',
        'output_videos/output2.avi':      'output_videos/shorter_output2.avi',
        'output_videos/output3.avi':      'output_videos/shorter_output3.avi',
        'output_videos/output4.avi':      'output_videos/shorter_output4.avi',
        'output_videos/output6.avi':      'output_videos/shorter_output6.avi',
    }
    for _src, _dst in _rename_map.items():
        if os.path.exists(_src):
            os.replace(_src, _dst)
            print(f"Renamed {_src} -> {_dst}")

if __name__ == '__main__':
    main()
