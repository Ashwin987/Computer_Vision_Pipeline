from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys
from tqdm import tqdm
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

# Known ByteTrack ID reassignments: the SAME numeric track_id silently
# started representing a different physical player mid-clip, with no
# tracking gap in between (confirmed by a color-purity changepoint scan +
# 5 independent visual crops per pid — see detect_id_swaps.py and
# investigate_red_as_green.py). Applied once, in-memory, every time tracks
# are obtained here — never bakes into a cached stub on disk — so every
# downstream consumer (position, speed/distance, stamina, team assignment,
# tactical events) sees two consistent identities instead of one
# contaminated one.
#
# CLIP-SPECIFIC, not a general rule: these are two individually
# investigated, one-off ByteTrack mistakes found on ONE clip
# (121364_0.mp4), identified by the specific numeric track_id ByteTrack
# happened to assign that player in that run. Track ids are per-clip and
# restart from small integers every time — pid 4 and pid 7 exist on
# essentially any clip with more than a handful of players, and have no
# relationship whatsoever to this clip's specific swap events. Confirmed
# real bug (2026-08-01): the very first time this pipeline was pointed at
# different footage (an Algeria/Argentina segment), this fired
# unconditionally and silently split that clip's own pid 4 and pid 7 at
# frames 205/203 for no reason connected to that footage at all (952 and
# 955 frames re-keyed respectively — most of the clip). _apply_known_id_
# splits() now requires video_path to match KNOWN_ID_SPLITS_VIDEO before
# applying anything, specifically to stop that.
#
# If a different clip develops its own confirmed swap, add a similarly
# scoped, separately-video-gated entry — do NOT widen this dict to cover
# multiple clips' pids under one shared check.
KNOWN_ID_SPLITS_VIDEO = 'Match_videos/121364_0.mp4'
#
# pid 4, 121364_0.mp4 (the short demo clip): frames 0-204 are a green
# player (15 confident color samples, 100% pure). Frames 205-749 are a
# different, red player (520 confident samples, 100% pure). Zero-frame
# gap at the fn204->205 boundary, 7.3px bbox displacement — consistent
# with an IOU-association mistake during a close pass, not a lost/
# reacquired track.
#
# pid 7, 121364_0.mp4: frames 0-202 are a green player (visually confirmed
# solid green fn10-190; independently corroborated by detect_id_swaps.py's
# purity-changepoint scan, window fn186-194, no tracking gap). Frames
# 203-749 are a different, red player (visually confirmed solid red from
# fn215 through fn500+). No gap at the boundary — the raw bbox is smoothly,
# continuously tracked the whole time (48px -> 37px width narrowing right
# at fn203, then a few more frames of wobble before settling), consistent
# with an IOU-association mistake during a close-contact duel around
# fn199-215, not a lost/reacquired track. fn203 is the first frame the
# jersey color and bbox width both flip, so it's used as the split point
# even though the surrounding duel is noisier than pid 4's clean handoff.
#
# Each entry: old_id -> first frame number belonging to the NEW physical
# player. Everything from that frame onward is re-keyed to a new
# synthetic id; everything before keeps old_id.
KNOWN_ID_SPLITS = {
    4: 205,
    7: 203,
}


class Tracker:
    def __init__(self, model_path, use_ball_fallback=False):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        # Opt-in targeted fallback: only instantiated (and only ever loads
        # the Roboflow model) if a caller explicitly asks for it, since it
        # is 10-25x slower per triggered frame — see ball_fallback.py.
        self.ball_fallback = None
        if use_ball_fallback:
            from ball_fallback import BallFallback
            self.ball_fallback = BallFallback()
        # pid -> new synthetic id, populated by _apply_known_id_splits()
        # once tracks have been obtained. Callers (e.g. main.py) read this
        # to keep merge_fragmented_tracks from re-gluing a split pair back
        # together — see its exclude_pairs parameter.
        self.applied_splits = {}

    # NOTE on referee/player class-flip contamination (a single ByteTrack
    # track_id whose per-frame CLASS label flips between 'player' and
    # 'referee' for the same physical entity): that reconciliation used to
    # live here, but moved to TeamAssigner
    # (_reject_referee_colored_players, called from resolve_all_teams) —
    # deciding "is this really a referee" needs the fitted team colors
    # (does this entity's own jersey color confidently match EITHER team?
    # a real referee's kit is a third, distinct color that shouldn't), and
    # those colors don't exist yet at tracking time, before
    # TeamAssigner.assign_team_color has run. An earlier version tried to
    # decide this here using only a majority-vote on which label won more
    # often, which seemed reasonable (94-91% referee-labeled for 3
    # confirmed cases) until a 4th confirmed real referee tested at only
    # 26% — the model was flat-out wrong more often than right about that
    # one. Vote share isn't a reliable signal on its own; color is.

    def _apply_known_id_splits(self, tracks, video_path=None):
        """Re-key confirmed ByteTrack ID reassignments (KNOWN_ID_SPLITS)
        into separate synthetic ids. In-memory only; never mutates a
        pickled stub. Populates self.applied_splits.

        Clip-gated: only applies when video_path matches
        KNOWN_ID_SPLITS_VIDEO, the one clip these specific pid/frame
        corrections were derived from (see the comment above
        KNOWN_ID_SPLITS_VIDEO). video_path=None (caller didn't say what
        clip this is) is treated as "don't apply", not "apply anyway" —
        track ids are per-clip and pid 4/7 existing on some other clip is
        a coincidence, not evidence of a real swap there."""
        if not KNOWN_ID_SPLITS:
            return
        if video_path is None:
            return
        if os.path.normcase(os.path.abspath(video_path)) != \
                os.path.normcase(os.path.abspath(KNOWN_ID_SPLITS_VIDEO)):
            return
        existing_ids = set()
        for frame in tracks["players"]:
            existing_ids.update(frame.keys())
        next_id = (max(existing_ids) + 1) if existing_ids else 1

        for old_id, split_frame in KNOWN_ID_SPLITS.items():
            new_id = next_id
            next_id += 1
            moved = 0
            for frame_num, frame in enumerate(tracks["players"]):
                if frame_num < split_frame:
                    continue
                if old_id not in frame:
                    continue
                frame[new_id] = frame.pop(old_id)
                moved += 1
            if moved:
                self.applied_splits[old_id] = new_id
                print(f"[tracker] split id {old_id} -> {new_id} at frame "
                      f"{split_frame} ({moved} frame(s) re-keyed)")

    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size=20
        detections = []
        with tqdm(total=len(frames), desc="Detecting objects", unit="frame") as pbar:
            for i in range(0,len(frames),batch_size):
                detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
                detections += detections_batch
                pbar.update(len(frames[i:i+batch_size]))
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None, video_path=None,
                          progress_callback=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            self._apply_known_id_splits(tracks, video_path=video_path)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Capture the model's own goalkeeper/player distinction BEFORE it's
            # collapsed below — ByteTrack needs keeper+outfield unified into one
            # class for stable track continuity, but downstream (team_assigner)
            # needs to know which tracks were actually keepers. Stored in
            # Detections.data, which supervision carries through tracker updates
            # alongside each matched detection.
            detection_supervision.data['is_goalkeeper'] = np.array([
                cls_names[cid] == "goalkeeper" for cid in detection_supervision.class_id
            ])

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                is_gk = bool(frame_detection[5].get('is_goalkeeper', False))

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox, "is_goalkeeper": is_gk}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

            # Targeted fallback: only when the primary model just missed
            # the ball on THIS frame. Does not change what happens when
            # the fallback also misses (frame is left as-is, same as
            # before — existing interpolation/gap handling downstream is
            # untouched).
            if self.ball_fallback is not None and 1 not in tracks["ball"][frame_num]:
                recovered_bbox = self.ball_fallback.try_recover(frames[frame_num])
                if recovered_bbox is not None:
                    tracks["ball"][frame_num][1] = {"bbox": recovered_bbox, "fallback": True}

            # progress_callback(frame_num, total_frames) once per frame --
            # doesn't know or care about ball_fallback internals itself (a
            # caller with a `tracker` reference can read
            # tracker.ball_fallback.n_triggered when this fires); kept this
            # generic so the same hook covers plain detection-only runs too.
            if progress_callback is not None:
                progress_callback(frame_num, len(frames))

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        self._apply_known_id_splits(tracks, video_path=video_path)

        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        total = team_1_num_frames + team_2_num_frames
        team_1 = team_1_num_frames / total if total > 0 else 0
        team_2 = team_2_num_frames / total if total > 0 else 0

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_annotations(self,video_frames, tracks,team_ball_control):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                color = tuple(int(c) for c in color)   # numpy array → (int, int, int)
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))


            output_video_frames.append(frame)

        return output_video_frames