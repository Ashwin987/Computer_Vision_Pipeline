import pickle
import cv2
import numpy as np
import os
import sys 
sys.path.append('../')
from utils import measure_xy_distance

class CameraMovementEstimator():
    def __init__(self,frame):
        # Movement-detected threshold, in px, applied to the MEDIAN
        # displacement across all successfully-tracked features (see
        # get_camera_movement). Was 5 when this compared against the
        # single max-distance feature, which structurally overstates real
        # motion — a threshold tuned for that inflated signal is too
        # strict for the much cleaner median signal: at 5px it silently
        # zeroed out large parts of confirmed genuine multi-frame pans
        # (e.g. only 6/9 and 14/22 frames of two verified real panning
        # sequences reported nonzero movement). Recalibrated to 3 against
        # those same sequences: 1.5/2.0/3.0 all give full coverage of both
        # confirmed real pans with identical spike suppression, so 3 is a
        # middle-ground choice rather than the smallest that still works.
        self.minimum_distance = 3

        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20] = 1
        mask_features[:,900:1050] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance =3,
            blockSize = 7,
            mask = mask_features
        )

    def add_adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted
                    


    def estimate_frame_movement(self, old_gray, old_features, frame_gray):
        """Single-step camera-movement estimate between two consecutive
        grayscale frames. This is the one place the actual math lives —
        both get_camera_movement() (list-based) and any streaming caller
        (e.g. fast_setup.py, which can't hold a long video's frames in
        memory at once — see its own docstring) call this per frame pair
        instead of each keeping a separate copy of the algorithm.

        Median displacement across all successfully-tracked features, not
        the single feature with the largest displacement. A lone
        mistracked point (occlusion, motion blur, a feature drifting onto
        a moving player) can report a huge spurious displacement while
        ~100 other features agree the camera barely moved — picking the
        max let that one bad point dictate the whole frame's reported
        camera movement (confirmed: frame 327 of the tuned clip showed a
        72.5px "movement" via max-distance while the median across all
        tracked features was 0.5px). The median is robust to that single
        outlier by construction, and empirically still tracks genuine
        multi-frame pans smoothly (verified against several real panning
        sequences before rolling this out).

        Returns
        -------
        (camera_movement_x, camera_movement_y, movement_magnitude, new_features)
        new_features is the refreshed feature set if movement was detected
        (movement_magnitude > self.minimum_distance), otherwise old_features
        passed straight through — feed this back in as old_features on the
        next call, same as old_gray = frame_gray.
        """
        new_features, status, _ = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, old_features, None, **self.lk_params)
        status = status.reshape(-1) if status is not None else np.ones(len(old_features))

        dx_list, dy_list = [], []
        for i, (new, old) in enumerate(zip(new_features, old_features)):
            if status[i] == 0:
                continue
            dx, dy = measure_xy_distance(old.ravel(), new.ravel())
            dx_list.append(dx)
            dy_list.append(dy)

        if dx_list:
            camera_movement_x = float(np.median(dx_list))
            camera_movement_y = float(np.median(dy_list))
            movement_magnitude = (camera_movement_x ** 2 + camera_movement_y ** 2) ** 0.5
        else:
            camera_movement_x, camera_movement_y = 0.0, 0.0
            movement_magnitude = 0.0

        if movement_magnitude > self.minimum_distance:
            refreshed_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            return camera_movement_x, camera_movement_y, movement_magnitude, refreshed_features
        return camera_movement_x, camera_movement_y, movement_magnitude, old_features

    def get_camera_movement(self,frames,read_from_stub=False, stub_path=None):
        # Read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)

        camera_movement = [[0,0]]*len(frames)

        old_gray = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray,**self.features)

        for frame_num in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num],cv2.COLOR_BGR2GRAY)
            cam_x, cam_y, magnitude, old_features = self.estimate_frame_movement(
                old_gray, old_features, frame_gray)

            if magnitude > self.minimum_distance:
                camera_movement[frame_num] = [cam_x, cam_y]

            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)

        return camera_movement
    
    def draw_camera_movement(self,frames, camera_movement_per_frame):
        output_frames=[]

        for frame_num, frame in enumerate(frames):
            frame= frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha =0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame) 

        return output_frames