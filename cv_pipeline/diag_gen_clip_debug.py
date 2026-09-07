"""Generate pitch_clip_debug.jpg without running the full render."""
import sys, os, pickle
sys.path.insert(0, r'c:\UCLA\yolo model 2')
import cv2, numpy as np
from utils import get_center_of_bbox, get_foot_position
from team_assigner import TeamAssigner

TRACK_STUB = 'stubs/track_stubs_121364.pkl'
CAM_STUB   = 'stubs/camera_movement_stub_121364.pkl'
SRC_VIDEO  = 'output_videos/output_video.avi'
RAW_VIDEO  = 'Match_videos/121364_0.mp4'
FRAME      = 50

with open(TRACK_STUB, 'rb') as f:  tracks = pickle.load(f)
with open(CAM_STUB,   'rb') as f:  cam_mv = pickle.load(f)

# Add positions
for obj, obj_t in tracks.items():
    for fn, track in enumerate(obj_t):
        for tid, ti in track.items():
            bbox = ti['bbox']
            pos  = get_center_of_bbox(bbox) if obj == 'ball' else get_foot_position(bbox)
            ti['position'] = pos
for obj, obj_t in tracks.items():
    for fn, track in enumerate(obj_t):
        cam = cam_mv[fn]
        for tid, ti in track.items():
            p = ti['position']
            ti['position_adjusted'] = (p[0]-cam[0], p[1]-cam[1])

# Selective frame load for team assignment
NEED = {0, 10, 20, 30, 40, FRAME}
cap  = cv2.VideoCapture(RAW_VIDEO)
vf   = {}
for fn in sorted(NEED):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frm = cap.read()
    if ret: vf[fn] = frm
cap.release()

class SFL:
    def __init__(self, d, n): self.d=d; self.n=n
    def __len__(self): return self.n
    def __getitem__(self, i): return self.d[i]

ta = TeamAssigner()
ta.assign_team_color(SFL(vf, len(tracks['players'])), tracks['players'])
for fn in sorted(NEED):
    if fn not in vf: continue
    for pid, info in tracks['players'][fn].items():
        team = ta.get_player_team(vf[fn], info['bbox'], pid)
        tracks['players'][fn][pid]['team'] = team

# Draw debug image
cap2 = cv2.VideoCapture(SRC_VIDEO)
cap2.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
ret2, base = cap2.read()
cap2.release()
if not ret2:
    cap3 = cv2.VideoCapture(RAW_VIDEO)
    cap3.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
    ret3, base = cap3.read()
    cap3.release()

orig_h, orig_w = base.shape[:2]
dbg = base.copy()

pitch_clip = [(110.0,1035.0),(265.0,275.0),(1750.0,260.0),(1900.0,1035.0)]
pts = np.array([[int(x),int(y)] for x,y in pitch_clip], dtype=np.int32)
cv2.polylines(dbg, [pts], True, (0, 220, 0), 3)
labels = ['P1 BL', 'P2 TL', 'P3 TR', 'P4 BR']
for i, (px, py) in enumerate(pitch_clip):
    cv2.circle(dbg, (int(px), int(py)), 10, (0, 0, 255), -1)
    cv2.putText(dbg, labels[i], (int(px)+12, int(py)+8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)

for pid, info in tracks['players'][FRAME].items():
    adj  = info.get('position_adjusted')
    team = info.get('team', 0)
    if adj is None: continue
    cx, cy = int(float(adj[0])), int(float(adj[1]))
    col = (80,80,220) if team==1 else (80,220,80) if team==2 else (160,160,160)
    cv2.circle(dbg, (cx, cy), 8, col, -1)
    cv2.circle(dbg, (cx, cy), 8, (0,0,0), 1)
    cv2.putText(dbg, str(pid), (cx+10, cy+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

cv2.imwrite('pitch_clip_debug.jpg', dbg)
print(f"Saved pitch_clip_debug.jpg  ({orig_w}x{orig_h}, frame {FRAME})")
print(f"Polygon: {pitch_clip}")
