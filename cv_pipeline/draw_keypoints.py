import cv2, numpy as np

IMG   = r'C:\UCLA\yolo model 2\pitch_keypoints_data\train\images\00-05-11-00-05-43-000000_jpg.rf.caa79fdb1f45b979ed78f24247e0539a.jpg'
LABEL = r'C:\UCLA\yolo model 2\pitch_keypoints_data\train\labels\00-05-11-00-05-43-000000_jpg.rf.caa79fdb1f45b979ed78f24247e0539a.txt'
OUT   = r'C:\UCLA\yolo model 2\keypoint_map.jpg'

img = cv2.imread(IMG)
ih, iw = img.shape[:2]

with open(LABEL) as f:
    tokens = f.read().split()

# YOLO pose format: class cx cy w h  then 48*(x y conf)
kp_toks = tokens[5:]
for i in range(48):
    x  = float(kp_toks[i*3])
    y  = float(kp_toks[i*3+1])
    c  = float(kp_toks[i*3+2])
    px = int(x * iw)
    py = int(y * ih)
    if c == 0:
        col = (80, 80, 80)   # invisible keypoint — grey
    else:
        col = (0, 200, 255)  # visible — yellow-orange
    cv2.circle(img, (px, py), 6, col, -1)
    cv2.putText(img, str(i), (px + 7, py + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(img, str(i), (px + 7, py + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

cv2.imwrite(OUT, img)
print("Saved:", OUT)
