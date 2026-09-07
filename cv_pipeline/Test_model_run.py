import cv2
import os

videos = [
    r'C:\UCLA\yolo model 2\Match_videos\ChelseaCity.mp4',
    r'C:\UCLA\yolo model 2\Match_videos\LiverpoolPSG_short.mp4',
    r'C:\UCLA\yolo model 2\Match_videos\121364_0.mp4',
]

output_folder = r'C:\UCLA\yolo model 2\label_frames'
os.makedirs(output_folder, exist_ok=True)
total = 0

for video_path in videos:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = 0
    video_name = os.path.basename(video_path).replace('.mp4', '')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % int(fps * 3) == 0:
            cv2.imwrite(f'{output_folder}/{video_name}_frame_{frame_num:06d}.jpg', frame)
            total += 1
        frame_num += 1
    cap.release()
    print(f"Done: {video_name} — {total} frames so far")

print(f"Total frames extracted: {total}")