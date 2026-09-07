import cv2
from tqdm import tqdm

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    with tqdm(total=total, desc="Reading video", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            pbar.update(1)
    return frames

def save_video(ouput_video_frames,output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()
