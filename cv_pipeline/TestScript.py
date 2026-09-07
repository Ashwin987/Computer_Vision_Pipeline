import shutil
import os

label_folder = r'C:\UCLA\yolo model 2\label_frames'
selected_folder = r'C:\UCLA\yolo model 2\frames_to_label'
os.makedirs(selected_folder, exist_ok=True)

total = 0
for prefix in ['ChelseaCity', 'LiverpoolPSG_short', '121364_0']:
    frames = sorted([f for f in os.listdir(label_folder) if f.startswith(prefix)])
    step = max(1, len(frames) // 67)
    selected = frames[::step][:67]
    for f in selected:
        shutil.copy(os.path.join(label_folder, f), os.path.join(selected_folder, f))
    total += len(selected)
    print(f"{prefix}: selected {len(selected)} frames")

print(f"Total frames to label: {total}")