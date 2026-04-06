import cv2
import numpy as np

IMG_SIZE = 112
FRAMES_PER_VIDEO = 20

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    frame_count = 0

    while len(frames) < FRAMES_PER_VIDEO:
        ret, frame = cap.read()
        if not ret:
            break

        # 🔥 Skip frames to speed up
        if frame_count % 2 == 0:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame / 255.0
            frames.append(frame)

        frame_count += 1

    cap.release()

    if len(frames) == 0:
        return None

    # Padding
    while len(frames) < FRAMES_PER_VIDEO:
        frames.append(frames[-1])

    return np.array(frames, dtype="float32")