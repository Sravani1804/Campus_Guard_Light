import cv2
import numpy as np
import random

# 🔥 MODEL DISABLED FOR DEPLOYMENT
model = None

IMG_SIZE = 224
FRAMES_PER_VIDEO = 20


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while len(frames) < FRAMES_PER_VIDEO:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 2 == 0:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame / 255.0
            frames.append(frame)

        frame_count += 1

    cap.release()

    if len(frames) == 0:
        return None

    while len(frames) < FRAMES_PER_VIDEO:
        frames.append(frames[-1])

    return np.array(frames, dtype="float32")


def predict_violence(video_path):
    # 🔥 SIMULATED OUTPUT (for demo)
    score = random.uniform(0.3, 0.95)

    if score > 0.6:
        return "Violence Detected", round(score, 2)
    else:
        return "No Violence", round(score, 2)
