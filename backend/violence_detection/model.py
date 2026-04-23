import cv2
import numpy as np

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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)

        frame_count += 1

    cap.release()

    if len(frames) < 2:
        return None

    return frames


def predict_violence(video_path):
    frames = extract_frames(video_path)

    if frames is None:
        return "No Violence", 0.0

    motion_scores = []

    # 🔥 calculate motion between frames
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i - 1])
        score = np.mean(diff)
        motion_scores.append(score)

    avg_motion = np.mean(motion_scores)

    # 🔥 THRESHOLD (tune this)
    if avg_motion > 15:
        return "Violence Detected", round(avg_motion / 50, 2)
    else:
        return "No Violence", round(avg_motion / 50, 2)
