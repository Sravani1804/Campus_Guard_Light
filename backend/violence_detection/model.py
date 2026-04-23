import cv2
import numpy as np

IMG_SIZE = 224


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return None

    # 🔥 FIX: deterministic frame selection (same frames every time)
    indices = np.linspace(0, total_frames - 1, 15, dtype=int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frames.append(gray)

    cap.release()

    if len(frames) < 2:
        return None

    return frames


def predict_violence(video_path):
    frames = extract_frames(video_path)

    if frames is None:
        return "No Violence", 0.0

    motion_scores = []

    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i - 1])

        # 🔥 FIX: normalize motion properly
        score = np.sum(diff) / (IMG_SIZE * IMG_SIZE)
        motion_scores.append(score)

    avg_motion = np.mean(motion_scores)

    # 🔥 FIX: stable threshold
    THRESHOLD = 18

    if avg_motion > THRESHOLD:
        return "Violence Detected", round(avg_motion / 50, 2)
    else:
        return "No Violence", round(avg_motion / 50, 2)
