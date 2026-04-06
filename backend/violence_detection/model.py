import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "violence_model.h5")

model = load_model(MODEL_PATH)

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
    frames = extract_frames(video_path)

    if frames is None:
        return "Error", 0

    predictions = []

    for frame in frames:
        frame = np.expand_dims(frame, axis=0)
        pred = model.predict(frame, verbose=0)[0][0]
        predictions.append(pred)

    avg_score = sum(predictions) / len(predictions)

    if avg_score > 0.5:
        return "Violence Detected", avg_score
    else:
        return "No Violence", avg_score