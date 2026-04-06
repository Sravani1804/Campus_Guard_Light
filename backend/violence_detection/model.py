import numpy as np
from tensorflow.keras.models import load_model
from .utils import extract_frames
import cv2

model = load_model("violence_detection/violence_model.h5")


def preprocess_frames(frames):
    processed = []

    for frame in frames:
        frame = cv2.resize(frame, (112, 112))
        frame = frame / 255.0
        processed.append(frame)

    return np.array(processed)


def predict_violence(video_path):
    frames = extract_frames(video_path)

    if frames is None or len(frames) == 0:
        return "Error"

    # 🔥 Take exactly 20 frames
    if len(frames) >= 20:
        frames = frames[:20]
    else:
        # Pad frames if less
        while len(frames) < 20:
            frames.append(frames[-1])

    frames = preprocess_frames(frames)

    # 🔥 Shape becomes (1, 20, 112, 112, 3)
    sample = np.expand_dims(frames, axis=0)

    prediction = model.predict(sample, verbose=0)[0][0]

    print("Prediction:", prediction)

    return "Violence" if prediction > 0.6 else "Non-Violence"