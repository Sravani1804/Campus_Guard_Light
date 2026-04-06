import numpy as np
from tensorflow.keras.models import load_model
from .utils import extract_frames

# Load once (IMPORTANT)
model = load_model("violence_detection/violence_model.h5")

def predict_violence(video_path):
    frames = extract_frames(video_path)

    if frames is None:
        return "Error"

    sample = np.expand_dims(frames, axis=0)

    prediction = model.predict(sample, verbose=0)[0][0]

    return "Violence" if prediction > 0.5 else "Non-Violence"