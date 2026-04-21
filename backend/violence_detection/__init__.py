import numpy as np
# from tensorflow.keras.models import load_model
from .utils import extract_frames

# Load model once
model = None

def predict_violence(video_path):
    frames = extract_frames(video_path)

    if frames is None:
        return "Error"

    sample = np.expand_dims(frames, axis=0)

    prediction = model.predict(sample)[0][0]

    return "Violence" if prediction > 0.5 else "Non-Violence"
