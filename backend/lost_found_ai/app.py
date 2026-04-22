from fastapi import APIRouter, UploadFile, File
import os
import shutil
import cv2
import numpy as np
from PIL import Image

router = APIRouter(
    prefix="/lost-found",
    tags=["Lost & Found AI"]
)

BASE_DIR = os.path.dirname(__file__)


# ---------------- FRAME EXTRACTION ----------------
def extract_frames(video_path, interval=20):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            frames.append((frame, timestamp))

        count += 1

    cap.release()
    return frames


# ---------------- MAIN API ----------------
@router.post("/analyze")
async def analyze(
    lost_image: UploadFile = File(...),
    video: UploadFile = File(...)
):
    temp_dir = os.path.join(BASE_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    lost_path = os.path.join(temp_dir, "lost.jpg")
    video_path = os.path.join(temp_dir, "video.mp4")

    # Save files
    with open(lost_path, "wb") as f:
        shutil.copyfileobj(lost_image.file, f)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Load lost image (template)
    template = cv2.imread(lost_path)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    h, w = template.shape

    frames = extract_frames(video_path)

    best_score = 0
    best_timestamp = 0

    for frame, timestamp in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize frame if too small
        if gray.shape[0] < h or gray.shape[1] < w:
            continue

        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            best_timestamp = timestamp

    # 🔥 FINAL DECISION (VERY IMPORTANT)
    if best_score > 0.6:
        return {
            "status": "MATCH_FOUND",
            "camera_id": "Camera 2",
            "room_no": "Block B - Room 205",
            "confidence": round(float(best_score), 2),
            "timestamp": round(best_timestamp, 2)
        }

    return {
        "status": "NO_MATCH",
        "confidence": round(float(best_score), 2)
    }
