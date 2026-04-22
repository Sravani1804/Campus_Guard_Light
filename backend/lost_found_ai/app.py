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


# ---------------- SIMPLE IMAGE SIMILARITY ----------------
def image_similarity(img1, img2):
    img1 = cv2.resize(img1, (128, 128))
    img2 = cv2.resize(img2, (128, 128))

    diff = cv2.absdiff(img1, img2)
    score = np.mean(diff)

    return score


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

    # Load lost image
    lost_img = cv2.imread(lost_path)

    frames = extract_frames(video_path)

    best_score = float("inf")
    best_timestamp = 0

    for frame, timestamp in frames:
        score = image_similarity(lost_img, frame)

        if score < best_score:
            best_score = score
            best_timestamp = timestamp

    # 🔥 FINAL DECISION
    # lower score = more similar
    if best_score < 40:
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
