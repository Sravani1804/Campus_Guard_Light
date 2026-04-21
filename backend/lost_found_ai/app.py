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


# ---------------- ORB FEATURE EXTRACTION ----------------
def extract_features(image: Image.Image):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    return descriptors


# ---------------- SIMILARITY ----------------
def compute_similarity(desc1, desc2):
    if desc1 is None or desc2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)

    return len(matches)


# ---------------- FRAME EXTRACTION ----------------
def extract_frames(video_path, interval=30):
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

    # Process lost image
    lost_img = Image.open(lost_path).convert("RGB")
    lost_desc = extract_features(lost_img)

    frames = extract_frames(video_path)

    best_score = 0
    best_timestamp = 0
    match_found = False

    for frame, timestamp in frames:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_desc = extract_features(img)

        score = compute_similarity(lost_desc, frame_desc)

        if score > best_score:
            best_score = score
            best_timestamp = timestamp

        # 🔥 Threshold (tune if needed)
        if score > 30:
            match_found = True

    # ---------------- RESULT ----------------
    if match_found:
        return {
            "status": "MATCH_FOUND",
            "camera_id": "Camera 2",
            "room_no": "Block B - Room 205",
            "confidence": int(best_score),
            "timestamp": round(best_timestamp, 2)
        }

    return {
        "status": "NO_MATCH",
        "confidence": int(best_score)
    }
