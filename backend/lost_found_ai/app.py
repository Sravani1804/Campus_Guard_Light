from fastapi import APIRouter, UploadFile, File
import os, shutil, json, cv2
from PIL import Image

from .model import extract_features
from .video_utils import extract_frames
from .similarity import compute_similarity
from utils.email_sender import send_alert
router = APIRouter(
    prefix="/lost-found",
    tags=["Lost & Found AI"]
)

# ---------------- CONFIG ----------------
MIN_THRESHOLD = 0.65        # minimum similarity to consider a frame
REQUIRED_FRAMES = 3         # number of frames required for confirmation
BASE_DIR = os.path.dirname(__file__)
# ----------------------------------------

# Load camera metadata
with open(os.path.join(BASE_DIR, "../utils/camera_mapping.json")) as f:
    camera_map = json.load(f)


@router.post("/analyze")
async def analyze(
    lost_image: UploadFile = File(...),
    video: UploadFile = File(...)
):
    temp_dir = os.path.join(BASE_DIR, "temp_videos")
    os.makedirs(temp_dir, exist_ok=True)

    lost_path = os.path.join(temp_dir, "lost.jpg")
    video_path = os.path.join(temp_dir, "video.mp4")

    # Save uploaded files
    with open(lost_path, "wb") as f:
        shutil.copyfileobj(lost_image.file, f)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Feature extraction for lost item
    lost_img = Image.open(lost_path).convert("RGB")
    lost_emb = extract_features(lost_img)

    frames = extract_frames(video_path)

    match_count = 0
    best_score = 0.0
    best_timestamp = None

    for frame, timestamp in frames:
        img = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        frame_emb = extract_features(img)
        score = float(compute_similarity(lost_emb, frame_emb))

        # Track best score
        if score > best_score:
            best_score = score
            best_timestamp = timestamp

        # Count strong matches
        if score >= MIN_THRESHOLD:
            match_count += 1

    # ---------------- FINAL DECISION ----------------
    if match_count >= REQUIRED_FRAMES:
        meta = camera_map.get(video.filename)

        if not meta:
            return {
                "status": "NO_MATCH",
                "reason": "Camera metadata not found"
            }
        send_alert(
    "📍 Lost Item Found",
    f"""
Camera ID: {meta['camera_id']}
Room No: {meta['room_no']}
Confidence: {round(best_score, 2)}
Timestamp: {round(float(best_timestamp), 2)} sec
"""
)

        return {
            "status": "MATCH_FOUND",
            "camera_id": meta["camera_id"],
            "room_no": meta["room_no"],
            "confidence": round(best_score, 2),
            "timestamp": round(float(best_timestamp), 2),
            "frames_matched": match_count
        }

    return {
        "status": "NO_MATCH",
        "confidence": round(best_score, 2),
        "frames_matched": match_count
    }
