from fastapi import APIRouter, UploadFile, File
import os, shutil, json, cv2
from PIL import Image

from .model import extract_features
from .video_utils import extract_frames
from .similarity import compute_similarity

router = APIRouter(
    prefix="/lost-found",
    tags=["Lost & Found AI"]
)

BASE_DIR = os.path.dirname(__file__)

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

    # Save files
    with open(lost_path, "wb") as f:
        shutil.copyfileobj(lost_image.file, f)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Extract features
    lost_img = Image.open(lost_path).convert("RGB")
    lost_desc = extract_features(lost_img)

    frames = extract_frames(video_path)

    best_score = 0
    best_timestamp = 0

    for frame, timestamp in frames:
        img = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        frame_desc = extract_features(img)

        score = compute_similarity(lost_desc, frame_desc)

        if score > best_score:
            best_score = score
            best_timestamp = timestamp

    # 🔥 FINAL DECISION (TUNED)
    if best_score > 15:
        meta = list(camera_map.values())[0]

        return {
            "status": "MATCH_FOUND",
            "camera_id": meta["camera_id"],
            "room_no": meta["room_no"],
            "confidence": int(best_score),
            "timestamp": round(float(best_timestamp), 2)
        }

    return {
        "status": "NO_MATCH",
        "confidence": int(best_score)
    }
