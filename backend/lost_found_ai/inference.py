import os, shutil, json, cv2
from PIL import Image

from .model import extract_features
from .video_utils import extract_frames
from .similarity import compute_similarity

BASE_DIR = os.path.dirname(__file__)
THRESHOLD = 0.60

with open(os.path.join(BASE_DIR, "metadata/camera_mapping.json")) as f:
    camera_map = json.load(f)

def find_match(lost_image, video):
    temp_dir = os.path.join(BASE_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    lost_path = os.path.join(temp_dir, "lost.jpg")
    video_path = os.path.join(temp_dir, "video.mp4")

    with open(lost_path, "wb") as f:
        shutil.copyfileobj(lost_image.file, f)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    lost_img = Image.open(lost_path).convert("RGB")
    lost_emb = extract_features(lost_img)

    best_score = 0
    best_result = None

    frames = extract_frames(video_path)

    for frame, timestamp in frames:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_emb = extract_features(img)

        score = compute_similarity(lost_emb, frame_emb)

        if score > best_score:
            best_score = score
            best_result = {
                "confidence": round(float(score), 2),
                "timestamp": round(float(timestamp), 2)
            }

    if best_result and best_score >= THRESHOLD:
        meta = list(camera_map.values())[0]
        return {
            "status": "MATCH_FOUND",
            "camera_id": meta["camera_id"],
            "room_no": meta["room_no"],
            "confidence": best_result["confidence"],
            "timestamp": best_result["timestamp"]
        }

    return {"status": "NO_MATCH"}
