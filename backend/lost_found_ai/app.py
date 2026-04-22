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


# ---------------- ORB FEATURES ----------------
def extract_orb(image):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=800)
    kp, desc = orb.detectAndCompute(gray, None)

    return desc


# ---------------- ORB MATCH ----------------
def orb_score(desc1, desc2):
    if desc1 is None or desc2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    return len(good)


# ---------------- COLOR SIMILARITY ----------------
def color_similarity(img1, img2):
    img1 = cv2.resize(img1, (100, 100))
    img2 = cv2.resize(img2, (100, 100))

    hist1 = cv2.calcHist([img1], [0,1,2], None, [8,8,8], [0,256]*3)
    hist2 = cv2.calcHist([img2], [0,1,2], None, [8,8,8], [0,256]*3)

    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


# ---------------- FRAME EXTRACTION ----------------
def extract_frames(video_path, interval=25):
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

    with open(lost_path, "wb") as f:
        shutil.copyfileobj(lost_image.file, f)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Load lost image
    lost_img = Image.open(lost_path).convert("RGB")
    lost_np = np.array(lost_img)
    lost_desc = extract_orb(lost_img)

    frames = extract_frames(video_path)

    best_score = 0
    best_timestamp = 0
    match_found = False

    for frame, timestamp in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ORB score
        frame_desc = extract_orb(Image.fromarray(frame_rgb))
        o_score = orb_score(lost_desc, frame_desc)

        # Color score
        c_score = color_similarity(lost_np, frame_rgb)

        # 🔥 Combined score
        final_score = o_score + (c_score * 20)

        if final_score > best_score:
            best_score = final_score
            best_timestamp = timestamp

        # 🔥 Threshold (balanced)
        if final_score > 25:
            match_found = True

    if match_found:
        return {
            "status": "MATCH_FOUND",
            "camera_id": "Camera 2",
            "room_no": "Block B - Room 205",
            "confidence": round(best_score, 2),
            "timestamp": round(best_timestamp, 2)
        }

    return {
        "status": "NO_MATCH",
        "confidence": round(best_score, 2)
    }
