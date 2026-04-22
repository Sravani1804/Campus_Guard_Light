from fastapi import APIRouter, UploadFile, File
import os
import shutil
import cv2

router = APIRouter(
    prefix="/lost-found",
    tags=["Lost & Found AI"]
)

BASE_DIR = os.path.dirname(__file__)


# ---------------- FRAME EXTRACTION ----------------
def extract_frames(video_path, interval=15):
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


# ---------------- GOOD MATCHES USING RATIO TEST ----------------
def orb_good_matches(img1, img2):
    orb = cv2.ORB_create(nfeatures=1000)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:   # 🔥 KEY FIX
            good.append(m)

    return len(good)


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
    lost_img = cv2.imread(lost_path, cv2.IMREAD_GRAYSCALE)

    frames = extract_frames(video_path)

    best_matches = 0
    best_timestamp = 0

    for frame, timestamp in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        matches = orb_good_matches(lost_img, gray)

        if matches > best_matches:
            best_matches = matches
            best_timestamp = timestamp

    # 🔥 FINAL STRICT DECISION
    if best_matches > 12:
        return {
            "status": "MATCH_FOUND",
            "camera_id": "Camera 2",
            "room_no": "Block B - Room 205",
            "confidence": int(best_matches),
            "timestamp": round(best_timestamp, 2)
        }

    return {
        "status": "NO_MATCH",
        "confidence": int(best_matches)
    }
