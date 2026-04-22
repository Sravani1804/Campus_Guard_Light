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


# ---------------- ORB GOOD MATCH RATIO ----------------
def orb_match_ratio(img1, img2):
    orb = cv2.ORB_create(nfeatures=1000)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # 🔥 KEY: ratio instead of count
    return len(good) / len(matches) if len(matches) > 0 else 0


# ---------------- SLIDING WINDOW ----------------
def find_best_match(lost_img, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    best_ratio = 0

    for size in [100, 150, 200]:
        if size > h or size > w:
            continue

        for y in range(0, h - size, 40):
            for x in range(0, w - size, 40):

                crop = gray[y:y+size, x:x+size]

                ratio = orb_match_ratio(lost_img, crop)

                if ratio > best_ratio:
                    best_ratio = ratio

    return best_ratio


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

    best_ratio = 0
    best_timestamp = 0

    for frame, timestamp in frames:
        ratio = find_best_match(lost_img, frame)

        if ratio > best_ratio:
            best_ratio = ratio
            best_timestamp = timestamp

    # 🔥 FINAL DECISION (THIS IS THE MAGIC NUMBER)
    if best_ratio > 0.15:
        return {
            "status": "MATCH_FOUND",
            "camera_id": "Camera 2",
            "room_no": "Block B - Room 205",
            "confidence": round(best_ratio, 2),
            "timestamp": round(best_timestamp, 2)
        }

    return {
        "status": "NO_MATCH",
        "confidence": round(best_ratio, 2)
    }
