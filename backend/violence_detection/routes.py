from fastapi import APIRouter, UploadFile, File
import tempfile

from .model import predict_violence
from utils.email_sender import send_alert
from utils.camera_utils import get_camera_details

router = APIRouter(
    prefix="/violence",
    tags=["Violence Detection"]
)


@router.post("/predict")
async def detect_violence(file: UploadFile = File(...)):
    try:
        # 🔥 Get camera details from filename
        camera_info = get_camera_details(file.filename)
        camera_id = camera_info["camera_id"]
        room_number = camera_info["room_no"]

        # Save temp video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            video_path = tmp.name

        # Predict
        result, score = predict_violence(video_path)

        # 🚨 Send email ONLY if violence detected
        if result == "Violence Detected":
            message = f"""
🚨 Violence Alert!

Camera: {camera_id}
Room: {room_number}
File: {file.filename}
Status: {result}
Confidence: {score:.2f}

Immediate action required!
"""
            send_alert(  "🚨 Violence Alert",message)

        return {
            "camera": camera_id,
            "room": room_number,
            "result": result,
            "confidence": round(score, 2)
        }

    except Exception as e:
        return {"error": str(e)}
    