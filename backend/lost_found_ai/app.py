from fastapi import APIRouter, UploadFile, File

router = APIRouter(
    prefix="/lost-found",
    tags=["Lost & Found AI"]
)

@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    return {
        "status": "MATCH_FOUND",
        "camera_id": "Camera 2",
        "room_no": "Block B - Room 205",
        "confidence": 0.92,
        "timestamp": 12.5
    }
