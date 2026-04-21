from fastapi import APIRouter, UploadFile, File

router = APIRouter(prefix="/lost-found", tags=["Lost & Found"])


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # 🔥 Dummy response (no heavy ML)

    return {
        "status": "MATCH_FOUND",
        "camera_id": "Camera 3",
        "room_no": "Block A - Room 102",
        "confidence": 0.87,
        "timestamp": "12.5"
    }
