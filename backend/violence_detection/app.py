from fastapi import APIRouter, UploadFile, File
import tempfile

from .model import predict_violence

router = APIRouter(
    prefix="/violence",
    tags=["Violence Detection"]
)

@router.post("/predict")
async def detect_violence(file: UploadFile = File(...)):
    try:
        # Save temp video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            video_path = tmp.name

        result = predict_violence(video_path)

        return {"result": result}

    except Exception as e:
        return {"error": str(e)}