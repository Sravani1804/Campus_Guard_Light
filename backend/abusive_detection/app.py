from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
import speech_recognition as sr
import tempfile
from pydub import AudioSegment 


from .model import predict

# 🔥 IMPORTANT
router = APIRouter()

# -------- TEXT API --------
class InputText(BaseModel):
    text: str

@router.post("/predict")
def detect_abuse(data: InputText):
    return {"result": predict(data.text)}


# -------- AUDIO API --------
@router.post("/predict-audio")
async def detect_abuse_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            input_path = tmp.name

        # 🔥 Convert ANY format → WAV
        sound = AudioSegment.from_file(input_path)
        wav_path = input_path + ".wav"
        sound.export(wav_path, format="wav")

        # Speech recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)

        print("Recognized:", text)  # Debug

        result = predict(text)

        return {
            "recognized_text": text,
            "result": result
        }

    except Exception as e:
        return {"error": str(e)}