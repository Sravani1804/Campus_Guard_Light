from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
import speech_recognition as sr
import tempfile
from pydub import AudioSegment 
from utils.email_sender import send_alert

from .model import predict




# 🔥 IMPORTANT
router = APIRouter()

# -------- TEXT API --------
class InputText(BaseModel):
    text: str

@router.post("/predict-audio")
async def detect_abuse_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            input_path = tmp.name

        sound = AudioSegment.from_file(input_path)
        wav_path = input_path + ".wav"
        sound.export(wav_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)

        print("Recognized:", text)

        result = predict(text)
        
        
        
            

        # 🔥 ADD EMAIL HERE
        if result == "Abusive":
            send_alert(
                "⚠️ Abusive Language Detected",
                f"""
Text: {text}
Result: {result}
"""
            )

        return {
            "recognized_text": text,
            "result": result
        }

    except Exception as e:
        return {"error": str(e)}