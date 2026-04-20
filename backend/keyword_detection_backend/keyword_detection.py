from fastapi import APIRouter, UploadFile, File
import speech_recognition as sr
from pydub import AudioSegment
import os
import uuid
import re

router = APIRouter(prefix="/keyword", tags=["Keyword Detection"])

# 🔥 Removed ML model loading (no pickle, no vectorizer)

emergency_keywords = [
    "help", "save me", "danger", "emergency", "attack",
    "fire", "thief", "gun", "knife", "police"
]

def convert_to_wav(input_path):
    wav_path = input_path.replace(".webm", ".wav")
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(wav_path, format="wav")
    return wav_path


@router.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    temp_name = f"temp_{uuid.uuid4()}.webm"

    # Save uploaded file
    with open(temp_name, "wb") as f:
        f.write(await file.read())

    wav_path = convert_to_wav(temp_name)

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
    except:
        text = ""

    clean_text = re.sub(r"[^a-zA-Z ]", "", text.lower())

    if clean_text.strip() == "":
        os.remove(temp_name)
        os.remove(wav_path)
        return {"prediction": "NORMAL", "recognized_text": ""}

    # 🔥 Simple keyword-based detection (no ML)
    keyword_found = any(word in clean_text for word in emergency_keywords)

    if keyword_found:
        prediction = "EMERGENCY"
    else:
        prediction = "NORMAL"

    # Cleanup temp files
    os.remove(temp_name)
    os.remove(wav_path)

    return {
        "prediction": prediction,
        "recognized_text": text
    }
