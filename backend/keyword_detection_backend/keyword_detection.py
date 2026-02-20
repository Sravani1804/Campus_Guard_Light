from fastapi import APIRouter, UploadFile, File
import speech_recognition as sr
from pydub import AudioSegment
import os
import uuid
import pickle
import re

router = APIRouter(prefix="/keyword", tags=["Keyword Detection"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model/emergency_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "model/vectorizer.pkl"), "rb"))

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
        return {"prediction": "NORMAL", "recognized_text": ""}

    X = vectorizer.transform([clean_text])
    proba = model.predict_proba(X)[0][1]

    keyword_found = any(word in clean_text for word in emergency_keywords)

    if proba > 0.75 and keyword_found:
        prediction = "EMERGENCY"
    else:
        prediction = "NORMAL"

    os.remove(temp_name)
    os.remove(wav_path)

    return {
        "prediction": prediction,
        "recognized_text": text
    }