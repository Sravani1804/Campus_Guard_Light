from flask import Flask, request, jsonify, send_from_directory
import os
import whisper
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize app
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)   # ✅ FIX
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load models
print("Loading models...")
asr_model = whisper.load_model("base")
model = load_model("model.keras", compile=False)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100

# Clean text function
def clean_text(text):
    return text.lower()

# Routes to serve frontend files
@app.route("/")
def home():
    return send_from_directory("../frontend", "index.html")

@app.route("/style.css")
def style():
    return send_from_directory("../frontend", "style.css")

@app.route("/script.js")
def script():
    return send_from_directory("../frontend", "script.js")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    print("PREDICT CALLED")   # ✅ DEBUG

    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"})

    file = request.files["audio"]

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # ✅ FIX: Pass file path directly
    result = asr_model.transcribe(filepath)
    text = result["text"]

    print("Transcribed Text:", text)

    # Preprocess
    text_clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([text_clean])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

    # Predict
    prob = model.predict(pad)[0][0]
    prediction = "Abusive" if prob > 0.5 else "Safe"

    return jsonify({
        "transcription": text,
        "prediction": prediction,
        "probability": float(prob)
    })

# Run app
if __name__ == "__main__":
    app.run(debug=True)