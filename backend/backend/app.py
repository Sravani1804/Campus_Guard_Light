from flask import Flask, jsonify, request
import subprocess
import os
import pickle
import re

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model & vectorizer
model = pickle.load(open(os.path.join(BASE_DIR, "emergency_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

@app.route("/")
def home():
    return "Campus Guard Backend Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    if text.strip() == "":
        return jsonify({"error": "Empty input"}), 400

    text = clean_text(text)
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]

    result = "Emergency" if prediction == 1 else "Normal"
    return jsonify({"prediction": result})

@app.route("/detect", methods=["GET"])
def detect():
    subprocess.Popen(["python", os.path.join(BASE_DIR, "live_detection.py")])
    return jsonify({"status": "Live detection started"})

if __name__ == "__main__":
    app.run(debug=True)
