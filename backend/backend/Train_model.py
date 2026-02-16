import os
import re
import pickle
import pandas as pd
import speech_recognition as sr
from pydub import AudioSegment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# ---------- STEP 1: Convert audio to PCM WAV ----------
input_folders = [
    "campus_guard_audio/audio/emergency",
    "campus_guard_audio/audio/non_emergency"
]



output_base = "audio_pcm"

for folder in input_folders:
    label = folder.split("/")[-1]
    out_folder = f"{output_base}/{label}"
    os.makedirs(out_folder, exist_ok=True)

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            audio = AudioSegment.from_file(os.path.join(folder, file))
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(os.path.join(out_folder, file), format="wav")

print("✅ Audio converted to PCM WAV")

# ---------- STEP 2: Audio → Text ----------
recognizer = sr.Recognizer()
data = []

folders = {
    "audio_pcm/emergency": 1,
    "audio_pcm/non_emergency": 0
}


for folder, label in folders.items():
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            with sr.AudioFile(path) as source:
                audio = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio)
                except:
                    text = ""
            data.append([text.lower().strip(), label])

df = pd.DataFrame(data, columns=["text", "label"])
df = df[df["text"] != ""]

print("Samples after cleaning:", len(df))

# ---------- STEP 3: Train Model ----------
X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
if cm.shape == (2, 2):
    fp = cm[0][1]
    tn = cm[0][0]
    print("False Positive Rate:", fp / (fp + tn))
else:
    print("False Positive Rate: Not computable (single class in test set)")

# ---------- STEP 4: Save Model ----------
pickle.dump(model, open("emergency_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model & vectorizer saved")
