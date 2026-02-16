import speech_recognition as sr
import pickle
import re

# Load trained model & vectorizer
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "emergency_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))


# Emergency keywords
emergency_keywords = [
    "help", "save me", "danger", "emergency", "attack",
    "fire", "thief", "gun", "knife", "police"
]

recognizer = sr.Recognizer()
mic = sr.Microphone()

print("🎧 Live listening started... Press CTRL + C to stop")

while True:
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("\nListening...")
            audio = recognizer.listen(source)

        # Speech to text
        text = recognizer.recognize_google(audio)
        print("You said:", text)

        # Ignore short noise
        if len(text.split()) < 3:
            print("⏭ Ignored (too short)")
            continue

        # Clean text
        clean_text = re.sub(r"[^a-zA-Z ]", "", text.lower())

        # Vectorize
        X = vectorizer.transform([clean_text])

        # Predict probability
        proba = model.predict_proba(X)[0]
        emergency_prob = proba[1]

        # Keyword check
        keyword_found = any(word in clean_text for word in emergency_keywords)

        # Final decision
        if emergency_prob > 0.75 and keyword_found:
            print("🚨🚨 EMERGENCY DETECTED 🚨🚨")
        else:
            print("✅ Normal speech")

    except sr.UnknownValueError:
        print("❌ Could not understand audio")

    except KeyboardInterrupt:
        print("\n🛑 Live detection stopped")
        break
