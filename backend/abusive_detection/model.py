import joblib

# Load model + vectorizer
model = joblib.load("abusive_detection/abusive_model.pkl")
vectoriser = joblib.load("abusive_detection/abuse_vectoriser.pkl")

# Custom abusive words
custom_abusive_words = [
    "waste fellow",
    "useless",
    "idiot",
    "hurt",
    "stupid",
    "bloody",
    "nonsense",
    "shut up"
]

def predict(text):
    text_lower = text.lower()

    # Rule-based override
    for word in custom_abusive_words:
        if word in text_lower:
            return "Abusive"

    vec = vectoriser.transform([text])
    result = model.predict(vec)[0]

    return "Abusive" if result == 1 else "Normal"