# 🔥 Lightweight version (No ML model, safe for deployment)

# Custom abusive words list
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

def predict(text: str):
    """
    Simple rule-based abusive detection
    Works without ML model (safe for Render deployment)
    """

    if not text:
        return "Normal"

    text_lower = text.lower()

    # Check for abusive keywords
    for word in custom_abusive_words:
        if word in text_lower:
            return "Abusive"

    return "Normal"
