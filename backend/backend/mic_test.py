import speech_recognition as sr

r = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
    print("🎤 Say something...")
    audio = r.listen(source)

print("Recognizing...")
print(r.recognize_google(audio))
