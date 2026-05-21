import speech_recognition as sr
import asyncio
import edge_tts
import os
import uuid
from playsound import playsound
conversation_started = False

# Initialize recognizer
recognizer = sr.Recognizer()
recognizer.energy_threshold = 100
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 1.8
recognizer.non_speaking_duration = 0.8

# Voice
VOICE = "en-US-JennyNeural"
import re

def clean_text(text):
    # remove emojis and weird symbols
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Speak function
async def speak_async(text):
    filename = f"voice_{uuid.uuid4().hex}.mp3"

    communicate = edge_tts.Communicate(
        text,
        VOICE,
        rate="+2%",
        pitch="+1Hz"
    )

    await communicate.save(filename)

    try:
        playsound(filename)
    finally:
        try:
            os.remove(filename)   # delete after playing
        except:
            pass

def speak(text):
    text = clean_text(text)   # remove emojis
    print("FRIDAY:", text)
    asyncio.run(speak_async(text))

# Listen function
def listen():
    with sr.Microphone() as source:
        print("Listening...")

        recognizer.adjust_for_ambient_noise(source, duration=1)

        try:
            audio = recognizer.listen(
                source,
                timeout=None,
                phrase_time_limit=15  # increased
            )
        except:
            return ""

    try:
        command = recognizer.recognize_google(audio)
        print("You:", command)
        return command.lower().strip()
    except:
        return ""
# Main loop
while True:
    command = listen()

    if command == "":
        continue

    # EXIT
    if "exit" in command or "bye" in command or "stop" in command:
        speak("Alright, talk to you later")
        break

    # FIRST TIME HELLO
    elif "hello" in command and not conversation_started:
        speak("Hey! What's up? What’s the plan for today?")
        conversation_started = True

    # HELLO AGAIN (DON’T RESET)
    elif "hello" in command and conversation_started:
        speak("Hey again  what’s going on?")

    else:
        speak("Give me a second...")