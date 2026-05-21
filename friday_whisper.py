import os
import threading
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import asyncio
import edge_tts
import uuid
from playsound import playsound
import requests
import re
import time
import random
import pyaudio
import webrtcvad
import audioop
import wave
from datetime import datetime

# ================= GLOBAL STATE =================
speak_lock = threading.Lock()
is_speaking = False
last_speak_time = 0
COOLDOWN = 1.2  # tuned balance

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

VOICE = "en-US-JennyNeural"

# ================= TEXT CLEAN =================
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9 .,?!]', '', text)

# ================= TTS =================
async def speak_async(text):
    filename = f"temp_voice.mp3"

    communicate = edge_tts.Communicate(
        text,
        VOICE,
        rate="+15%",
        pitch="+0Hz"
    )

    await communicate.save(filename)

    playsound(filename)

    time.sleep(0.5)

    try:
        os.remove(filename)
    except Exception as e:
        print("Delete failed:", e)

def speak(text):
    global is_speaking, last_speak_time

    text = clean_text(text)
    print("FRIDAY:", text)

    def run():
        global is_speaking, last_speak_time
        with speak_lock:
            is_speaking = True
            try:
                asyncio.run(speak_async(text))
            except:
                pass
            is_speaking = False
            last_speak_time = time.time()

    threading.Thread(target=run, daemon=True).start()

# ================= MEMORY =================
conversation_history = []

# ================= AI =================
def ask_ai(prompt):
    global conversation_history

    conversation_history.append(f"User: {prompt}")
    history_text = "\n".join(conversation_history[-4:])

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral:7b-instruct-q4_0",
                "prompt": f"""You are FRIDAY, a smart and friendly AI assistant.

Rules:
- Speak naturally
- Keep replies short (1–2 sentences)
- Avoid repeating responses
- Do not include labels like 'User:' or 'FRIDAY:'
- Do not repeat the user's message
- Never repeat words or phrases unnecessarily
- Reply like a real conversational assistant
- Be concise unless the user asks for detail

Conversation history:
{history_text}

Current user message:
{prompt}

Respond naturally as FRIDAY:
""",
                "stream": False,
                "options": {
                    "num_predict": 150
                }
            },
            timeout=30
        )

        data = response.json()

        reply = data.get("response", "").strip()

        # remove repeated FRIDAY labels
        reply = re.sub(
            r'^(FRIDAY[:, ]*)+',
            '',
            reply,
            flags=re.IGNORECASE
        )

        # extra cleanup
        reply = reply.replace("User:", "")
        reply = reply.replace("Assistant:", "")
        reply = reply.strip()

        if not reply:
            reply = "I couldn't think of a proper response."

    except Exception as e:
        print("AI ERROR:", e)
        reply = "AI is not responding properly."

    conversation_history.append(f"FRIDAY: {reply}")

    return reply

# ================= WHISPER =================
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
    download_root=r"D:\whisper_models"
)

# ================= LISTEN =================
# ================= LISTEN =================
def listen():
    global last_speak_time, is_speaking

    # block while speaking
    if is_speaking:
        time.sleep(0.1)
        return ""

    # cooldown after speaking
    if time.time() - last_speak_time < COOLDOWN:
        time.sleep(0.1)
        return ""

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 320

    vad = webrtcvad.Vad(3)

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("Listening...")

    speaking = False
    silence_start = None

    SILENCE_LIMIT = 2

    audio_frames = []

    while True:

        frame = stream.read(CHUNK, exception_on_overflow=False)

        is_speech = vad.is_speech(frame, RATE)

        volume = audioop.rms(frame, 2)

        if volume < 1200:
            is_speech = False

        # SPEECH
        if is_speech:

            audio_frames.append(frame)

            if not speaking:
                speaking = True
                print("Speech started")

            silence_start = None

        # SILENCE
        else:

            if speaking:

                audio_frames.append(frame)

                if silence_start is None:
                    silence_start = time.time()

                elif time.time() - silence_start > SILENCE_LIMIT:

                    speaking = False
                    silence_start = None

                    print("Speech ended")

                    # SAVE AUDIO
                    wf = wave.open("speech.wav", "wb")
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(p.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(audio_frames))
                    wf.close()

                    # TRANSCRIBE
                    segments, _ = model.transcribe("speech.wav")

                    segments = list(segments)

                    if not segments:
                        audio_frames = []
                        return ""

                    text = ""

                    for seg in segments:
                        text += seg.text

                    text = text.strip().lower()

                    print("You:", text)

                    audio_frames = []

                    stream.stop_stream()
                    stream.close()
                    p.terminate()

                    return text
#  INTENT CLASSIFIER 
def classify_intent(command):

    intent_prompt = f"""You are an intent classifier for a voice assistant named FRIDAY.

Classify this message into exactly one intent:
- exit
- greeting
- time
- date
- general

Rules:
- 'exit' only if the user is telling the assistant itself to stop
- greetings only if directed at the assistant
- normal conversational usage should be 'general'

Examples:
"exit" → exit
"i think you should shut down"-> exit
"you should exit right now" → exit
"friday stop yourself" → exit
"I want to exit the library" → general
"this thing is exciting" → general
"hello" → greeting
"hey how are you" → greeting
"I said hi to him" → general
"what time is it" → time
"what's today's date" → date
"stop talking for a second" → exit
"goodbye friday" → exit
"can you please turn off" → exit
"i need to stop by the grocery store" → general
"don't stop explaining" → general
"my friend had to say bye" → general
"tell me the time" → time
"what is the current time" → time
"i don't have time for this" → general
"how long does it take to travel to Mars" → general
"back in my day" → general
"what day is it today" → date
"tell me today's date" → date
"what is the date today" → date
"i went on a date yesterday" → general
"what is the release date of that movie" → general

Respond with ONLY one word.

Message: "{command}"

Intent:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral:7b-instruct-q4_0",
                "prompt": intent_prompt,
                "stream": False,
                "options": {
                    "num_predict": 5
                }
            },
            timeout=10
        )

        data = response.json()

        intent = data.get("response", "").strip().lower()

        valid_intents = ["exit", "greeting", "time", "date", "general"]

        if intent not in valid_intents:
            intent = "general"

        return intent

    except Exception as e:
        print("Intent Error:", e)
        return "general"


#  MAIN LOOP 
while True:

    command = listen()

    if not command:
        time.sleep(0.3)
        continue

    raw_intent = classify_intent(command)

    print("Detected intent:", raw_intent)

    intent = "general"

    for i in ["exit", "greeting", "time", "date"]:
        if i == raw_intent:
            intent = i
            break

    # EXIT
    if intent == "exit":

        print("FRIDAY: Alright, goodbye")

        asyncio.run(
            speak_async("Alright, goodbye")
        )

        break

    # GREETINGS
    elif intent == "greeting":

        responses = [
            "Hey, what's up?",
            "Hi there, what are you up to?",
            "Hello, how's your day going?",
            "Hey again, need something?",
            "Hi, what can I help you with?"
        ]

        speak(random.choice(responses))

    # TIME
    elif intent == "time":

        speak(
            f"The time is {datetime.now().strftime('%I:%M %p')}"
        )

    # DATE
    elif intent == "date":

        speak(
            f"Today is {datetime.now().strftime('%B %d, %Y')}"
        )

    # GENERAL AI RESPONSE
    else:

        response = ask_ai(command)

        speak(response)