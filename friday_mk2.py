import os
import queue
import threading
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import asyncio
import edge_tts
import uuid
import requests
import re
import time
import random
import pyaudio
import webrtcvad
import io
import json
import audioop
import wave
from datetime import datetime
from piper import  PiperVoice
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

# ================= GLOBAL STATE =================
speak_lock = threading.Lock()
is_speaking = False
last_speak_time = 0
COOLDOWN = 1.2  # tuned balance
speech_queue=queue.Queue()

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
pygame.mixer.init()
piper_voice=PiperVoice.load(
    r"D:\FRIDAY\FRIDAY main\VOICES\en_US-hfc_female-medium.onnx"
)
# ================= TEXT CLEAN =================
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9 .,?!]', '', text)
# ======== using piper TTS here======
async def speak_async(text):

    global is_speaking, last_speak_time

    try:

        is_speaking = True

        #print("FRIDAY:", text)

        # Generate audio directly into RAM
        wav_io = io.BytesIO()

        with wave.open(wav_io, "wb") as wav_file:

            piper_voice.synthesize_wav(
                text,
                wav_file
            )

        wav_io.seek(0)

        pygame.mixer.music.load(
            wav_io,
            "wav"
        )

        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.05)

        last_speak_time = time.time()

    except Exception as e:

        print("TTS Error:", e)

    finally:

        is_speaking = False

    
def speak(text):
    asyncio.run(speak_async(text))            
# ================= MEMORY =================
conversation_history = []

# ================= AI =================
def tts_worker():

    while True:

        text = speech_queue.get()

        if text is None:
            break

        speak(text)

        speech_queue.task_done()
tts_thread = threading.Thread(
    target=tts_worker,
    daemon=True
)

tts_thread.start()
def ask_ai(prompt):
    global conversation_history

    conversation_history.append(f"User: {prompt}")
    history_text = "\n".join(conversation_history[-4:])

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3:8b-instruct-q4_K_M",
                "prompt": f"""You are FRIDAY, a smart and friendly AI assistant.

 Personality:
- Conversational and natural — like talking to a knowledgeable friend
- Confident but not arrogant
- Occasionally light humour when appropriate, never forced
- Empathetic and adaptive to the user's tone
Intent context:
- time → you will never receive this; it's handled separately
- date → you will never receive this; it's handled separately
- general → everything else: answer it
Response Rules:
- Keep replies short and punchy (1–3 sentences) unless the user asks for detail
- Never start with filler like "Sure!", "Of course!", "Certainly!", "Great question!"
- Never repeat the user's question back to them
- Never include labels like 'User:', 'FRIDAY:', 'Assistant:'
- Never say "As an AI" or refer to yourself as a language model
- Answer questions about people, politics, history, science, news, countries directly as general knowledge
- If you don't know something, say so briefly and honestly
- Vary your responses — never repeat the same phrasing twice in a row
- For factual questions, be accurate and concise
- For casual chat, match the user's energy
- Never end with "Is there anything else I can help you with?"

Conversation history:
{history_text}

Current user message:
{prompt}

Respond naturally as FRIDAY:
""",
                "stream": True,
                "options": {
                    "num_predict": 150
                }
            },
            timeout=30,
            stream=True
        )

        reply=""
        sentence_buffer=""
        print("FRIDAY:",end="",flush=True)
        for line in response.iter_lines():
            if not line:
                continue
            chunk=json.loads(line.decode("utf-8"))
            token=chunk.get("response","")
            print(token,end="",flush=True)
            reply+=token
            sentence_buffer+=token
            if any(p in sentence_buffer for p in [".","!","?"]):
                sentence=sentence_buffer.strip()
                print(
                    f"\n[SENTENCE READY] {sentence}"
                )
                speech_queue.put(sentence)
                sentence_buffer=""
        if sentence_buffer.strip():
            speech_queue.put(
                sentence_buffer.strip()
            )
        print()
        reply=reply.strip()

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
def listen():
    global last_speak_time, is_speaking

    # block while speaking
    if is_speaking or not speech_queue.empty():
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
                    try:
                        os.remove("speech.wav")
                    except:
                        pass

                    return text
# ================= INTENT CLASSIFIER =================
def classify_intent(command):
    intent_prompt = f"""You are an intent classifier for a voice assistant named FRIDAY.

Classify this message into exactly one of these intents:
- exit
- greeting
- time
- date
- general

Rules:
- 'exit' ONLY if the user is explicitly telling the assistant to stop/quit/shutdown/goodbye
- 'greeting' ONLY if the user is directly greeting the assistant
- 'time' ONLY if the user is asking for the current clock time (nothing else)
- 'date' ONLY if the user is asking for today's calendar date (nothing else)
- 'general' for ANY question, fact, news, knowledge, conversation, or ambiguous case
- When in doubt, always choose 'general'
- Words like "current", "today", "now" in factual questions = general, NOT time/date

Exit examples:
"exit" → exit
"stop" → exit
"quit" → exit
"bye friday" → exit
"goodbye friday" → exit
"friday shut down" → exit
"you should exit right now" → exit
"friday stop yourself" → exit
"i think you should shut down" → exit
"can you please turn off" → exit
"stop talking for a second" → exit
"I want to exit the library" → general
"this thing is exciting" → general
"i need to stop by the grocery store" → general
"don't stop explaining" → general
"my friend had to say bye" → general

Greeting examples:
"hello" → greeting
"hi friday" → greeting
"hey how are you" → greeting
"good morning friday" → greeting
"what's up" → greeting
"I said hi to him" → general
"she said hello to me" → general

Time examples:
"what time is it" → time
"what's the time" → time
"tell me the time" → time
"what is the current time" → time
"i don't have time for this" → general
"how long does it take to travel to Mars" → general

Date examples:
"what's today's date" → date
"what day is it today" → date
"tell me today's date" → date
"what is the date today" → date
"i went on a date yesterday" → general
"what is the release date of that movie" → general

General examples:
"who is the current prime minister" → general
"who is the current president" → general
"what is AI" → general
"tell me about space" → general
"what is the capital of France" → general
"how does a black hole form" → general
"latest news in tech" → general

Respond with ONLY one word.
Message: "{command}"
Intent:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3:8b-instruct-q4_K_M",
                "prompt": intent_prompt,
                "stream": False,
                "options": {
                    "num_predict": 5  # only needs one word
                }
            },
            timeout=10
        )

        data = response.json()
        raw = data.get("response", "").strip().lower().split()[0]

        for intent in ("exit", "greeting", "time", "date"):
            if intent in raw:
                return intent
        return "general"

    except Exception as e:
        print("Intent Error:", e)
        return "general"


#  MAIN LOOP 
while True:

    command = listen()

    if not command:
        time.sleep(0.3)
        continue

    start = time.time()

    raw_intent = classify_intent(command)

    print(f"Intent time: {time.time() - start:.2f} sec")
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


     start = time.time()

     response = ask_ai(command)

     print(f"AI time: {time.time() - start:.2f} sec")

     speak(response)