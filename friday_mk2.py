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
import keyboard
import io
import json
import audioop
import wave
from datetime import datetime
from piper import  PiperVoice
from classifier import classify_both
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

# ================= GLOBAL STATE =================
speak_lock = threading.Lock()
is_speaking = False
last_speak_time = 0
interrupt_requested=False
cancel_response=threading.Event()
assistant_busy=False
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
        #print("TTS START")

        #print("FRIDAY:", text)

        # Generate audio directly into RAM
        wav_io = io.BytesIO()

        with wave.open(wav_io, "wb") as wav_file:

            piper_voice.synthesize_wav(
                text,
                wav_file
            )

        wav_io.seek(0)
        pygame.mixer.music.stop()
        pygame.mixer.music.load(
            wav_io,
            "wav"
        )
        
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():

            if cancel_response.is_set():
              pygame.mixer.music.stop()
              break

            await asyncio.sleep(0.05)

        last_speak_time = time.time()

    except Exception as e:

        print("TTS Error:", e)

    finally:

        is_speaking = False
       # print("TTS END")

    
def speak(text):
    asyncio.run(speak_async(text)) 
def interrupt_friday():
    global interrupt_requested

    interrupt_requested = True
    cancel_response.set()

    pygame.mixer.music.stop()        
# ================= MEMORY =================
conversation_history = []
conversation_state="idle"
SESSION_START_TIME=datetime.now()


# ================= AI =================
def tts_worker():
    global interrupt_requested

    while True:
        try:
            text=speech_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if cancel_response.is_set():
            speech_queue.task_done()
            continue

        if text is None:
            break
       # print("WORKER:",text[:80])
        speak(text)

        if cancel_response.is_set():

          while not speech_queue.empty():
            try:
              speech_queue.get_nowait()
              speech_queue.task_done()
            except queue.Empty:
             break
        speech_queue.task_done()
        time.sleep(0.3)
tts_thread = threading.Thread(
    target=tts_worker,
    daemon=True
)

tts_thread.start()
def get_temporal_context():

    now = datetime.now()
    

    current_time = now.strftime("%I:%M %p")

    current_date = now.strftime("%B %d, %Y")

    elapsed_delta = now - SESSION_START_TIME

    elapsed_minutes = int(
        elapsed_delta.total_seconds() // 60
    )

    temporal_prompt = (
        f"Current Date: {current_date}\n"
        f"Current Time: {current_time}\n"
        f"Session Duration: {elapsed_minutes} minutes.\n"
    )

    return temporal_prompt


def ask_ai(prompt):

    global conversation_history
    global conversation_state
    global assistant_busy
    cancel_response.clear()
    assistant_busy=True
    

    conversation_history.append(f"User: {prompt}")
    history_text = "\n".join(conversation_history[-12:])
    time_context=get_temporal_context()
    

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

conversation state:
{conversation_state}
Use the conversation state to guide your response:

- idle:
  Treat the user's message as a new request. Do not assume an ongoing task.

- casual_chat:
  Respond naturally and conversationally. Match the user's tone.

- storytelling:
  Continue the current story or modify it according to the user's request. Maintain characters, events, and context from the conversation.

- coding:
  Treat the conversation as a coding/debugging session. Use previous code and errors from the conversation when relevant.

- planning:
  Continue the existing plan or help develop the next steps. Keep previous goals and decisions in context.

- question_answering:
  Give a direct and accurate explanation. Use the previous conversation when the question depends on earlier context.
{time_context}
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
            if cancel_response.is_set():
              print("\nAI STREAM INTERRUPTED")
              break
            chunk=json.loads(line.decode("utf-8"))
            token=chunk.get("response","")
            print(token,end="",flush=True)
            reply+=token
            sentence_buffer+=token
            if any(p in sentence_buffer for p in [".","!","?"]):
                sentence=sentence_buffer.strip()
               # print(
                  #  f"\n[SENTENCE READY] {sentence}"
                #)
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
    assistant_busy=False

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


def interrupt_key(event):
    global assistant_busy

    if assistant_busy:
        print("INTERRUPTED")
        interrupt_friday()
keyboard.on_press_key("left ctrl", interrupt_key)

#CLASSIFYER FOR conversation state

while True:
    command = listen()

    if not command:
        time.sleep(0.3)
        continue
    cancel_response.clear()
    interrupt_requested = False

    start = time.time()

    intent, intent_conf, conversation_state, state_conf = classify_both(
        command,
        conversation_state
    )

    print(f"Classifier time: {time.time() - start:.3f} sec")
    print(f"Detected intent: {intent} ({intent_conf})")
    print(f"Conversation State: {conversation_state} ({state_conf})")
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

     #speak(response)