import pyaudio
import webrtcvad
import time
import audioop
import wave
from faster_whisper import WhisperModel

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

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

speaking = False
silence_start = None

SILENCE_LIMIT = 2

audio_frames = []

try:
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

                    print("Saved speech.wav")

                    # TRANSCRIBE
                    segments, _ = model.transcribe("speech.wav")

                    text = ""

                    for seg in segments:
                        text += seg.text

                    print("You said:", text.strip())

                    audio_frames = []

except KeyboardInterrupt:
    print("Stopping...")

stream.stop_stream()
stream.close()
p.terminate()

