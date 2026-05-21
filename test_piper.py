from piper import PiperVoice
import wave
from playsound import playsound

voice = PiperVoice.load(
    r"D:\FRIDAY\FRIDAY main\VOICES\en_US-hfc_female-medium.onnx"
)

output_file = "test.wav"

print("Generating voice...")

with wave.open(output_file, "wb") as wav_file:

    voice.synthesize_wav(
        "Hello, I am FRIDAY bosedeakay. Piper offline voice is finally working, fuck you ",
        wav_file
    )

print("Playing voice...")

playsound(output_file)

print("Done.")