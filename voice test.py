import asyncio
import edge_tts
from playsound import playsound

async def test(voice):
    file = "test.mp3"
    tts = edge_tts.Communicate("Hello, I am Friday", voice)
    await tts.save(file)
    playsound(file)

voices = [
    "en-US-JennyNeural",
    "en-US-AriaNeural",
    "en-GB-SoniaNeural",
    "en-IN-NeerjaNeural"
]

for v in voices:
    print("Testing:", v)
    asyncio.run(test(v))