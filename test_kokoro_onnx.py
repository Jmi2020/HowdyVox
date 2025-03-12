import soundfile as sf
from kokoro_onnx import Kokoro

kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
samples, sample_rate = kokoro.create(
    "Cowboys emerged as iconic figures of the American frontier, symbolizing a rugged individualism and unyielding resilience that defined a transformative era in United States history; their lives were marked by the grueling demands of cattle drives across vast, untamed landscapes, where the harsh realities of nature and the relentless pursuit of opportunity converged into a way of life that has fascinated generations; the mythos of the cowboy, steeped in both historical fact and romanticized narrative, has been perpetuated through literature, film, and music, creating an enduring image of the solitary, stoic hero whose adventures encapsulate the spirit of exploration and the relentless quest for freedom; as modern society reflects on this legacy, the cultural significance of cowboys persists in contemporary celebrations such as rodeos and festivals, reminding us that the values of courage, perseverance, and an intimate connection to the land remain integral to the national identity, inspiring both admiration and critical analysis of how myth and reality intertwine in the storied tapestry of the American West.", voice="am_michael", speed=1.0, lang="en-us"
)
sf.write("audio.wav", samples, sample_rate)
print("Created audio.wav")