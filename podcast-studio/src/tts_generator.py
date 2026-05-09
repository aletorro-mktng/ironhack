import os
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_audio(script_text: str, output_dir: str = "outputs") -> str:
    if not script_text or len(script_text.strip()) < 50:
        raise ValueError("Script is too short to generate audio.")

    Path(output_dir).mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"podcast_episode_{timestamp}.mp3"

    with client.audio.speech.with_streaming_response.create(
        model=os.getenv("TTS_MODEL", "tts-1"),
        voice=os.getenv("TTS_VOICE", "coral"),
        input=script_text[:4000],
    ) as response:
        response.stream_to_file(output_path)

    return str(output_path)
