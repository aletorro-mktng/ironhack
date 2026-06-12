from __future__ import annotations

import os
import base64
import mimetypes
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-5.4-mini")

def generate_text(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Send a prompt to the selected LLM provider and return generated text.
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY was not found. Add it to your .env file."
        )

    client = OpenAI()

    response = client.responses.create(
        model=model,
        input=prompt
    )

    return response.output_text


def generate_text_with_image(prompt: str, image_path: str | Path, model: str = DEFAULT_MODEL) -> str:
    """
    Send a prompt plus an image to the selected LLM provider and return text.
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY was not found. Add it to your .env file."
        )

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    image_url = f"data:{mime_type};base64,{encoded}"

    client = OpenAI()

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
    )

    return response.output_text


if __name__ == "__main__":
    test_prompt = "Write one sentence confirming the Mythos Content Engine is working."

    result = generate_text(test_prompt)

    print("LLM test successful.")
    print(result)
