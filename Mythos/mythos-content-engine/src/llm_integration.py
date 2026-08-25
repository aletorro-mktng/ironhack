from __future__ import annotations

import os
import base64
import mimetypes
from pathlib import Path
from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI, OpenAIError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-5.4-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def _format_openai_error(exc: OpenAIError) -> RuntimeError:
    if isinstance(exc, AuthenticationError):
        return RuntimeError(
            "OpenAI API key was rejected. Replace OPENAI_API_KEY in .env with a valid key, then restart the app."
        )
    return RuntimeError(f"OpenAI request failed: {exc.__class__.__name__}: {exc}")


def embeddings_available() -> bool:
    """True when an OpenAI key is present so semantic retrieval can run."""
    return bool(os.getenv("OPENAI_API_KEY"))


def embed_texts(texts, model: str = EMBEDDING_MODEL, batch_size: int = 128):
    """Embed texts with OpenAI and return an L2-normalized float32 numpy array.

    Rows are normalized so cosine similarity equals a dot product. Raises if no
    API key is set; callers should guard with ``embeddings_available()`` and fall
    back to keyword retrieval when embeddings are unavailable.
    """
    import numpy as np

    items = [str(text or "") for text in texts]
    if not items:
        return np.zeros((0, 0), dtype=np.float32)
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY was not found. Add it to your .env file.")

    client = OpenAI()
    vectors: list[list[float]] = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        try:
            response = client.embeddings.create(model=model, input=batch)
        except OpenAIError as exc:
            raise _format_openai_error(exc) from None
        vectors.extend(item.embedding for item in response.data)

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def generate_text(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Send a prompt to the selected LLM provider and return generated text.
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY was not found. Add it to your .env file."
        )

    client = OpenAI()

    try:
        response = client.responses.create(
            model=model,
            input=prompt
        )
    except OpenAIError as exc:
        raise _format_openai_error(exc) from None

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

    try:
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
    except OpenAIError as exc:
        raise _format_openai_error(exc) from None

    return response.output_text


if __name__ == "__main__":
    test_prompt = "Write one sentence confirming the Tell Tales Ink is working."

    result = generate_text(test_prompt)

    print("LLM test successful.")
    print(result)
