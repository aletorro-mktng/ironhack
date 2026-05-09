import os
from openai import OpenAI
from dotenv import load_dotenv
from src.data_processor import PodcastInput

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def create_podcast_script(podcast_input: PodcastInput, length: str = "short") -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    prompt = f"""
Create a short educational podcast script.

Title: {podcast_input.title}
Source type: {podcast_input.source_type}

Requirements:
- Friendly podcast host tone
- Clear intro
- Explain the main ideas
- Include 3 to 5 key takeaways
- Short closing
- Do not invent facts

Source material:
{podcast_input.raw_text[:12000]}
"""

    response = client.responses.create(
        model=model,
        input=prompt
    )

    return response.output_text