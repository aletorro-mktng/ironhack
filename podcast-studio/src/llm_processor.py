import os
from openai import OpenAI
from dotenv import load_dotenv
from src.data_processor import PodcastInput

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def create_podcast_script(
    podcast_input: PodcastInput,
    length: str = "short",
    speaker_count: str = "2 speakers",
    script_type: str = "interview"
) -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    prompt = f"""
Create a {length} educational podcast script for {speaker_count}.

Title: {podcast_input.title}
Source type: {podcast_input.source_type}
Script type: {script_type}

Requirements:
- Friendly podcast tone
- Clear intro
- Explain the main ideas
- Include 3 to 5 key takeaways
- Short closing
- Do not invent facts
- Match the script type. For example, a debate should include distinct viewpoints, an interview should use questions and answers, and a roundtable should share turns across speakers.
- Use speaker labels at the start of each spoken line, like "Host:" and "Co-host:".
- Use only the speaker labels at line starts. Do not include stage directions.
- Keep each speaker turn concise and natural.
- You may add occasional performance tags like [laughs], [sighs], [coughs], or [pause:1s].
- Do not overuse performance tags.

Source material:
{podcast_input.raw_text[:12000]}
"""

    response = client.responses.create(
        model=model,
        input=prompt
    )

    return response.output_text


def verify_and_tag_script(script_text: str) -> str:
    if not script_text or len(script_text.strip()) < 50:
        raise ValueError("Script is too short to verify.")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    prompt = f"""
Review and lightly improve this editable podcast script.

Tasks:
- Preserve the user's wording as much as possible.
- Keep speaker labels at the start of spoken lines, like "Host:" or "Guest:".
- Do not add narration outside speaker lines.
- Fix obvious formatting issues in speaker labels.
- Add a small number of relevant tags where they improve performance or pacing.
- Allowed performance tags: [laughs], [sighs], [coughs], [breath], [pause:0.5s], [pause:1s], [pause:2s].
- Allowed sound effect tags: [sfx:ding], [sfx:chime], [sfx:transition], [sfx:whoosh], [sfx:applause].
- Do not overuse tags. Use them only where they naturally fit.
- Return only the improved script.

Script:
{script_text[:12000]}
"""

    response = client.responses.create(
        model=model,
        input=prompt
    )

    return response.output_text
