import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from src.data_processor import PodcastInput

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MARKDOWN_HEADING_PATTERN = re.compile(r"^\s*\**\s*(title|script|podcast script)\s*:.*\**\s*$", re.IGNORECASE)
BRACKET_LABEL_PATTERN = re.compile(r"^\s*\[(HOST|CO-HOST|GUEST)\]\s*(.+)$", re.IGNORECASE)
COLON_LABEL_PATTERN = re.compile(r"^\s*\**\s*(Host|Co-host|Guest)\s*\**\s*:\s*\**\s*(.+)$", re.IGNORECASE)


def normalize_script_output(script_text: str) -> str:
    normalized_lines = []

    for line in script_text.splitlines():
        stripped = line.strip()

        if not stripped or MARKDOWN_HEADING_PATTERN.match(stripped):
            continue

        bracket_match = BRACKET_LABEL_PATTERN.match(stripped)
        colon_match = COLON_LABEL_PATTERN.match(stripped)

        if bracket_match:
            speaker = bracket_match.group(1).upper()
            text = bracket_match.group(2).strip().strip("*").strip()
            normalized_lines.append(f"[{speaker}] {text}")
            continue

        if colon_match:
            speaker = colon_match.group(1).upper()
            text = colon_match.group(2).strip().strip("*").strip()
            normalized_lines.append(f"[{speaker}] {text}")
            continue

        cleaned_line = stripped.strip("*").strip()

        if cleaned_line:
            normalized_lines.append(cleaned_line)

    return "\n".join(normalized_lines)


def create_podcast_script(
    podcast_input: PodcastInput,
    length: str = "short",
    speaker_count: str = "2 speakers",
    script_type: str = "interview",
    target_audience: str = "general learners",
    tone: str = "friendly"
) -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    prompt = f"""
Create a {length} educational podcast script for {speaker_count}.

Title: {podcast_input.title}
Source type: {podcast_input.source_type}
Script type: {script_type}
Target audience: {target_audience}
Tone: {tone}

Requirements:
- Return only the spoken script lines. Do not include a title, subtitle, heading, Markdown bold text, bullet list, or explanation.
- Do not wrap labels or text in Markdown formatting.
- Match the requested tone and target audience.
- Clear intro
- Explain the main ideas
- Include 3 to 5 key takeaways
- Short closing
- Do not invent facts
- Match the script type. For example, a debate should include distinct viewpoints, an interview should use questions and answers, and a roundtable should share turns across speakers.
- Automatically assign speaker roles based on the script type and speaker count.
- Use only these bracketed speaker labels when applicable: "[HOST]", "[CO-HOST]", and "[GUEST]".
- For 1 speaker, use "[HOST]" only.
- For 2 speakers, use "[HOST]" and "[CO-HOST]" unless the script type is interview; for interviews, use "[HOST]" and "[GUEST]".
- For 3 speakers, use "[HOST]", "[CO-HOST]", and "[GUEST]".
- Every spoken line must start with exactly one bracketed speaker label.
- Never use "Host:", "**Host:**", "Co-host:", "**Co-host:**", "Guest:", or "**Guest:**".
- Do not include stage directions.
- Keep each speaker turn concise and natural.
- Add occasional relevant performance tags automatically, such as [laughs], [sighs], [breath], [pause:0.5s], [pause:1s], or [pause:2s].
- Add occasional relevant sound effect tags automatically, such as [sfx:ding], [sfx:chime], [sfx:transition], [sfx:whoosh], or [sfx:applause].
- Do not overuse tags. They should support pacing and emotion, not clutter the script.

Correct output format example:
[HOST] Welcome back, everyone. [pause:0.5s]
[CO-HOST] Today we are unpacking the main idea.
[GUEST] I am excited to join this conversation.

Source material:
{podcast_input.raw_text[:12000]}
"""

    response = client.responses.create(
        model=model,
        input=prompt
    )

    return normalize_script_output(response.output_text)


def verify_and_tag_script(script_text: str) -> str:
    if not script_text or len(script_text.strip()) < 50:
        raise ValueError("Script is too short to verify.")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    prompt = f"""
Review and lightly improve this editable podcast script.

Tasks:
- Preserve the user's wording as much as possible.
- Return only spoken script lines. Remove titles, headings, Markdown bold formatting, bullets, and explanations.
- Convert speaker labels to bracketed labels: [HOST], [CO-HOST], and [GUEST].
- Every spoken line must start with exactly one bracketed speaker label.
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

    return normalize_script_output(response.output_text)
