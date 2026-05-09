import os
import gradio as gr
from dotenv import load_dotenv
from src.data_processor import process_input
from src.llm_processor import create_podcast_script, verify_and_tag_script
from src.tts_generator import generate_audio

load_dotenv()

VOICE_CHOICES = [
    "alloy",
    "ash",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
]

TAG_SUGGESTIONS = {
    "Laugh": "[laughs]",
    "Sigh": "[sighs]",
    "Cough": "[coughs]",
    "Breath": "[breath]",
    "Short pause": "[pause:0.5s]",
    "Pause": "[pause:1s]",
    "Long pause": "[pause:2s]",
}

SOUND_EFFECT_SUGGESTIONS = {
    "Ding": "[sfx:ding]",
    "Chime": "[sfx:chime]",
    "Transition": "[sfx:transition]",
    "Whoosh": "[sfx:whoosh]",
    "Applause": "[sfx:applause]",
}

MUSIC_CHOICES = [
    "None",
    "Generated: upbeat",
    "Generated: calm",
    "Generated: news",
    "Generated: tech",
    "Generated: warm",
]


def append_tag_to_script(script, tag_label):
    tag = TAG_SUGGESTIONS.get(tag_label) or SOUND_EFFECT_SUGGESTIONS.get(tag_label) or tag_label
    script = (script or "").rstrip()

    if not script:
        return tag

    return f"{script} {tag}"


def verify_script(script):
    try:
        if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
            return script, "Add your OPENAI_API_KEY in .env to verify and auto-tag the script."

        verified_script = verify_and_tag_script(script)
        return verified_script, "Script verified and lightly tagged. Review the edits before generating audio."

    except Exception as error:
        return script, f"Error: {str(error)}"


def generate_script(title, pasted_text, uploaded_file, url, length, speaker_count, script_type):
    try:
        podcast_input = process_input(
            pasted_text=pasted_text,
            uploaded_file=uploaded_file,
            url=url,
            title=title
        )

        if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
            return (
                "",
                f"{podcast_input.source_type.upper()} uploaded and {len(podcast_input.raw_text)} characters were extracted. Add your OPENAI_API_KEY in .env to generate the script and audio."
            )

        script = create_podcast_script(
            podcast_input=podcast_input,
            length=length,
            speaker_count=speaker_count,
            script_type=script_type
        )

        status = f"Script generated from {podcast_input.source_type} input. Review or edit it, then generate audio."

        return script, status

    except Exception as error:
        return "", f"Error: {str(error)}"


def generate_podcast_audio(
    script,
    speaker_1,
    voice_1,
    speaker_2,
    voice_2,
    speaker_3,
    voice_3,
    intro_music,
    intro_music_file,
    outro_music,
    outro_music_file
):
    try:
        if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
            return None, "Add your OPENAI_API_KEY in .env to generate audio."

        speaker_voices = {
            speaker_1: voice_1,
            speaker_2: voice_2,
            speaker_3: voice_3,
        }

        audio_path = generate_audio(
            script,
            speaker_voices=speaker_voices,
            intro_music=intro_music,
            intro_music_file=intro_music_file,
            outro_music=outro_music,
            outro_music_file=outro_music_file
        )
        return audio_path, "Podcast audio generated from the edited script."

    except Exception as error:
        return None, f"Error: {str(error)}"


with gr.Blocks(title="Podcast Studio") as demo:
    gr.Markdown("# Podcast Studio")
    gr.Markdown(
        "Turn pasted text, PDF files, DOCX files, or website articles into short podcast recaps."
    )

    with gr.Row():
        title = gr.Textbox(
            label="Podcast Title",
            value="Daily Lesson Recap"
        )

        length = gr.Radio(
            choices=["short", "medium"],
            value="short",
            label="Podcast Length"
        )

        speaker_count = gr.Radio(
            choices=["1 speaker", "2 speakers", "3 speakers"],
            value="2 speakers",
            label="Speakers"
        )

    script_type = gr.Dropdown(
        choices=[
            "interview",
            "debate",
            "roundtable",
            "solo recap",
            "narrative explainer",
            "news briefing",
            "teacher and student"
        ],
        value="interview",
        label="Script Type"
    )

    pasted_text = gr.Textbox(
        label="Paste transcript, notes, or article text",
        lines=8,
        placeholder="Paste your class notes, transcript, or article text here..."
    )

    uploaded_file = gr.File(
        label="Upload PDF or DOCX",
        file_types=[".pdf", ".docx"],
        type="filepath"
    )

    url = gr.Textbox(
        label="Website or article URL",
        placeholder="https://example.com/article"
    )

    generate_script_button = gr.Button("Generate Editable Script")

    script_output = gr.Textbox(
        label="Editable Podcast Script",
        lines=15,
        placeholder="Host: Welcome back. [laughs]\nCo-host: Today we are unpacking the main ideas. [pause:1s]\nHost: Let's begin. [sfx:transition]"
    )

    with gr.Row():
        tag_suggestion = gr.Dropdown(
            choices=list(TAG_SUGGESTIONS.keys()),
            value="Pause",
            label="Tag Suggestion"
        )

        add_tag_button = gr.Button("Add Tag")

    with gr.Row():
        sound_effect_suggestion = gr.Dropdown(
            choices=list(SOUND_EFFECT_SUGGESTIONS.keys()),
            value="Transition",
            label="Sound Effect"
        )

        add_sound_effect_button = gr.Button("Add Sound Effect")

    verify_script_button = gr.Button("Verify & Auto Tag Script")

    with gr.Row():
        speaker_1 = gr.Textbox(
            label="Speaker 1 Label",
            value="Host"
        )

        voice_1 = gr.Dropdown(
            choices=VOICE_CHOICES,
            value="coral",
            label="Speaker 1 Voice"
        )

    with gr.Row():
        speaker_2 = gr.Textbox(
            label="Speaker 2 Label",
            value="Co-host"
        )

        voice_2 = gr.Dropdown(
            choices=VOICE_CHOICES,
            value="alloy",
            label="Speaker 2 Voice"
        )

    with gr.Row():
        speaker_3 = gr.Textbox(
            label="Speaker 3 Label",
            value="Guest"
        )

        voice_3 = gr.Dropdown(
            choices=VOICE_CHOICES,
            value="verse",
            label="Speaker 3 Voice"
        )

    with gr.Row():
        intro_music = gr.Dropdown(
            choices=MUSIC_CHOICES,
            value="Generated: upbeat",
            label="Intro Music"
        )

        outro_music = gr.Dropdown(
            choices=MUSIC_CHOICES,
            value="Generated: warm",
            label="Outro Music"
        )

    with gr.Row():
        intro_music_file = gr.File(
            label="Upload Intro Music",
            file_types=[".mp3", ".wav", ".m4a", ".aac"],
            type="filepath"
        )

        outro_music_file = gr.File(
            label="Upload Outro Music",
            file_types=[".mp3", ".wav", ".m4a", ".aac"],
            type="filepath"
        )

    generate_audio_button = gr.Button("Generate Audio From Edited Script")

    audio_output = gr.Audio(
        label="Generated Audio",
        type="filepath"
    )

    status_output = gr.Textbox(
        label="Status"
    )

    generate_script_button.click(
        fn=generate_script,
        inputs=[title, pasted_text, uploaded_file, url, length, speaker_count, script_type],
        outputs=[script_output, status_output]
    )

    add_tag_button.click(
        fn=append_tag_to_script,
        inputs=[script_output, tag_suggestion],
        outputs=[script_output]
    )

    add_sound_effect_button.click(
        fn=append_tag_to_script,
        inputs=[script_output, sound_effect_suggestion],
        outputs=[script_output]
    )

    verify_script_button.click(
        fn=verify_script,
        inputs=[script_output],
        outputs=[script_output, status_output]
    )

    generate_audio_button.click(
        fn=generate_podcast_audio,
        inputs=[
            script_output,
            speaker_1,
            voice_1,
            speaker_2,
            voice_2,
            speaker_3,
            voice_3,
            intro_music,
            intro_music_file,
            outro_music,
            outro_music_file
        ],
        outputs=[audio_output, status_output]
    )


if __name__ == "__main__":
    demo.launch()
