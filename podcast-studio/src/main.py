import gradio as gr
from src.data_processor import process_input
from src.llm_processor import create_podcast_script
from src.tts_generator import generate_audio


def generate_podcast(title, pasted_text, uploaded_file, url, length):
    try:
        podcast_input = process_input(
            pasted_text=pasted_text,
            uploaded_file=uploaded_file,
            url=url,
            title=title
        )

        script = create_podcast_script(
            podcast_input=podcast_input,
            length=length
        )

        audio_path = generate_audio(script)

        status = f"Podcast generated successfully from {podcast_input.source_type} input."

        return script, audio_path, status

    except Exception as error:
        return "", None, f"Error: {str(error)}"


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

    generate_button = gr.Button("Generate Podcast")

    script_output = gr.Textbox(
        label="Generated Podcast Script",
        lines=15
    )

    audio_output = gr.Audio(
        label="Generated Audio",
        type="filepath"
    )

    status_output = gr.Textbox(
        label="Status"
    )

    generate_button.click(
        fn=generate_podcast,
        inputs=[title, pasted_text, uploaded_file, url, length],
        outputs=[script_output, audio_output, status_output]
    )


if __name__ == "__main__":
    demo.launch()