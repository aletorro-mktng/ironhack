# Mythos Content Engine

Mythos Content Engine generates brand-specific content for the Mortal Vengeance universe using curated markdown knowledge, reusable prompt templates, and a two-stage LLM pipeline.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file with:

```env
OPENAI_API_KEY=
LLM_MODEL=
ELEVENLABS_API_KEY=
ELEVENLABS_MODEL_ID=eleven_multilingual_v2
ELEVENLABS_OUTPUT_FORMAT=mp3_44100_128
```

## Run the CLI

```bash
python src/main.py
```

## Run the Uniqueness Comparison

Generate a saved Markdown report comparing a fresh ChatGPT baseline response
against the knowledge-base-driven Mythos pipeline:

```bash
python src/uniqueness_comparison.py
```

Try another content type or topic:

```bash
python src/uniqueness_comparison.py \
  --content-type press_release \
  --topic "Announce Mortal Vengeance winning the IndieReader Discovery Award"
```

Compare against a ChatGPT response you already saved:

```bash
python src/uniqueness_comparison.py \
  --content-type instagram_caption \
  --topic "Announce Mortal Vengeance winning the IndieReader Discovery Award" \
  --chatgpt-output-file path/to/chatgpt_response.md
```

The script saves the ChatGPT baseline prompt/output, Mythos filtered context,
Mythos generation prompt, Mythos draft, and final comparison report in
`outputs/`. The report includes a human assessment scorecard for you to judge
the comparison yourself.

## Run the Local UI

```bash
python src/nicegui_ui.py
```

Then open:

```text
http://127.0.0.1:7860
```

The NiceGUI shell supports structured input, draft generation, editable output, podcast generation, and saving revised drafts into `outputs/`.
The main Generator tab now includes native panels for blog posts, press releases, newsletters, character spotlights, Instagram, LinkedIn, and YouTube.

The Podcast Studio is built into the NiceGUI app as a native tab, so you do not need to start a separate Gradio interface for the main workflow.

## VSCode Agent Configuration

The project includes a lightweight agent configuration at:

```text
config/vscode_agent.json
```

Current configuration:

```json
{
  "project": "mythos-content-engine",
  "entrypoint": "src/nicegui_ui.py",
  "outputsDirectory": "outputs"
}
```

Use this file as the local editor/agent contract:

- `project`: names the active project folder.
- `entrypoint`: points agents or editor tasks to the CLI workflow.
- `outputsDirectory`: tells agents where generated drafts, prompts, filtered context, exports, audio packages, quote graphics, and feedback logs are stored.

Recommended VSCode workflow:

1. Open the `mythos-content-engine` folder as the active project.
2. Keep `.env` local and never commit API keys.
3. Run the CLI with `python src/main.py` for terminal-based generation.
4. Run the NiceGUI app with `python src/nicegui_ui.py` for structured briefs, human review, editing, approval logs, podcast generation, and exports.
5. Review generated artifacts in `outputs/`, especially:
   - `*_filtered_context_*.md`
   - `*_generation_prompt_*.md`
   - `*_draft_*.md`
   - `outputs/feedback/*.md`

When extending the project, update `config/vscode_agent.json` if the entrypoint, output folder, or agent workflow changes.
