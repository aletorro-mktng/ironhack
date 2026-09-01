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

## Run the Local App

```bash
pip install -r requirements.txt
npm install
npm run api
```

In a second terminal:

```bash
npm run dev
```

Then open:

```text
http://127.0.0.1:5173
```

The new frontend is React + TypeScript + Vite. The Python backend is exposed through FastAPI at `http://127.0.0.1:8000`.

Migration is intentionally feature-by-feature. Podcast Studio is the first React slice.
Generator, Campaign Mode, Chapter Promos, Gallery and Library remain in the legacy
NiceGUI/Gradio interface until their own migration pass. Keep `nicegui` and `gradio`
installed until the last legacy route is removed.

## Backend Architecture

The FastAPI app is assembled in `backend/main.py`.

```text
backend/
├── api/
├── services/
├── models/
├── repositories/
├── workers/
└── main.py
```

Services accept and return ordinary Python/Pydantic data. They do not import React,
NiceGUI, Gradio, or frontend component objects.

Podcast preview and full-render requests return queued jobs immediately. Progress is
available through `GET /api/jobs/{job_id}` polling and
`GET /api/jobs/{job_id}/events` server-sent events. Job status is persisted under
`outputs/jobs/index.json`. The current worker is an isolated local queue, so it can be
replaced with Celery or another durable queue without changing the API contract.

## Frontend Architecture

The independent Vite app lives under `frontend/src/`.

```text
frontend/src/
├── api/
├── components/
├── features/
├── hooks/
├── routes/
├── stores/
├── styles/
└── types/
```

Validation uses TypeScript explicitly:

```bash
npm run generate:types
npm run typecheck
npm run build
```

`npm run generate:types` exports the FastAPI OpenAPI schema to
`frontend/src/types/openapi.json` and regenerates
`frontend/src/types/api.generated.ts`. Frontend API contracts should be imported
from those generated types instead of being maintained by hand.

To inventory a running legacy Gradio API during migration, run:

```bash
npm run catalog:legacy
```

Set `LEGACY_GRADIO_OPENAPI_URL` if the Gradio server is not exposing
`/openapi.json` at `http://127.0.0.1:7860`. The catalog is documentation only;
FastAPI remains the permanent product API.

## VSCode Agent Configuration

The project includes a lightweight agent configuration at:

```text
config/vscode_agent.json
```

Current configuration:

```json
{
  "project": "mythos-content-engine",
  "entrypoint": "backend/main.py",
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
4. Run the FastAPI backend with `npm run api` and the React frontend with `npm run dev`.
5. Review generated artifacts in `outputs/`, especially:
   - `*_filtered_context_*.md`
   - `*_generation_prompt_*.md`
   - `*_draft_*.md`
   - `outputs/feedback/*.md`

When extending the project, update `config/vscode_agent.json` if the entrypoint, output folder, or agent workflow changes.
