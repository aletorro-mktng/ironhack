# Podcast Studio

Turn pasted text, PDF files, DOCX files, or website articles into short podcast recaps.

## Setup

1. Create and activate the virtual environment:
   ```bash
   cd /Users/alejandrotorresdelarocha/ironhack/podcast-studio
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create your environment file:
   ```bash
   cp .env.example .env
   ```

4. Open `.env` and set your `OPENAI_API_KEY`.

## Run

```bash
source .venv/bin/activate
python -m src.main
```

## Notes

- `.env` is excluded from version control by `.gitignore`.
- If you are using a different OpenAI model or TTS voice, update the values in `.env`.
