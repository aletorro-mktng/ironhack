# OpenAI Native RAG Lab

This repository contains a Python implementation of a Retrieval-Augmented Generation system using OpenAI native APIs.

## Files

| File | Description |
|------|-------------|
| `rag_openai_native.py` | Main Python script. Loads documents, chunks text, generates embeddings, performs vector search, and answers questions using RAG. |
| `embeddings_store.json` | Local embedding store generated from the PDF and podcast transcript chunks. |
| `podcast_transcript.txt` | Podcast transcript used as one source document. |
| `ai_hleg_ethics_guidelines_for_trustworthy_ai-en_87F84A41-A6E8-F38C-BFF661481B40077B_60419.pdf` | Trustworthy AI PDF document used as the second source document. |
| `requirements.txt` | Python dependencies required to run the project. |
| `lab_summary.md` | Short summary paragraph explaining the main design choices. |

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate