# Mythos Content Engine - Project Requirements

## 1. Project Overview

The Mythos Content Engine is a Python-based content generation system for creating brand-consistent marketing, editorial, and social media assets for the Mythos universe and the Mortal Vengeance series.

The system should combine curated knowledge base files, reusable prompt templates, and LLM integration into a repeatable pipeline that can generate drafts for multiple platforms.

## 2. Primary Goals

- Store brand, story, character, theme, and market context in structured markdown files.
- Load relevant knowledge base material into prompts.
- Support reusable templates for different content formats.
- Generate content drafts through an LLM integration layer.
- Save generated outputs to the `outputs/` directory.
- Keep the project modular so each part can be tested and replaced independently.

## 3. Project Scope

### In Scope

- Markdown-based knowledge base management.
- Prompt template loading and formatting.
- Basic document processing for local text and markdown files.
- Content pipeline orchestration.
- LLM provider abstraction.
- CLI entry point through `src/main.py`.
- Output file generation.

### Out of Scope for MVP

- Web dashboard.
- User authentication.
- Database-backed knowledge storage.
- Automated publishing to social platforms.
- Multi-user collaboration.
- Full content performance analytics.

## 4. Directory Responsibilities

### `src/`

Contains the Python application code.

- `document_processor.py`: Loads, cleans, and normalizes source documents.
- `knowledge_base.py`: Reads and organizes primary and secondary knowledge files.
- `prompt_templates.py`: Loads prompt templates and prepares template variables.
- `content_pipeline.py`: Coordinates knowledge, templates, prompts, LLM calls, and output saving.
- `llm_integration.py`: Connects to the selected LLM provider.
- `main.py`: CLI entry point.

### `knowledge_base/primary/`

Stores core canon and brand material.

- `brand_voice.md`: Tone, style, vocabulary, and writing rules.
- `mortal_vengeance_series.md`: Series premise, lore, canon, and timeline.
- `characters.md`: Character profiles, relationships, motivations, and voice notes.
- `themes.md`: Major themes, motifs, emotional territory, and boundaries.
- `past_content.md`: Existing content examples and reusable patterns.

### `knowledge_base/secondary/`

Stores market and platform context.

- `genre_positioning.md`: Comparable genres, reader expectations, and positioning.
- `platform_best_practices.md`: Platform-specific format and engagement guidance.
- `competitor_analysis.md`: Comparable creators, campaigns, strengths, and gaps.

### `templates/`

Stores reusable content templates.

- `instagram_caption.md`
- `tiktok_post.md`
- `blog_post.md`
- `newsletter_blurb.md`
- `character_spotlight.md`
- `review_pull_quote.md`

### `outputs/`

Stores generated content drafts and exported files.

### `config/`

Stores local configuration files, including agent/editor configuration.

## 5. Functional Requirements

1. The system must load markdown files from the knowledge base.
2. The system must distinguish primary knowledge from secondary knowledge.
3. The system must load a selected content template from `templates/`.
4. The system must combine user input, relevant knowledge, and a template into a final prompt.
5. The system must send prompts through a single LLM integration interface.
6. The system must return generated content as plain text.
7. The system must save generated content to `outputs/`.
8. The system must support at least these content types:
   - Instagram caption
   - TikTok post
   - Blog post
   - Newsletter blurb
   - Character spotlight
   - Review pull quote

## 6. Non-Functional Requirements

- Code should be readable, modular, and easy to extend.
- Configuration should not require hardcoded API keys.
- Secrets must be stored in `.env` and excluded from version control.
- Markdown files should remain human-editable.
- Generated output should be traceable to a selected template and content type.
- The project should run locally from the command line.

## 7. Environment Requirements

- Python 3.11 or newer.
- Virtual environment recommended.
- Dependencies listed in `requirements.txt`.
- Environment variables loaded from `.env`.

Suggested environment variables:

```env
OPENAI_API_KEY=
LLM_MODEL=
OUTPUT_DIRECTORY=outputs
```

## 8. MVP Workflow

1. User selects a content type.
2. User provides a topic or brief.
3. System loads relevant knowledge base files.
4. System loads the matching template.
5. System builds a complete prompt.
6. System sends the prompt to the LLM provider.
7. System saves the generated draft in `outputs/`.
8. System prints the output path and a short completion message.

## 9. Acceptance Criteria

- Running `python src/main.py` starts the application without import errors.
- Knowledge base markdown files can be loaded from disk.
- Template markdown files can be loaded from disk.
- The pipeline can build a prompt from a content type and topic.
- LLM integration is isolated in `llm_integration.py`.
- Generated content can be saved to `outputs/`.
- Missing files produce clear, actionable errors.

## 10. Future Enhancements

- Add automatic content calendar generation.
- Add platform-specific validation rules.
- Add batch generation.
- Add content revision workflows.
- Add metadata files for generated outputs.
- Add tests for document loading, template formatting, and pipeline orchestration.
- Add optional web UI.
---

## 11. Knowledge Base Architecture

The project uses two knowledge layers.

### Primary Knowledge Base

The primary knowledge base contains brand-specific and story-specific material for the Mortal Vengeance series.

It includes:

- Brand voice and tone
- Series overview
- Character information
- Themes and motifs
- Existing promotional content
- Canon-specific terminology and positioning

This layer ensures that generated content feels specific to Mortal Vengeance instead of sounding like generic thriller marketing copy.

### Secondary Research Layer

The secondary research layer contains broader market, genre, and platform context.

It includes:

- Genre positioning
- Platform best practices
- Comparable titles or creators
- Audience expectations
- Social media content conventions

This layer helps the system adapt the same brand identity across different platforms and audience types.

---

## 12. Uniqueness Strategy

The system is designed to avoid generic AI-generated content by using:

- Brand-specific markdown knowledge files
- Platform-specific prompt templates
- Mortal Vengeance-specific tone and terminology
- Audience and persona targeting
- Comparison against generic ChatGPT-style output

The final project will include side-by-side examples showing:

1. A generic AI response generated from a simple prompt.
2. A Mythos Content Engine response generated using the knowledge base and custom prompt templates.

The comparison will explain how the Mythos output is more specific, more brand-aligned, and less generic.

---

## 13. Project Management and Kanban

The project will be managed through a Trello Kanban board named:

**ACFT0520 - Project 2 - Mythos Content Engine**

The board will include the following lists:

- Backlog
- Ready
- In Progress
- Review / Testing
- Done
- Scope Changes / Parking Lot

### Definition of Done

A task is considered done only when:

- The code or documentation is completed.
- The feature has been tested.
- Acceptance criteria are met.
- Any meaningful AI prompts used are logged.
- Related files are committed to GitHub.

### Required Screenshots

The project will include three full-board screenshots:

- Planning screenshot
- Midpoint screenshot
- Final screenshot

Suggested filenames:

- `p2-kanban-planning.png`
- `p2-kanban-midpoint.png`
- `p2-kanban-final.png`

---

## 14. Functional Requirements with IDs

### FR-001: Markdown Knowledge Base Loading

The system shall load markdown files from both the primary and secondary knowledge base folders.

Acceptance Criteria:

- The system can read `.md` files from `knowledge_base/primary/`.
- The system can read `.md` files from `knowledge_base/secondary/`.
- Missing folders or files produce clear error messages.

### FR-002: Knowledge Layer Separation

The system shall distinguish between brand-specific knowledge and secondary research context.

Acceptance Criteria:

- Primary knowledge and secondary knowledge are stored separately.
- The final prompt identifies both knowledge layers.
- Generated content reflects Mortal Vengeance-specific context.

### FR-003: Prompt Template Loading

The system shall load reusable prompt templates from the `templates/` folder.

Acceptance Criteria:

- Each supported content type has a corresponding template.
- Missing templates return a clear error.
- Templates can be edited without changing Python code.

### FR-004: Content Pipeline

The system shall combine user input, selected content type, knowledge base context, and prompt templates into a generation workflow.

Acceptance Criteria:

- The user can select a content type.
- The user can provide a topic or brief.
- The system builds a complete prompt.
- The system sends the prompt to the LLM layer.
- The system returns generated content.

### FR-005: Output Saving

The system shall save generated content into the `outputs/` folder.

Acceptance Criteria:

- Generated files are saved locally.
- Output filenames include the content type and timestamp.
- The system prints the saved file path.

### FR-006: LLM Integration

The system shall send prompts through a single LLM integration interface.

Acceptance Criteria:

- API logic is isolated in `llm_integration.py`.
- API keys are loaded from `.env`.
- The model can be changed through configuration.

### FR-007: Uniqueness Demonstration

The system shall provide evidence that its output is different from generic AI content.

Acceptance Criteria:

- At least one generic output is created for comparison.
- At least one Mythos-generated output is created using the knowledge base.
- The comparison explains differences in specificity, voice, platform fit, and brand alignment.

---

## 15. AI Prompt Tracking Log

This section tracks meaningful AI prompts that influenced the codebase, structure, debugging, or project decisions.

| Date | Prompt Summary | Purpose | Result / Decision |
|---|---|---|---|
| TBD | Asked for project concept selection | Defined MVP direction | Chose Mythos Content Engine using Mortal Vengeance as flagship case study |
| TBD | Asked how to structure project requirements | Documentation setup | Created root-level PROJECT_REQUIREMENTS.md |
| TBD | Asked how to organize project folders | Architecture planning | Adopted modular Python structure with src, knowledge_base, templates, outputs, and config |

---

## 16. Change Log

This section records scope or requirement changes.

| Date | Change | Reason | Impact |
|---|---|---|---|
| TBD | Project name changed from Mortal Vengeance Content Studio to Mythos Content Engine | Needed a scalable name beyond one book series | Mortal Vengeance becomes the MVP case study instead of the entire product identity |
| TBD | Vector database and full RAG moved out of MVP | Project brief does not require full RAG | Keeps scope realistic for two-week timeline |
| TBD | Image and video generation moved to future enhancements | Avoids overbuilding MVP | Focus remains on markdown ingestion, prompt templates, and LLM generation |
| TBD | Asked to integrate Alejandro Torres content playbook into templates | Improved strategic quality of outputs | Added content objective, audience intent, editorial framework, CTA, QA, and channel strategy rules into all prompt templates |