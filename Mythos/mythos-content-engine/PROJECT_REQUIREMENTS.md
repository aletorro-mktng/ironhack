# Project Requirements

## 1. Project Summary

- Project name: Mythos Content Engine
- Team members: Alejandro Torres De La Rocha
- Problem we are solving: Authors and story brands need a way to create platform-specific content without losing canon accuracy, brand voice, real-review grounding, or human editorial control.
- Target user or client: Authors, book marketers, and story-driven brands promoting the Mortal Vengeance universe and related Mythos content.
- Success criteria:
  - The app loads markdown documents from primary and secondary knowledge bases.
  - The app generates brand-aligned drafts through an LLM integration.
  - The app supports content-specific workflows for podcast, Instagram, YouTube, LinkedIn, press release, blog, newsletter, quote post, pull quote, character spotlight, and campaign mode.
  - The app saves prompts, filtered context, drafts, exports, audio, quote graphics, revisions, approvals, and calendars to `outputs/`.
  - The app demonstrates content that is more specific and less generic than a basic ChatGPT response.
  - Human review, revision, and approval are part of the workflow.

## 2. Scope

### Must Have

- [x] Markdown ingestion from `knowledge_base/primary/` and `knowledge_base/secondary/`.
- [x] LLM API integration through `src/llm_integration.py`.
- [x] Structured Python automation framework under `src/`.
- [x] Reusable prompt templates under `templates/`.
- [x] Local NiceGUI UI through `src/nicegui_ui.py`, with Podcast Studio migrating first to React + TypeScript + Vite behind FastAPI.
- [x] Primary and secondary knowledge base separation.
- [x] Output saving for prompts, filtered context, and generated drafts.
- [x] Human-in-the-loop revision and approval workflow.
- [x] Prompt tracking log and change log in this file.
- [x] VSCode agent configuration in `config/vscode_agent.json` and documentation in `README.md`.
- [x] Quote and pull-quote workflows constrained to approved or manuscript-grounded quotes.
- [x] Press release workflow constrained to verified claims, awards, reviews, and user-supplied facts.
- [x] Campaign Mode with configurable selected content types.
- [x] Content calendar generation.

### Should Have

- [x] ElevenLabs podcast audio rendering when API credentials are configured.
- [x] Podcast setup with episode title, episode number, voices, intro music, outro music, sound effects, and delivery cues.
- [x] Platform-like results previews for Instagram, LinkedIn, and YouTube.
- [x] Press release newsroom-style results panel with media-readiness checks.
- [x] Image upload support for Instagram, LinkedIn, YouTube, and Blog workflows.
- [x] SEO and accessibility metadata generation for uploaded images.
- [x] Audience-aware hashtag and messaging guidance.
- [x] Quote graphics using Milonet for quote text and Unthinkers for attribution.

### Could Have

- [ ] Full vector-store RAG with embeddings.
- [ ] Direct publishing to Instagram, LinkedIn, YouTube, email, or PR distribution services.
- [ ] Multi-user authentication.
- [ ] Cloud-hosted database for drafts and calendars.
- [ ] Advanced performance analytics.
- [ ] Professional podcast mastering and distribution.

### Out of Scope

- Public deployment for MVP.
- Payment or subscription features.
- Automated social publishing.
- Multi-user collaboration.
- Replacing human review with fully autonomous publishing.
- Storing live API keys in source control.

## 3. Functional Requirements

| ID | Requirement | Acceptance Criteria | Trello Card |
|----|-------------|---------------------|-------------|
| FR-001 | The app can ingest markdown files from both knowledge bases. | Given valid `.md` files, when the app runs, then parsed primary and secondary content is available for prompts. | KBN-001 |
| FR-002 | The app keeps primary and secondary knowledge separate. | Given both knowledge folders, when context is loaded, then brand/canon context and market/platform context are labeled separately. | KBN-002 |
| FR-003 | The app can generate brand-aligned content using LLM prompts. | Given parsed context and a user brief, when generation runs, then the output reflects Mortal Vengeance context, brand voice, and selected audience. | KBN-003 |
| FR-004 | The app uses reusable prompt templates. | Given a selected content type, when generation runs, then the matching template or style variant is applied. | KBN-004 |
| FR-005 | The app saves traceable artifacts. | Given a generated draft, when the workflow completes, then draft, prompt, and filtered context files are saved in `outputs/`. | KBN-005 |
| FR-006 | The local UI supports structured content creation. | Given the legacy UI or the React/FastAPI Podcast Studio, when the app opens, then users can enter briefs, configure content, generate drafts, edit, revise, approve, and save. | KBN-006 |
| FR-007 | Campaign Mode supports configurable multi-content campaigns. | Given multiple selected content types, when Campaign Mode opens, then each selected type exposes settings for format, quantity, style, audience, and constraints. | KBN-007 |
| FR-008 | The app generates content calendars. | Given campaign dates, cadence, channels, content types, audience, and CTA, when calendar generation runs, then an editable calendar artifact is saved. | KBN-008 |
| FR-009 | The app supports human revision workflow. | Given draft feedback, when revision runs, then a revision candidate is created, reviewed, accepted if approved, and saved as a traceable artifact. | KBN-009 |
| FR-010 | The app logs feedback and approvals. | Given a reviewed draft, when the user records feedback or approval, then a log entry is saved. | KBN-010 |
| FR-011 | Podcast scripts can render audio through ElevenLabs. | Given valid ElevenLabs credentials and selected voices, when audio rendering runs, then segment files and/or a full podcast file are created or a clear error is shown. | KBN-011 |
| FR-012 | Podcast setup includes production controls. | Given podcast content type, when setup opens, then users can configure episode title, episode number, voices, intro/outro music, sound effects, delivery cues, format, and length. | KBN-012 |
| FR-013 | Production tags are handled safely for audio. | Given bracketed delivery cues or production notes, when text is sent to ElevenLabs, then non-spoken tags are stripped where appropriate. | KBN-013 |
| FR-014 | Instagram workflow supports only Instagram destinations. | Given Instagram content type, when platform setup opens, then only Instagram appears and users can select feed post, reel, story, and/or carousel. | KBN-014 |
| FR-015 | LinkedIn and YouTube results resemble their platforms. | Given generated LinkedIn or YouTube content, when results render, then the preview follows the platform's familiar post/title/description structure. | KBN-015 |
| FR-016 | Press release results behave like a media workspace. | Given press release generation, when results render, then the panel shows document preview, media-readiness status, checklist, copy/export actions, and PR-specific details. | KBN-016 |
| FR-017 | Image upload workflows generate SEO and platform metadata. | Given an uploaded `.png`, `.jpg`, `.jpeg`, or `.webp`, when generation runs, then the app returns SEO title, meta title, image description, meta description, social copy, blog caption, and alt title where applicable. | KBN-017 |
| FR-018 | Character images are identified or confirmed. | Given an uploaded character image, when the filename or assets identify a character, then the app uses that character; otherwise it asks for clarification. | KBN-018 |
| FR-019 | Quote selection is grounded. | Given a quote or character request, when quote options are generated, then the app returns at least five matching quotes when enough source material exists and does not invent quotes. | KBN-019 |
| FR-020 | Factual claims are grounded. | Given awards, reviews, quotes, or press facts, when generation runs, then claims must come from the knowledge base or explicit user input. | KBN-020 |
| FR-021 | The app demonstrates uniqueness versus a fresh ChatGPT baseline. | Given a Mythos vs ChatGPT comparison, when the uniqueness script or saved artifact is reviewed by the human judge, then differences in specificity, voice, context, and platform fit are documented. | KBN-021 |
| FR-022 | The repository documents the VSCode agent setup. | Given the README and `config/vscode_agent.json`, when reviewed, then the agent workflow, entrypoint, output folder, and review expectations are clear. | KBN-022 |

## 4. Non-Functional Requirements

- Reliability: The app should fail with clear, actionable errors when files, credentials, voices, assets, or templates are missing.
- Privacy and API-key handling: API keys must live in `.env` and must not be committed. Generated outputs may contain private drafts and should remain local unless intentionally shared.
- Maintainability: Code should stay modular, readable, and organized by responsibility. Prompt templates and markdown knowledge files should remain editable without changing core Python code.
- Usability: The UI should be content-specific, avoid irrelevant global actions, and let the user configure important choices before generation.
- Traceability: Generated drafts should be connected to content type, timestamp, prompt, filtered context, review notes, and revision artifacts.
- Accuracy: Quotes, reviews, awards, and factual claims must be grounded in approved project context or explicit user input.

## 5. Kanban / Project Management

- Board name: ACFT0520 - Project 2 - Mythos Content Engine
- Board link, if shareable: Not added yet
- Workflow columns:
  - Backlog
  - Ready
  - In Progress
  - Review / Testing
  - Done
  - Scope Changes / Parking Lot
- WIP limit: Maximum 3 active implementation tasks at a time
- Definition of Done:
  - Code or documentation is complete.
  - The related acceptance criteria are met.
  - The local workflow runs without import errors.
  - Generated artifacts save to the expected folder when relevant.
  - Errors are understandable when required inputs are missing.
  - Human review has checked AI-generated code or content.
  - Meaningful AI-agent prompts are logged in this file.
  - Scope changes are recorded in the change log.
- Review cadence: Daily review during active development, plus a final review before GitHub submission.

## 6. AI Coding-Agent Rules

- Which AI coding agent(s) we used: Codex / ChatGPT coding agent in the local workspace.
- What the agent is allowed to do:
  - Inspect repository files.
  - Propose and implement scoped code changes.
  - Edit documentation.
  - Add or update prompt templates.
  - Run local checks, compile checks, and UI smoke tests.
  - Help debug errors and improve user workflows.
- What the agent is not allowed to do:
  - Commit or push code without explicit human approval.
  - Publish generated content to external platforms.
  - Expose, print, or commit API keys or private secrets.
  - Revert unrelated user changes.
  - Replace human approval for final content decisions.
- How humans review AI-generated code:
  - Read changed files before committing.
  - Run the app locally when UI or generation behavior changes.
  - Check generated outputs for brand voice, factual accuracy, quote grounding, and platform fit.
  - Adjust prompts, templates, and UI copy when output feels generic or misleading.
- How we protect secrets and private data:
  - Store credentials in `.env`.
  - Keep `.env` out of version control.
  - Avoid pasting API keys into prompts or documentation.
  - Treat manuscripts, drafts, reviews, and generated campaign materials as private local project data.

## 7. Prompt Tracking Log

| Date | Tool / Agent | Prompt Goal | Prompt Summary | Output Used? | Human Review Notes | Related Commit / PR |
|------|--------------|-------------|----------------|--------------|--------------------|---------------------|
| 2026-06-04 | Codex / ChatGPT | Plan project structure | Asked agent to structure the requirements, folders, knowledge bases, templates, outputs, and config. | Yes | Human kept the project focused on markdown, prompt templates, and LLM generation. | Pending |
| 2026-06-04 | Codex / ChatGPT | Refactor generation pipeline | Asked agent to separate context filtering from final generation to reduce prompt confusion. | Yes | Human reviewed pipeline behavior and output artifacts. | Pending |
| 2026-06-04 | Codex / ChatGPT | Improve CLI selections | Asked agent to prevent duplicate custom-input prompts and improve structured multi-select behavior. | Yes | Human tested CLI flow and requested follow-up UX fixes. | Pending |
| 2026-06-04 | Codex / ChatGPT | Add quote guardrails | Asked agent to prevent quote posts from selecting unavailable books or invented quotes. | Yes | Human reviewed against `quote_bank.md`. | Pending |
| 2026-06-04 | Codex / ChatGPT | Add podcast workflow | Asked agent to add podcast template, script structure, voices, production tags, and ElevenLabs helper path. | Yes | Human later requested better podcast usability and audio rendering fixes. | Pending |
| 2026-06-04 | Codex / ChatGPT | Build local UI | Asked agent to add a Gradio UI for structured content creation, output review, editing, and saving. | Yes | Human reviewed screenshots and requested several content-specific redesigns. | Pending |
| 2026-06-05 | Codex / ChatGPT | Redesign dashboard | Asked agent to make the main screen match the provided Mythos visual reference. | Yes | Human supplied image references and requested multiple polish passes. | Pending |
| 2026-06-05 | Codex / ChatGPT | Redesign press release page | Asked agent to replace generic results with a newsroom/media-kit workspace. | Yes | Human corrected podcast-specific actions appearing on press release results. | Pending |
| 2026-06-06 | Codex / ChatGPT | Improve podcast usability | Asked agent to add voice selection, episode title, episode number, intro/outro music, and sound effects. | Yes | Human later flagged missing voice/music/SFX loading behavior. | Pending |
| 2026-06-06 | Codex / ChatGPT | Reduce generic AI content | Asked agent to add style variants per content type and feedback/approval logging. | Yes | Human reviewed against Hivemind anti-generic-content criteria. | Pending |
| 2026-06-08 | Codex / ChatGPT | Fix Instagram setup | Asked agent to show only Instagram, support multi-select post types, and remove irrelevant platform options. | Yes | Human reviewed UI screenshots and requested result preview improvements. | Pending |
| 2026-06-08 | Codex / ChatGPT | Add YouTube and LinkedIn | Asked agent to remove TikTok and add YouTube and LinkedIn with configurable content subtypes. | Yes | Human requested platform-like result previews. | Pending |
| 2026-06-08 | Codex / ChatGPT | Add image upload SEO workflow | Asked agent to support image uploads and generate SEO titles, descriptions, captions, meta fields, and alt titles. | Yes | Human supplied character images and requested filename-based character recognition. | Pending |
| 2026-06-08 | Codex / ChatGPT | Improve quote graphics | Asked agent to use Milonet and Unthinkers fonts and character-name assets for quote and pull-quote posts. | Yes | Human verified font and character requirements. | Pending |
| 2026-06-08 | Codex / ChatGPT | Improve quote matching | Asked agent to use quote bank and manuscript-backed retrieval so character quote requests return correct options. | Partly | Human required at least five grounded quote options when source material allows. | Pending |
| 2026-06-08 | Codex / ChatGPT | Configure Campaign Mode | Asked agent to make every selected campaign content type open its own configuration panel. | Yes | Human rejected random/default campaign asset generation. | Pending |
| 2026-06-09 | Codex / ChatGPT | Reclassify real reviews | Asked agent to move `real_reviews.md` from primary to secondary knowledge base. | Yes | Human identified reviews as external social proof rather than canon. | Pending |
| 2026-06-09 | Codex / ChatGPT | Add revision workflow | Asked agent to add revision candidate, accept-into-editor, and revision artifacts. | Yes | Human requested human-in-the-loop review evidence. | Pending |
| 2026-06-09 | Codex / ChatGPT | Add content calendar | Asked agent to add content calendar generation for campaigns. | Yes | Human requested campaign planning support. | Pending |
| 2026-06-09 | Codex / ChatGPT | Remove generic result actions | Asked agent to remove Generated Draft, Regenerate, Shorten, Punchier, and Add CTA generic controls everywhere. | Yes | Human identified generic controls as inappropriate across content types. | Pending |
| 2026-06-09 | Codex / ChatGPT | Update requirements file | Asked agent to align `PROJECT_REQUIREMENTS.md` with required rubric sections, Kanban mapping, prompt log, and change log. | Yes | Human supplied required structure and submission expectations. | Pending |
| 2026-06-10 | Codex / ChatGPT | Switch main UI to NiceGUI | Asked agent to replace the primary Gradio shell with a responsive NiceGUI app while keeping the podcast studio available through Gradio. | Yes | Human requested NiceGUI after comparing layout and coding ergonomics. | Pending |
| 2026-06-10 | Codex / ChatGPT | Integrate podcast natively | Asked agent to replace the Gradio podcast iframe with a native NiceGUI podcast form and direct generation flow. | Yes | Human requested the podcast workflow live inside the new shell. | Pending |
| 2026-06-10 | Codex / ChatGPT | Add native content panels | Asked agent to add dedicated NiceGUI panels for blog posts, newsletters, and character spotlights in the main generator. | Yes | Human wanted the rest of the content types moved toward content-specific layouts. | Pending |
| 2026-06-10 | Codex / ChatGPT | Add press release panel | Asked agent to add a newsroom-style press release panel with dateline, contact, evidence, and media-target fields. | Yes | Human wanted press release to be treated like a newsroom workflow. | Pending |
| 2026-06-10 | Codex / ChatGPT | Add platform previews | Asked agent to add native Instagram, LinkedIn, and YouTube preview cards in the NiceGUI generator. | Yes | Human wanted the platform-specific layouts to feel more like real channel previews. | Pending |

## 8. Change Log

| Date | Requirement / Decision Changed | Why It Changed | Approved By |
|------|--------------------------------|----------------|-------------|
| 2026-06-04 | Project name standardized as Mythos Content Engine. | Needed a scalable product name beyond one book title. | Alejandro Torres De La Rocha |
| 2026-06-04 | Full vector database moved to could-have scope. | MVP requirements allow markdown and prompt-based RAG ideas without requiring embeddings. | Alejandro Torres De La Rocha |
| 2026-06-04 | Press Release added as a core content type. | The project needed media/PR workflow and verified-claim handling. | Alejandro Torres De La Rocha |
| 2026-06-04 | Podcast added as a core content type. | The project needed scripts, show notes, voice planning, and eventual audio rendering. | Alejandro Torres De La Rocha |
| 2026-06-05 | Gradio UI became the primary user experience. | CLI alone was not user-friendly enough for structured content creation and review. | Alejandro Torres De La Rocha |
| 2026-06-10 | NiceGUI became the primary app shell. | The project needed a more responsive, easier-to-layout interface with podcast generation inside the native UI. | Alejandro Torres De La Rocha |
| 2026-06-10 | Blog posts, newsletters, and character spotlights received native generator panels. | The first phase of the content-specific NiceGUI migration needed structured inputs for the easier non-podcast workflows. | Alejandro Torres De La Rocha |
| 2026-06-10 | Press releases received a native newsroom-style panel. | The press-release workflow needed dateline, timing, contact, and supporting-proof fields in the main app. | Alejandro Torres De La Rocha |
| 2026-06-10 | Instagram, LinkedIn, and YouTube received native preview cards. | The app needed platform-specific presentation for the social and video outputs. | Alejandro Torres De La Rocha |
| 2026-06-05 | Results panels became content-specific. | Generic draft boxes made different content types feel like one form wearing different hats. | Alejandro Torres De La Rocha |
| 2026-06-06 | Anti-generic content strategy became explicit. | Project needed evidence against a basic ChatGPT baseline. | Alejandro Torres De La Rocha |
| 2026-06-08 | TikTok removed from MVP and replaced with YouTube and LinkedIn. | User requested new content types and platform relevance. | Alejandro Torres De La Rocha |
| 2026-06-08 | Campaign Mode changed from random bundle to configurable bundle. | Users need to choose format, quantity, style, audience, and constraints per selected asset. | Alejandro Torres De La Rocha |
| 2026-06-08 | Image upload and visual analysis added. | The project needed SEO, social, blog, and accessibility metadata from user-provided images. | Alejandro Torres De La Rocha |
| 2026-06-08 | Quote matching expanded beyond fixed categories. | Character quote matching needed better grounding and more options. | Alejandro Torres De La Rocha |
| 2026-06-09 | Real reviews moved to secondary knowledge base. | Reviews are external social proof, not primary canon. | Alejandro Torres De La Rocha |
| 2026-06-09 | Revision and approval workflows added. | Human-in-the-loop review is required to avoid generic or unreviewed AI content. | Alejandro Torres De La Rocha |
| 2026-06-09 | Content calendar generation added. | Campaign planning needs scheduleable outputs. | Alejandro Torres De La Rocha |
| 2026-06-09 | Generic global result actions removed. | Regenerate, Shorten, Punchier, and Add CTA were not appropriate for every content type. | Alejandro Torres De La Rocha |

## 9. Final Deliverables

| Deliverable | What It Must Explain or Demonstrate | Evidence / Artifact |
|-------------|-------------------------------------|---------------------|
| Knowledge base architecture explanation | Explain the difference between the primary knowledge base and secondary research layer, why each file belongs where it does, and how the app uses markdown context in prompts. | `knowledge_base/primary/`, `knowledge_base/secondary/`, README notes, and this requirements file |
| Pipeline demonstration | Show the workflow from document ingestion to context filtering, brief creation, draft generation, review, export/save, revision, and iteration. | `src/content_pipeline.py`, `src/context_filter.py`, `src/ui.py`, generated files in `outputs/` |
| Uniqueness demonstration | Compare a fresh ChatGPT baseline response against a Mythos-generated response and let the human judge explain how knowledge base context, brand voice, audience settings, and templates improve specificity. | `src/uniqueness_comparison.py` and saved comparison artifacts in `outputs/` |
| Technical implementation highlights | Highlight the modular Python architecture, LLM provider abstraction, template system, Gradio UI, ElevenLabs integration, quote graphics, image upload workflow, and campaign calendar generation. | `src/`, `templates/`, `assets/`, `config/vscode_agent.json`, README |
| Challenges faced and solutions | Document the main issues encountered, including generic result panels, incorrect quote matching, podcast audio configuration, Instagram platform confusion, campaign randomness, and scope changes. Explain the implemented or planned solution for each. | Change log, prompt tracking log, git history, UI behavior, and generated test artifacts |
