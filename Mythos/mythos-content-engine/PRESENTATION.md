# Tell Tales Ink — Presentation Pack

> **Project:** Tell Tales Ink — an AI content engine for book marketing
> **Context:** Ironhack master's final project · 12 weeks · production-grade MVP (running web app, ~13,500 LOC)
> **Built for:** the *Mortal Vengeance* trilogy (Dominican dark-academia / horror–thriller)
>
> Every claim in this document was fact-checked against the actual codebase. See the
> **Accuracy notes** at the end for what was corrected from an earlier generic draft.

---

## One-liner
Tell Tales Ink is a **prompt-engineering system** that turns a few UI selections into a
layered, multi-stage LLM pipeline — producing on-brand, platform-native book-marketing
content across every channel, grounded in the actual novels, and benchmarked head-to-head
against raw ChatGPT to prove the prompting (not just the model) is what wins.

## The problem
Generic AI output lacks **domain context** and **brand voice**. Raw ChatGPT is fluent but
off-brand, and prone to inventing facts (reviews, awards, quotes). The hard part isn't
*calling* an LLM — it's engineering the prompts: turning messy intent into structured
instructions, enforcing voice and format per platform, preventing hallucination, and
keeping output publication-ready.

## The goal
One workspace that generates publication-ready, on-brand marketing content for every
channel — grounded in a private knowledge base **and the full text of the novels** —
then produces the finished assets (text, images, podcast audio), and proves it beats a
raw-ChatGPT baseline with a built-in side-by-side comparison.

---

## ⭐ The core: prompt engineering & LLM orchestration
This is the headline. The app is a **multi-stage prompting pipeline**, not a single chat call:

1. **Structured-brief assembly** — the UI's rich selections (content type, platform,
   audience, formats, hooks, CTA, tone, constraints, book, characters, moods…) are
   deterministically compiled into a precise structured brief.
2. **Stage-1 prompt — context filtering / relevance re-ranking** — a purpose-built prompt
   has the LLM read the query against retrieved candidates and keep only the relevant
   context (and flag spoiler/accuracy constraints). A prompt whose job is to shape the next prompt.
3. **Stage-2 prompt — templated generation** — each of **19 content types has its own
   engineered prompt template** (role, strategy rules, requirements, output format, and a
   built-in *Quality Check* the model must satisfy before returning), filled with the
   filtered context + structured brief.
4. **Parameterized prompt families** — the Chapter Promos prompts are genre-aware (6 tones)
   and pillar-driven (5 selectable teaser structures), composed at runtime from a system
   preamble + pillar guidance + per-platform structure.
5. **Anti-hallucination prompt rules** — every template hard-codes "use only verified
   facts / exact quotes / no invented reviews or awards," plus spoiler controls.
6. **Evaluation prompt** — a deliberately fair, KB-free ChatGPT-baseline prompt is built
   from the same brief so the two systems can be judged head-to-head.

**Takeaway:** the intelligence lives in *how the prompts are constructed, layered, and
constrained*. The model is the engine; the prompt system is the design.

---

## How it works (the pipeline)
```
UI selections
  → structured brief (deterministic assembly)
  → Stage-1: hybrid retrieve + LLM relevance re-ranking/distillation
  → Stage-2: per-content-type template prompt → draft
  → media (Pillow images / ElevenLabs audio) → save / export
```

### Stages (literal)
**Load → Retrieve → Generate → Export.** (Knowledge is loaded from markdown at runtime
with an on-disk embedding cache; "export" = save draft / `.docx` / download.)

### Two-tier knowledge base (source authority)
- **Primary = authoritative** — `knowledge_base/primary/` (brand voice, content playbook,
  canon / manuscript index, quote bank, characters, themes). **Ranking weight 0.70.**
- **Secondary = reference** — `knowledge_base/secondary/` (competitor analysis, market /
  genre research, platform best-practices, real reviews). **Ranking weight 0.30.**
- Two specialist layers also exist: `publishing/` and `private_manuscripts/`.

### Knowledge hierarchy (3 levels)
**Layer → Document → Chunk.** Chunking is structure-aware (splits on markdown sections
first, then by size) so retrieved snippets stay coherent.

### Retrieval (technical detail)
- **Embedding:** OpenAI `text-embedding-3-small`, L2-normalized, cached on disk (the
  corpus embeds once).
- **Two-stage retrieval:**
  1. **Hybrid pre-ranking** — keyword overlap + semantic cosine, multiplied by source
     authority (primary 0.70 / secondary 0.30), with mood / character / book boosts.
  2. **LLM relevance re-ranking + distillation** — the context-filter stage reads the
     query against the candidates and keeps only what's relevant.
  *(Note: this is LLM-based re-ranking, not a cross-encoder model.)*
- **Generation guardrails:** verified-context-only, exact-quote rules, spoiler controls.

---

## The four strategies behind the ChatGPT comparison
- **Source attribution** — review excerpts keep their source names.
- **Brand-voice prompting** — `brand_voice.md` + per-template voice rules.
- **Contextual RAG** — manuscript- and KB-grounded retrieval feeds every prompt.
- **Hallucination guardrails** — "Do not invent facts/quotes/reviews/awards" enforced in
  templates + spoiler controls.

---

## Core features
- **~19 content types** — Instagram, LinkedIn, YouTube, Blog, Newsletter, Press Release,
  Quote Post, Review Pull-Quote, Character Spotlight, Podcast + publishing deliverables.
- **Campaign Mode** — one brief → many coordinated, distinct-angle assets + auto posting calendar.
- **Chapter Promos** — reads a real chapter and produces a SWBST summary or genre-aware
  platform teasers (selectable teaser "pillars", per-genre tone).
- **Image generation** — branded quote-card graphics in **26 formats × 14 themes** (every
  social aspect ratio + carousels), plus AI images.
- **Podcast Studio** — generates a production-ready multi-speaker script and renders real
  audio via ElevenLabs.
- **Gallery + Library** — visual browser for every generated image; unified browser for all
  documents + podcast audio (inline players).
- **Compare vs ChatGPT** — side-by-side baseline + human vote across Generator, Campaign,
  Chapter Promos, and Podcast.
- Draft save/reopen, `.docx` export, brand-safety + spoiler controls throughout.

## Tech stack
- **Python + NiceGUI** (single web app)
- **OpenAI** — LLM generation (`gpt-5.4-mini`, configurable) + `text-embedding-3-small` for retrieval
- **ElevenLabs** — multi-voice TTS (`eleven_multilingual_v2` / `eleven_v3`)
- **Pillow** (images), **NumPy** (similarity search + on-disk embedding cache), **python-docx**

## By the numbers (verified)
- **19** engineered prompt templates · a **2-stage** LLM pipeline per generation
- **3 novels → 68 chapters** (12 / 26 / 30); ~2.8 MB of manuscript text → **2,317 cached passage vectors**
- **31** knowledge-base files · **30** characters · **26** image formats × **14** themes
- **1,589** generated images and **6** rendered podcast episodes already produced
- ~**13,500** lines of Python across **21** modules

---

## "What broke & how I fixed it" (real debugging stories)

**① RAG was blind to the actual books** *(the flagship story)*
- **Broke:** semantic retrieval only searched the curated *quote bank*, not the novels.
  Generated content could only reuse pre-picked quotes; the full manuscripts were never
  searchable, and a newly added book had only placeholder quotes.
- **Fixed:** built a manuscript-embedding layer — chunks the full novels, embeds them once
  with `text-embedding-3-small`, caches the vectors (2,317 chunks), and blends cosine
  similarity into retrieval. *Proof:* the query "a character overwhelmed by guilt and fear
  of abandonment" (almost no shared words with the prose) now retrieves the exact passage
  "Mario felt like the world was crashing down… Guilt, like a giant…".

**② Campaign podcast had voice settings but produced no audio**
- **Broke:** the handler used `asyncio.create_task(...)`, which detached the coroutine —
  errors were swallowed and the UI context lost, so nothing happened and nothing reported why.
- **Fixed:** returned the coroutine so NiceGUI awaits it in-context with proper error
  surfacing; confirmed with an ElevenLabs smoke test.

**③ Saved drafts silently lost form fields on reopen**
- **Broke:** the snapshot registry used wrong/incomplete widget names (e.g. `newsletter_*`
  vs `nl_*`), resetting fields across 6 content types.
- **Fixed:** rebuilt the field registry to all real widgets, ordered for value-change
  cascades, and hardened the snapshot against crashes. Every generation field now round-trips.

---

## Suggested live demo (≈4–5 min)
1. **Generate one post** → on-brand result.
2. **Compare vs ChatGPT** → run the baseline, show side-by-side, vote — strongest proof the
   *prompting* (not the model) wins.
3. **Chapter Promos** → switch Genre/Pillar on the same chapter → different controlled
   outputs (prompt parameterization made visible).
4. **Campaign Mode** → one brief → multi-asset bundle + calendar.
5. **Podcast Studio** → script → short audio preview.
6. **Gallery / Library** → browse everything just produced.

## Challenges & learnings
- Designing prompts that stay on-brand *and* platform-native *and* non-hallucinating took the
  two-stage split + strict in-prompt rules.
- Turning freeform intent into a reliable structured brief was as important as the generation prompt.
- Parameterizing prompt families (genre × pillar × platform) without prompt sprawl.
- Real grounding required semantic retrieval over the full manuscripts, not a quote list.

## Future work
- Prompt-eval dashboard (win-rate vs ChatGPT over time), prompt versioning / A-B testing,
  direct publishing APIs (Meta/LinkedIn/Buffer), multi-book / multi-author support,
  optional cross-encoder re-ranking.

---

## Accuracy notes (corrected from an earlier generic-RAG draft)
An earlier draft deck contained generic-template claims that did **not** match this code.
Corrected here so nothing on a slide can be contradicted by the repo:

| Earlier (generic) claim | Status | Accurate version |
|---|---|---|
| Primary/secondary weights 0.70 / 0.30 | ✅ now **true** | Implemented in `context_filter.py` (`layer_weight`); primary ranks above secondary. |
| Cross-encoder re-ranking | ❌ not implemented | It's **LLM-based relevance re-ranking** (the context-filter stage) + hybrid pre-ranking. No cross-encoder model. |
| Domain → Collection → Document → Chunk | ❌ wrong names | Real hierarchy is **Layer → Document → Chunk** (3 levels). |
| Six supported output formats | ❌ wrong count | **19** content templates (+ 26 image formats × 14 themes). |
| Stages: Ingest / Deliver | ⚠️ reworded | **Load → Retrieve → Generate → Export.** |
| "Intent analysis" / "self-review" steps | ⚠️ reworded | Structured-brief assembly + an **in-prompt Quality Check** (not separate LLM passes). |
| Two-tier knowledge base (primary/secondary) | ✅ true | Plus two specialist layers (`publishing/`, `private_manuscripts/`). |
| Hybrid retrieval; 4 grounding strategies | ✅ true | Keyword + semantic blend; source attribution, brand-voice, contextual RAG, hallucination guardrails. |
