"""Create a saved Mythos vs ChatGPT uniqueness comparison."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

from content_pipeline import run_pipeline
from llm_integration import generate_text
from prompt_templates import list_supported_content_types


OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CONTENT_TYPE = "instagram_caption"
DEFAULT_TOPIC = (
    "Create an Instagram caption announcing Mortal Vengeance winning the "
    "IndieReader Discovery Award for Best Young Adult Horror."
)

CHATGPT_BASELINE_INSTRUCTIONS = {
    "instagram_caption": (
        "Write an Instagram caption with a hook, short body copy, a CTA, and "
        "hashtags."
    ),
    "youtube_content": (
        "Write YouTube title ideas, a video description, and talking points."
    ),
    "linkedin_content": (
        "Write a professional LinkedIn post or article summary with a clear CTA."
    ),
    "blog_post": (
        "Write a useful blog post with a headline, intro, body sections, and conclusion."
    ),
    "newsletter_blurb": (
        "Write an email newsletter blurb with a subject line, preview text, body, and CTA."
    ),
    "character_spotlight": (
        "Write a character spotlight with a hook, profile, conflict, and audience CTA."
    ),
    "review_pull_quote": (
        "Select or write promotional review pull quotes and short supporting copy."
    ),
    "quote_post": (
        "Write a quote post caption with a short quote, attribution, and social CTA."
    ),
    "podcast": (
        "Write a podcast episode script with an intro, segments, transitions, and outro."
    ),
    "press_release": (
        "Write a press release with headline, subheadline, dateline, body, boilerplate, "
        "and media contact."
    ),
}

CONTENT_TYPE_LABELS = {
    "instagram_caption": "Instagram caption",
    "youtube_content": "YouTube content package",
    "linkedin_content": "LinkedIn content package",
    "blog_post": "blog post",
    "newsletter_blurb": "newsletter blurb",
    "character_spotlight": "character spotlight",
    "review_pull_quote": "review pull quote",
    "quote_post": "quote post",
    "podcast": "podcast script",
    "press_release": "press release",
}

BRAND_SIGNAL_TERMS = [
    "Mortal Vengeance",
    "Alejandro Torres De La Rocha",
    "Dominican",
    "Caribbean Gothic",
    "Santo Domingo",
    "Excelsior",
    "IndieReader",
    "Best Young Adult Horror",
    "brand voice",
    "quote bank",
    "review",
    "reader",
]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_text(content: str, stem: str, output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}_{timestamp()}.md"
    path.write_text(content, encoding="utf-8")
    return path


def read_topic(args: argparse.Namespace) -> str:
    if args.topic_file:
        return Path(args.topic_file).read_text(encoding="utf-8").strip()

    return args.topic.strip()


def build_chatgpt_prompt(content_type: str, topic: str, custom_prompt: str | None = None) -> str:
    if custom_prompt:
        return custom_prompt.format(topic=topic, content_type=content_type)

    instruction = CHATGPT_BASELINE_INSTRUCTIONS.get(
        content_type,
        "Write polished marketing content for the requested content type.",
    )
    readable_type = CONTENT_TYPE_LABELS.get(content_type, content_type.replace("_", " "))
    article = "an" if readable_type[:1].lower() in {"a", "e", "i", "o", "u"} else "a"

    return f"""You are ChatGPT responding in a fresh chat.

Create {article} {readable_type} for the topic below.

Instructions:
- {instruction}
- Make it clear, polished, and ready to use.
- Use only the topic supplied by the user and general writing knowledge.
- Do not use the Mythos Content Engine, private knowledge bases, brand documents,
  manuscript details, review bank, quote bank, or project-specific context.
- If details are missing, stay general instead of inventing specifics.

Topic:
{topic}
"""


def count_brand_signals(text: str) -> list[tuple[str, int]]:
    lowered = text.lower()
    counts = []

    for term in BRAND_SIGNAL_TERMS:
        count = lowered.count(term.lower())
        if count:
            counts.append((term, count))

    return counts


def render_signal_scan(chatgpt_output: str, mythos_output: str) -> str:
    chatgpt_counts = count_brand_signals(chatgpt_output)
    mythos_counts = count_brand_signals(mythos_output)

    def render_counts(counts: list[tuple[str, int]]) -> str:
        if not counts:
            return "- No tracked brand/context signals found."

        return "\n".join(f"- {term}: {count}" for term, count in counts)

    return f"""## Automated Signal Scan

This scan is intentionally simple. It counts project-specific terms that often
show up when the knowledge bases are shaping the output.

### ChatGPT Baseline Signals

{render_counts(chatgpt_counts)}

### Mythos Output Signals

{render_counts(mythos_counts)}
"""


def render_report(
    *,
    content_type: str,
    topic: str,
    chatgpt_prompt: str,
    chatgpt_output: str,
    mythos_result: dict,
    chatgpt_prompt_path: Path,
    chatgpt_output_path: Path,
) -> str:
    mythos_output = mythos_result["generated_content"]

    def display_path(path: Path | str) -> Path:
        path = Path(path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    return f"""# Mythos vs ChatGPT Uniqueness Comparison

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Scenario

- Content type: `{content_type}`
- Topic: {topic}

## Artifact Paths

- ChatGPT baseline prompt: `{display_path(chatgpt_prompt_path)}`
- ChatGPT baseline output: `{display_path(chatgpt_output_path)}`
- Mythos filtered context: `{display_path(mythos_result["filtered_context_path"])}`
- Mythos generation prompt: `{display_path(mythos_result["prompt_path"])}`
- Mythos draft: `{display_path(mythos_result["draft_path"])}`

## What This Demonstrates

The ChatGPT baseline is produced from a fresh-chat prompt with no project
documents, no Mythos knowledge base, and no reusable Mythos templates.
The Mythos output is produced through the two-stage pipeline: first the app
selects relevant markdown context from the primary and secondary knowledge
bases, then it generates with the content-type template.

Evidence to look for:

- Specific Mortal Vengeance world, author, book, award, audience, or review details
- Brand voice and positioning from the primary knowledge base
- Industry, audience, or platform strategy from the secondary research layer
- Content-type structure from the reusable template
- Fewer unsupported claims and fewer stock ChatGPT-style promo phrases

{render_signal_scan(chatgpt_output, mythos_output)}

## ChatGPT Baseline Prompt

```text
{chatgpt_prompt}
```

## ChatGPT Baseline Output

{chatgpt_output}

## Mythos Content Engine Output

{mythos_output}

## Human Comparison Assessment

Reviewer:

Decision: Pass / Needs revision / Fail

### Scorecard

| Criterion | ChatGPT Baseline | Mythos Output | Human Notes |
|---|---|---|---|
| Brand specificity |  |  | Does the output sound tied to Mythos/Mortal Vengeance rather than any thriller book? |
| Context use |  |  | Does it use knowledge from markdown files without dumping irrelevant context? |
| Audience fit |  |  | Does it address the selected reader, platform, or media audience? |
| Format fit |  |  | Does it match the selected content type's real-world format? |
| Factual discipline |  |  | Does it avoid invented quotes, awards, reviews, or unsupported claims? |
| Editorial usefulness |  |  | Could a human editor polish it quickly instead of rewriting from scratch? |

### Human Notes

- Where Mythos is more specific:
- Where Mythos still feels too much like ChatGPT:
- Required edits before this evidence is submission-ready:
"""


def run_uniqueness_comparison(
    *,
    content_type: str = DEFAULT_CONTENT_TYPE,
    topic: str = DEFAULT_TOPIC,
    custom_chatgpt_prompt: str | None = None,
    chatgpt_output_file: str | None = None,
) -> dict:
    chatgpt_prompt = build_chatgpt_prompt(
        content_type=content_type,
        topic=topic,
        custom_prompt=custom_chatgpt_prompt,
    )
    chatgpt_prompt_path = save_text(chatgpt_prompt, "uniqueness_chatgpt_prompt")

    if chatgpt_output_file:
        chatgpt_output = Path(chatgpt_output_file).read_text(encoding="utf-8").strip()
    else:
        chatgpt_output = generate_text(chatgpt_prompt)
    chatgpt_output_path = save_text(chatgpt_output, "uniqueness_chatgpt_output")

    mythos_result = run_pipeline(content_type=content_type, topic=topic)

    report = render_report(
        content_type=content_type,
        topic=topic,
        chatgpt_prompt=chatgpt_prompt,
        chatgpt_output=chatgpt_output,
        mythos_result=mythos_result,
        chatgpt_prompt_path=chatgpt_prompt_path,
        chatgpt_output_path=chatgpt_output_path,
    )
    report_path = save_text(report, "uniqueness_comparison")

    return {
        "report_path": report_path,
        "chatgpt_prompt_path": chatgpt_prompt_path,
        "chatgpt_output_path": chatgpt_output_path,
        "mythos_filtered_context_path": mythos_result["filtered_context_path"],
        "mythos_prompt_path": mythos_result["prompt_path"],
        "mythos_draft_path": mythos_result["draft_path"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a fresh ChatGPT response against Mythos Content Engine output "
            "and save a Markdown evidence report for human assessment."
        )
    )
    parser.add_argument(
        "--content-type",
        default=DEFAULT_CONTENT_TYPE,
        choices=list_supported_content_types(),
        help="Content type template to test.",
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help="Content request to use for both ChatGPT and Mythos outputs.",
    )
    parser.add_argument(
        "--topic-file",
        help="Optional text file containing the content request. Overrides --topic.",
    )
    parser.add_argument(
        "--chatgpt-prompt",
        help=(
            "Optional ChatGPT baseline prompt. Use {topic} and {content_type} "
            "placeholders if you want them inserted."
        ),
    )
    parser.add_argument(
        "--chatgpt-output-file",
        help=(
            "Optional file containing an actual ChatGPT response to compare against. "
            "When provided, the script does not generate the ChatGPT baseline."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result = run_uniqueness_comparison(
        content_type=args.content_type,
        topic=read_topic(args),
        custom_chatgpt_prompt=args.chatgpt_prompt,
        chatgpt_output_file=args.chatgpt_output_file,
    )

    print("Uniqueness comparison generated successfully.")
    print(f"Report: {result['report_path']}")
    print(f"ChatGPT baseline prompt: {result['chatgpt_prompt_path']}")
    print(f"ChatGPT baseline output: {result['chatgpt_output_path']}")
    print(f"Mythos filtered context: {result['mythos_filtered_context_path']}")
    print(f"Mythos generation prompt: {result['mythos_prompt_path']}")
    print(f"Mythos draft: {result['mythos_draft_path']}")


if __name__ == "__main__":
    main()
