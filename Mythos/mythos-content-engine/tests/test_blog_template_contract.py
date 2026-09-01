from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "templates" / "blog_post.md").read_text()


def test_blog_template_requires_exact_quotes_for_dialogue_posts():
    assert "quote, dialogue, line-ranking" in TEMPLATE
    assert "Do not invent or paraphrase dialogue" in TEMPLATE
    assert "copied exactly from the supplied manuscript or quote-bank context" in TEMPLATE


def test_blog_template_follows_generated_suggested_structure():
    assert "Suggested article structure to use" in TEMPLATE
    assert "include a short `Suggested Structure` section before the article body" in TEMPLATE


def test_blog_template_requires_ranked_character_criteria():
    assert "For ranked character posts, define the ranking criteria before the list" in TEMPLATE
    assert "Do not rank characters by vibe" in TEMPLATE
    assert "Each rank must include a visible **Book evidence** line" in TEMPLATE
    assert "rank + character name, why this rank, Book evidence, criteria notes" in TEMPLATE
    assert "Character rankings explain why each placement belongs there" in TEMPLATE
