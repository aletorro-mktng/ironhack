from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRAFT_MODEL = (ROOT / "frontend" / "src" / "features" / "generator" / "draftModel.ts").read_text()


def test_markdown_preview_renders_inline_formatting_safely():
    assert "const escapeHtml" in DRAFT_MODEL
    assert "const inlineMarkdownToHtml" in DRAFT_MODEL
    assert '.replace(/\\*\\*(.+?)\\*\\*/g, "<strong>$1</strong>")' in DRAFT_MODEL
    assert '.replace(/\\*(.+?)\\*/g, "<em>$1</em>")' in DRAFT_MODEL
    assert '.replace(/`(.+?)`/g, "<code>$1</code>")' in DRAFT_MODEL
    assert 'replace(/&/g, "&amp;")' in DRAFT_MODEL
    assert 'replace(/</g, "&lt;")' in DRAFT_MODEL
