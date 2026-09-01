from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "src" / "App.tsx").read_text()
CSS = (ROOT / "frontend" / "src" / "styles.css").read_text()


def test_generated_views_can_return_to_creation_flows():
    assert "onCreateNew={() => clearDraft(\"chapter-promos\")}" in APP
    assert "onCreateNew={() => clearDraft(tab)}" in APP
    assert "onCreateNew={() => clearDraft(\"podcast\")}" in APP
    assert "Back to builder" in APP
    assert "Back to brief" in APP
    assert "Create new promos" in APP
    assert "Create new script" in APP


def test_chapter_promo_generated_header_uses_full_app_navigation():
    chapter_studio = APP.split("function ChapterPromotionStudio", 1)[1].split("function PodcastStudio", 1)[0]
    for label in [
        "Dashboard",
        "Generator",
        "Campaign Mode",
        "Podcast Studio",
        "Chapter Promos",
        "Saved Drafts",
        "Gallery",
        "Library",
    ]:
        assert label in chapter_studio
    assert "Draft Studio" not in chapter_studio
    assert ".back-link" in CSS
