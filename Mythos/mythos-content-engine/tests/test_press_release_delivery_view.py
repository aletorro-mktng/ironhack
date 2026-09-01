from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "src" / "App.tsx").read_text()
CSS = (ROOT / "frontend" / "src" / "styles.css").read_text()
DRAFT_MODEL = (ROOT / "frontend" / "src" / "features" / "generator" / "draftModel.ts").read_text()


def test_press_release_full_deliverable_is_first_tab():
    assert 'contentType === "press_release"' in DRAFT_MODEL
    assert 'id: "full-press-release"' in DRAFT_MODEL
    assert 'label: "Full Press Release"' in DRAFT_MODEL
    assert "deliverables.unshift" in DRAFT_MODEL


def test_press_release_preview_uses_newspaper_template():
    assert "function PressReleaseNewspaper" in APP
    assert 'const isPressRelease = props.draft.contentType === "press_release"' in APP
    assert "isPressRelease ? <PressReleaseNewspaper" in APP
    assert "press-newspaper-preview" in APP
    assert "TellTales Ink Gazette" in APP
    assert ".press-newspaper-preview" in CSS
    assert ".newspaper-columns" in CSS
    assert "column-count: 2" in CSS
