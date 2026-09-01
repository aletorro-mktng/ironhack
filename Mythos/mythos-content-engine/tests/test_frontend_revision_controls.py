from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "src" / "App.tsx").read_text()


def test_regenerate_buttons_rerun_current_generation_flow():
    assert "async function regenerateCurrentDraft" in APP
    assert 'postGenerate("chapter", chapterForm, "chapter_promos")' in APP
    assert 'postGenerate("campaign", { ...genericForm, content_types: campaignTypes }, "campaign")' in APP
    assert 'postGenerate("content", genericForm, genericForm.content_type)' in APP
    assert "onRegenerate={regenerateCurrentDraft}" in APP
    assert "Use Edit brief to change inputs, then regenerate." not in APP


def test_revision_controls_update_selected_deliverable():
    assert "function revisionDirection" in APP
    assert "async function applyRevision" in APP
    assert "Selected deliverable to revise" in APP
    assert "Current selected draft" in APP
    assert "revisionContext" in APP
    assert 'postGenerate("chapter", {' in APP
    assert 'postGenerate("campaign", {' in APP
    assert 'postGenerate("content", {' in APP
    assert "updateSelectedContent(revised)" not in APP
    assert "Cinematic revision:" not in APP


def test_refine_presets_apply_immediately():
    assert 'onApplyRevision("Make the selected deliverable more cinematic.")' in APP
    assert 'onApplyRevision("Increase tension while preserving spoiler limits.")' in APP
    assert 'onApplyRevision("Strengthen the CTA.")' in APP


def test_suggested_hashtags_show_generated_or_fallback_tags():
    assert "function hashtagTextFromDraft" in APP
    assert "BOOKSTAGRAM_HASHTAG_POOL" in APP
    assert "#Bookstagram" in APP
    assert "#CaribbeanGothic" in APP
    assert "#HorrorBookstagram" in APP
    assert "#PsychologicalThriller" in APP
    assert "suggestedHashtags" in APP
    assert "#MortalVengeance #YAHorror #BookTok" in APP
    assert "hashtag-row--content" in APP
    assert "props.onCopy(suggestedHashtags)" in APP
