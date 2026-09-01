from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "src" / "App.tsx").read_text()
CSS = (ROOT / "frontend" / "src" / "styles.css").read_text()
SCHEMAS = (ROOT / "backend" / "models" / "schemas.py").read_text()
PODCAST_SERVICE = (ROOT / "backend" / "services" / "podcast_generator.py").read_text()


def test_podcast_settings_uses_guided_four_section_workflow():
    podcast_screen = APP.split("function PodcastBriefScreen", 1)[1].split("function DraftStudio", 1)[0]
    for text in [
        "Episode Essentials",
        "Tone & Audience",
        "Constraints & Guardrails",
        "Production Settings",
        "Episode Summary",
        "What this will create",
    ]:
        assert text in podcast_screen
    assert "brief-grid" not in podcast_screen


def test_podcast_settings_has_live_summary_and_action_buttons():
    assert "podcast-summary-card" in APP
    assert "Edit All Settings" in APP
    assert "Explore Inspiration" in APP
    assert "validateAndGenerate" in APP
    assert "onSave={savePodcastSettings}" in APP
    assert "api.createDraft" in APP


def test_podcast_settings_maps_structured_roles_to_existing_payload():
    assert "roleListFromValue" in APP
    assert "roleDefaultsForFormat" in APP
    assert "speaker_roles: nextRoles.join" in APP
    assert "Speaker {index + 1}" in APP


def test_podcast_payload_preserves_new_configuration_fields():
    for field in ["audience", "constraints", "performance_cues", "custom_constraints", "model_id", "knowledge_sources", "source_focus"]:
        assert field in SCHEMAS
        assert field in PODCAST_SERVICE


def test_podcast_settings_exposes_knowledge_source_toggles():
    podcast_screen = APP.split("function PodcastBriefScreen", 1)[1].split("function DraftStudio", 1)[0]
    assert "Knowledge Sources" in podcast_screen
    assert "KNOWLEDGE_SOURCE_OPTIONS.map" in podcast_screen
    assert "Manuscript/RAG moments" in APP
    assert "Pull quotes" in APP
    assert "Reviews" in APP
    assert "Character info" in APP
    assert "Source focus" in podcast_screen


def test_podcast_settings_layout_is_responsive():
    assert ".podcast-settings-layout" in CSS
    assert "grid-template-columns: minmax(0, 1fr) 340px" in CSS
    assert ".podcast-settings-sidebar" in CSS
    assert "grid-template-columns: 1fr" in CSS
