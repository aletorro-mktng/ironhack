from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "src" / "App.tsx").read_text()
CSS = (ROOT / "frontend" / "src" / "styles.css").read_text()


def podcast_studio_source() -> str:
    return APP.split("function PodcastStudio", 1)[1].split("function DraftCanvas", 1)[0]


def test_podcast_generated_tabs_render_distinct_panels():
    studio = podcast_studio_source()
    assert "renderScriptPanel" in studio
    assert 'section === "Voices"' in studio
    assert 'section === "Audio"' in studio
    assert 'section === "Assets"' in studio
    assert 'section === "Compare"' in studio
    assert "podcast-tab-panel" in CSS
    assert "compare-panel" in CSS


def test_podcast_turn_controls_are_clickable():
    studio = podcast_studio_source()
    assert "setEditingTurn(turn.id)" in studio
    assert 'toggleCue("Pause")' in studio
    assert "previewTurn(turn.id)" in studio
    assert "setTurnMenu" in studio
    assert "turn-menu" in studio
    assert "Save turn" in studio


def test_podcast_segment_and_performance_cues_are_independent_controls():
    studio = podcast_studio_source()
    assert "function addSegment" in studio
    assert "setExtraSegments" in studio
    assert 'onClick={addSegment}' in studio
    assert "function toggleCue" in studio
    assert "activeCues.includes(cue)" in studio
    assert "cue-grid button.selected" in CSS
    assert "selectedVoiceId" not in studio.split("function toggleCue", 1)[1].split("function previewTurn", 1)[0]


def test_podcast_inspector_separates_speaker_from_voice():
    studio = podcast_studio_source()
    assert "selectedSpeakerId" in studio
    assert "setSelectedSpeakerId" in studio
    assert "{turn.name} - {turn.role}" in studio
    assert "selectedVoiceId" in studio
    assert "props.voiceOptions.map" in studio


def test_podcast_assets_support_production_tags():
    studio = podcast_studio_source()
    for text in ["Voice performance tags", "Sound effects tags", "Music tags"]:
        assert text in studio
    assert "voiceTags" in studio
    assert "soundEffectTags" in studio
    assert "musicTags" in studio
