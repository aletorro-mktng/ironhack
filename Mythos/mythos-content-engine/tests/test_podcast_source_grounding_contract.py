from backend.models import PodcastGenerationRequest
from backend.services.podcast_generator import podcast_brief


def test_podcast_brief_includes_knowledge_source_controls():
    brief = podcast_brief(
        PodcastGenerationRequest(
            topic="Top 20 funniest lines in Mortal Vengeance: A Grim Tale",
            show_title="Grim Talk",
            episode_title="Funniest Lines",
            knowledge_sources=["Manuscript/RAG moments", "Pull quotes", "Character info"],
            source_focus="funniest dialogue and character context",
        )
    )

    assert "Knowledge sources to use: Manuscript/RAG moments, Pull quotes, Character info" in brief
    assert "Source focus: funniest dialogue and character context" in brief
    assert "Source grounding: when manuscript/RAG moments" in brief
    assert "Do not invent dialogue" in brief
    assert "Use character info to correctly attribute actions" in brief
    assert "Verified source evidence from the selected book:" in brief
    assert "Selected book/source: Mortal Vengeance: A Grim Tale" in brief
