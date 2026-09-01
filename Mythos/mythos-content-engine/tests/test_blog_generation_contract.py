from backend.models import ContentGenerationRequest
from backend.services import content_generator
from backend.services.content_generator import content_brief


def test_blog_generation_request_accepts_blog_and_image_fields():
    request = ContentGenerationRequest(
        content_type="blog_post",
        topic="Why Mortal Vengeance resonates with YA horror readers",
        related_book="Mortal Vengeance",
        related_books=["Mortal Vengeance", "Mortal Vengeance: A Grim Tale"],
        blog_format="character analysis",
        word_count="1200-1800",
        structure=["hook introduction", "pull quote block"],
        seo_keyword="YA horror revenge novel",
        heading_depth="Detailed",
        metadata_controls="Full metadata",
        image_generation="Generate visuals",
        supporting_image="Use generated scene image",
        visual_style="Dark Academia",
        visual_formats=["blog cover 16:9", "story/reel 9:16"],
        knowledge_sources=["Manuscript/RAG moments", "Pull quotes", "Character info"],
        source_focus="Alex savage moments",
    )

    assert request.blog_format == "character analysis"
    assert request.related_books == ["Mortal Vengeance", "Mortal Vengeance: A Grim Tale"]
    assert request.image_generation == "Generate visuals"
    assert request.visual_formats == ["blog cover 16:9", "story/reel 9:16"]
    assert request.knowledge_sources == ["Manuscript/RAG moments", "Pull quotes", "Character info"]
    assert request.source_focus == "Alex savage moments"


def test_blog_brief_includes_image_generation_controls():
    brief = content_brief(
        ContentGenerationRequest(
            content_type="blog_post",
            topic="Why Mortal Vengeance resonates with YA horror readers",
            related_books=["Mortal Vengeance", "Mortal Vengeance: A Grim Tale"],
            blog_format="theme essay",
            word_count="1200-1800",
            structure=["subheaded sections"],
            seo_keyword="YA horror revenge novel",
            heading_depth="Detailed",
            metadata_controls="Full metadata",
            image_generation="Generate visuals",
            supporting_image="Use book cover",
            visual_style="Gothic",
            visual_formats=["blog cover 16:9", "quote graphic"],
            knowledge_sources=["Reviews", "Pull quotes"],
            source_focus="reader reception and quotable lines",
        )
    )

    assert "Blog format: theme essay" in brief
    assert "Blog length: 1200-1800" in brief
    assert "Structure checklist: subheaded sections" in brief
    assert "Image generation: Generate visuals" in brief
    assert "Supporting image: Use book cover" in brief
    assert "Visual style: Gothic" in brief
    assert "Requested image formats: blog cover 16:9, quote graphic" in brief
    assert "Related book/source: Mortal Vengeance, Mortal Vengeance: A Grim Tale" in brief
    assert "Knowledge sources to use: Reviews, Pull quotes" in brief
    assert "Source focus: reader reception and quotable lines" in brief


def test_blog_quote_listicles_require_exact_source_quotes():
    brief = content_brief(
        ContentGenerationRequest(
            content_type="blog_post",
            topic="Create a blog listicle for the top 20 funniest dialogue lines in Mortal Vengeance: A Grim Tale",
            related_book="Mortal Vengeance: A Grim Tale",
            blog_format="listicle",
            structure=["suggest structure"],
            knowledge_sources=["Manuscript/RAG moments", "Pull quotes", "Character info"],
            source_focus="funniest dialogue lines",
        )
    )

    assert "Quote grounding: this request requires exact source quotes" in brief
    assert "Do not invent, paraphrase, modernize, or approximate dialogue" in brief
    assert "every numbered item must include one exact quoted line or clearly identified manuscript-grounded moment" in brief
    assert "When Character info is enabled" in brief
    assert "Suggested article structure to use:" in brief
    assert "Ranked list of up to 20 items" in brief
    assert "correct character attribution" in brief


def test_blog_character_moment_listicles_require_rag_grounding():
    brief = content_brief(
        ContentGenerationRequest(
            content_type="blog_post",
            topic="Top 20 times when Alex was a savage and the best friendship moments",
            related_book="Mortal Vengeance",
            blog_format="listicle",
            knowledge_sources=["Manuscript/RAG moments", "Character info", "Reviews"],
            source_focus="Alex savage moments and friendship scenes",
        )
    )

    assert "Knowledge sources to use: Manuscript/RAG moments, Character info, Reviews" in brief
    assert "Source focus: Alex savage moments and friendship scenes" in brief
    assert "Do not invent, paraphrase, modernize, or approximate dialogue" in brief
    assert "use review context only for reception/positioning" in brief


def test_blog_character_moment_listicles_include_selected_book_evidence_pack():
    brief = content_brief(
        ContentGenerationRequest(
            content_type="blog_post",
            topic="Top 10 times characters were savage in Mortal Vengeance: A Grim Tale",
            related_book="Mortal Vengeance: A Grim Tale",
            blog_format="listicle",
            knowledge_sources=["Manuscript/RAG moments", "Character info"],
            source_focus="savage character dialogue and moments",
        )
    )

    assert "Verified source evidence from the selected book:" in brief
    assert "Selected book/source: Mortal Vengeance: A Grim Tale" in brief
    assert "[Mortal Vengeance: A Grim Tale - Chapter" in brief
    assert "Do not use another book. Do not invent phrases, characters, or explanations." in brief
    assert "Mortal Vengeance II: To Reel or Not Too Real?" not in brief


def test_blog_source_evidence_pack_can_include_multiple_selected_books():
    brief = content_brief(
        ContentGenerationRequest(
            content_type="blog_post",
            topic="Top 10 friendship moments across Mortal Vengeance and Mortal Vengeance: A Grim Tale",
            related_books=["Mortal Vengeance", "Mortal Vengeance: A Grim Tale"],
            blog_format="listicle",
            knowledge_sources=["Manuscript/RAG moments", "Character info"],
            source_focus="friendship moments",
        )
    )

    assert "Selected book/source: Mortal Vengeance, Mortal Vengeance: A Grim Tale" in brief
    assert "[Mortal Vengeance - Chapter" in brief or "[Mortal Vengeance: A Grim Tale - Chapter" in brief


def test_generate_visual_artifacts_runs_for_requested_blog_visuals(monkeypatch, tmp_path):
    calls = []

    def fake_generate_external_visual(**kwargs):
        calls.append(kwargs)
        path = tmp_path / f"{len(calls)}.png"
        path.write_bytes(b"fake")
        return {"path": str(path), "provider": "test", "prompt": "prompt"}

    monkeypatch.setattr(content_generator, "generate_external_visual", fake_generate_external_visual)
    monkeypatch.setattr(content_generator, "new_visual_output_dir", lambda _prefix: tmp_path)

    result = content_generator._generate_visual_artifacts(
        ContentGenerationRequest(
            content_type="blog_post",
            topic="Top 10 funniest lines",
            image_generation="Generate visuals",
            visual_style="Gothic",
            visual_formats=["blog cover 16:9", "quote graphic"],
        ),
        "Generated blog content",
    )

    assert len(calls) == 2
    assert result["paths"] == [str(tmp_path / "1.png"), str(tmp_path / "2.png")]
    assert calls[0]["format_label"] == "blog cover 16:9"
    assert calls[1]["format_label"] == "quote graphic"


def test_visual_artifacts_do_not_run_for_prompt_only(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(content_generator, "generate_external_visual", lambda **kwargs: calls.append(kwargs))

    result = content_generator._generate_visual_artifacts(
        ContentGenerationRequest(
            content_type="blog_post",
            topic="A prompt only blog",
            image_generation="Create prompt only",
            visual_formats=["blog cover 16:9"],
        ),
        "Generated blog content",
    )

    assert calls == []
    assert result["paths"] == []


def test_suggest_structure_creates_concrete_article_architecture():
    brief = content_brief(
        ContentGenerationRequest(
            content_type="blog_post",
            topic="Explain the friendship dynamics in Mortal Vengeance",
            related_book="Mortal Vengeance",
            blog_format="theme essay",
            structure=["suggest structure"],
            knowledge_sources=["Manuscript/RAG moments", "Character info"],
        )
    )

    assert "Suggested article structure to use:" in brief
    assert "Strong title and opening hook" in brief
    assert "Main sections ordered from broad idea to specific source-backed support" in brief
    assert "Evidence from knowledge base or manuscript" in brief


def test_best_character_rankings_require_a_reasoned_source_grounded_rubric():
    brief = content_brief(
        ContentGenerationRequest(
            content_type="blog_post",
            topic="Write a blog ranking the best characters in Mortal Vengeance: A Grim Tale",
            related_book="Mortal Vengeance: A Grim Tale",
            blog_format="listicle",
            structure=["suggest structure"],
            knowledge_sources=["Manuscript/RAG moments", "Character info"],
            source_focus="best characters and strongest arcs",
        )
    )

    assert "Character ranking rubric:" in brief
    assert "Ranking Criteria" in brief
    assert "Rank characters by a consistent rubric, not by arbitrary preference" in brief
    assert "narrative impact, agency and decision-making, relationship significance" in brief
    assert "explain why that character belongs at that exact rank" in brief
    assert "Do not rank characters without source support" in brief
    assert "Mandatory ranked-entry format: Rank + character name; Why this rank; Book evidence; Criteria notes." in brief
    assert "If there is no Book evidence for a character, do not include that character" in brief
    assert "Suggested article structure to use:" in brief
    assert "Ranked list of up to 10 characters" in brief
    assert "Book evidence line" in brief
    assert "Character evidence map from the selected book/source:" in brief
    assert "Every ranked character must cite one of these evidence entries or be omitted." in brief
    assert "Verified source evidence from the selected book:" in brief
