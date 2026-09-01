from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "src" / "App.tsx").read_text()
CSS = (ROOT / "frontend" / "src" / "styles.css").read_text()


def test_chapter_promos_uses_guided_builder_sections():
    for text in [
        "Source Material",
        "Distribution",
        "Creative Strategy",
        "Guardrails",
        "CTA & Final Output",
        "Campaign Summary",
    ]:
        assert text in APP


def test_chapter_picker_is_scrollable_list_not_wrapped_tags():
    assert "chapter-list" in APP
    assert ".chapter-list" in CSS
    assert "max-height: 340px" in CSS
    assert "overflow: auto" in CSS


def test_chapter_builder_validates_required_inputs_before_generation():
    assert "validateAndGenerate" in APP
    assert "Choose at least one book or project." in APP
    assert "Choose at least one chapter." in APP
    assert "Choose at least one platform." in APP
    assert "Choose at least one deliverable." in APP


def test_chapter_summary_updates_from_live_form_state():
    assert "selectedChapters" in APP
    assert "chapterSummary" in APP
    assert "selectedChapters.map" in APP
    assert "form.platforms.map" in APP
    assert "form.moods.join" in APP
    assert "selectedOutputs.length" in APP


def test_chapter_selection_replaces_by_default_and_generation_uses_current_form():
    assert "function selectOnlyChapter" in APP
    assert "chapters: [id]" in APP
    assert "chapter-add-toggle" in APP
    assert "onGenerate({ ...form, chapters: [...form.chapters] })" in APP
    assert 'onGenerate={(payload) => postGenerate("chapter", payload, "chapter_promos")}' in APP


def test_source_material_renders_available_books_not_only_selected_books():
    assert "visibleBooks.map" in APP
    assert "availableBooks = options?.books" in APP
    assert "No books found." in APP


def test_creative_strategy_has_populated_fallback_options():
    for token in [
        "characterOptions",
        "hookOptions",
        "moodOptions",
        "genreOptions",
        "styleVariantOptions",
        "goalOptions",
        "modeOptions",
    ]:
        assert token in APP
    assert "Alex Herrera" in APP
    assert "Choose the Strongest Hook for Me" in APP
    assert "Spoiler-Safe Suspense" in APP


def test_explore_ideas_is_near_top_of_sidebar_and_targets_creative_section():
    summary = APP.split('<aside className="chapter-summary">', 1)[1].split("</aside>", 1)[0]
    assert summary.index("inspiration-card top") < summary.index("<dl>")
    assert 'data-section="creative"' in APP
    assert "Explore Ideas" in APP


def test_generated_chapter_promos_are_grouped_by_chapter_before_outputs():
    studio = APP.split("function ChapterPromotionStudio", 1)[1].split("function PodcastStudio", 1)[0]
    assert "chapterPromotionGroups" in APP
    assert "ChapterPromoGroup" in APP
    assert "chapterOutputGroups" in APP
    assert "ChapterOutputGroup" in APP
    assert "chapterGroups.map" in studio
    assert "openChapter(group)" in studio
    assert "<h2>Chapters</h2>" in studio
    assert "Outputs for" in studio
    assert "outputGroups.map" in studio
    assert "aria-expanded" in studio
    assert "expandedOutputGroup" in studio
    assert "chapter-switcher" in studio
    assert "chapter-context-panel" in studio


def test_generated_chapter_promo_layout_has_chapter_switcher_styles():
    assert ".chapter-switcher" in CSS
    assert ".chapter-output-groups" in CSS
    assert ".chapter-output-group-head" in CSS
    assert ".chapter-context-panel" in CSS


def test_generated_chapter_promo_uses_chapter_scoped_actions_and_compact_refine():
    studio = APP.split("function ChapterPromotionStudio", 1)[1].split("function PodcastStudio", 1)[0]
    assert "Copy chapter package" in studio
    assert "chapterPackage" in studio
    assert "quick-adjust-grid" in studio
    assert "Custom instruction" in studio
    assert "versionOpen" in studio
    assert "View history" in studio
    assert "chapter-selected-meta" in studio


def test_generated_chapter_source_tab_is_scoped_to_active_chapter():
    studio = APP.split("function ChapterPromotionStudio", 1)[1].split("function PodcastStudio", 1)[0]
    assert '<pre className="source">{chapterPackage}</pre>' in studio
    assert '<pre className="source">{props.draft.rawSource}</pre>' not in studio


def test_generated_chapter_promo_columns_scroll_independently():
    assert "height: calc(100vh - 198px)" in CSS
    assert "overflow: hidden" in CSS
    assert "overflow-y: auto" in CSS
