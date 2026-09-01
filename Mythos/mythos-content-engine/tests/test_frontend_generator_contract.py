from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "src" / "App.tsx").read_text()
CSS = (ROOT / "frontend" / "src" / "styles.css").read_text()
FIELDS = (ROOT / "frontend" / "src" / "components" / "Fields.tsx").read_text()


def test_deliverable_cards_use_full_width_grid_contract():
    assert "grid-template-columns: 44px minmax(0, 1fr) 20px" in CSS
    assert ".deliverable-card__content" in CSS
    assert "min-width: 0" in CSS
    assert "word-break: normal" in CSS
    assert "min-height: 110px" in CSS


def test_deliverable_registry_has_required_distinct_ids():
    for deliverable_id in [
        "instagram-caption",
        "social-caption",
        "carousel",
        "quote-post",
        "reel-script",
        "blog-article",
        "press-release",
        "newsletter",
        "podcast-episode",
        "campaign-package",
    ]:
        assert f'id: "{deliverable_id}"' in APP


def test_social_caption_has_distinct_outputs_and_summary_updates():
    assert 'id: "social-caption"' in APP
    assert "Platform adaptations" in APP
    assert "selected.defaultOutputs.map" in APP


def test_reel_script_belongs_to_video_and_activates_category_from_selection():
    reel_block = APP.split('id: "reel-script"', 1)[1].split('id: "blog-article"', 1)[0]
    assert 'category: "Video"' in reel_block
    assert "const category = selected.category" in APP


def test_category_switch_selects_visible_default_deliverable():
    assert "function selectCategory" in APP
    assert "supported.find((item) => item.category === next)" in APP
    assert "selectDeliverable(first)" in APP


def test_carousel_reveals_slide_controls():
    carousel_block = APP.split('id: "carousel"', 1)[1].split('id: "quote-post"', 1)[0]
    assert '"slide_count"' in carousel_block
    assert '"carousel_structure"' in carousel_block
    assert '"cover_hook"' in carousel_block


def test_blog_article_has_seo_controls_without_video_fields():
    blog_block = APP.split('id: "blog-article"', 1)[1].split('id: "press-release"', 1)[0]
    assert '"seo_keyword"' in blog_block
    assert '"metadata_controls"' in blog_block
    assert '"image_generation"' in blog_block
    assert '"visual_style"' in blog_block
    assert '"supporting_image"' in blog_block
    assert '"shot_density"' not in blog_block


def test_blog_article_restores_broad_format_and_image_generation_options():
    assert "suggest structure" in APP
    assert "character analysis" in APP
    assert "theme essay" in APP
    assert "award announcement feature" in APP
    assert "book club guide" in APP
    assert "worldbuilding deep dive" in APP
    assert "const IMAGE_GENERATION_MODES" in APP
    assert "Source books" in APP
    assert "related_books" in APP
    assert "Create prompt only" in APP
    assert "Do not generate visuals" in APP
    assert "Use book cover" in APP
    assert "const IMAGE_FORMATS" in APP
    assert "portrait 4:5" in APP
    assert "story/reel 9:16" in APP
    assert "transparent cutout" in APP
    assert "Knowledge Sources" in APP
    assert "Manuscript/RAG moments" in APP
    assert "Pull quotes" in APP
    assert "Character info" in APP
    assert "Source focus" in APP


def test_blog_advanced_settings_do_not_duplicate_visible_image_controls():
    blog_block = APP.split('id: "blog-article"', 1)[1].split('id: "press-release"', 1)[0]
    advanced_line = next(line for line in blog_block.splitlines() if "advancedFields:" in line)
    assert '"image_generation"' not in advanced_line
    assert '"visual_style"' not in advanced_line
    assert '"supporting_image"' not in advanced_line


def test_press_release_does_not_require_platform():
    press_block = APP.split('id: "press-release"', 1)[1].split('id: "newsletter"', 1)[0]
    assert "defaultDestinationIds: []" in press_block
    assert '"news_angle"' in press_block
    assert '"byline_city"' in press_block
    assert '"media_contact"' in press_block
    assert "Generate Press Release" in press_block


def test_podcast_reveals_runtime_and_voice_settings():
    podcast_block = APP.split('id: "podcast-episode"', 1)[1].split('id: "campaign-package"', 1)[0]
    assert '"podcast_format"' in podcast_block
    assert '"runtime"' in podcast_block
    assert '"voice_settings"' in podcast_block


def test_campaign_package_opens_campaign_mode():
    campaign_block = APP.split('id: "campaign-package"', 1)[1].split("];", 1)[0]
    assert "Open Campaign Mode" in campaign_block
    assert "onOpenCampaign()" in APP


def test_every_deliverable_normalizes_generation_payload():
    assert "const payload = {" in APP
    assert "destination_ids: destinations" in APP
    assert "formats: [selected.label" in APP


def test_shared_values_survive_deliverable_changes():
    assert "audience: form.audience.length ? form.audience" in APP
    assert "tone: form.tone?.length ? form.tone" in APP
    assert "cta: form.cta ||" in APP


def test_content_specific_fields_render_inside_brief_builder():
    assert "selected.advancedFields.map(renderAdvancedField)" in APP
    assert "ContentSpecificFields form={form}" in APP


def test_press_release_uses_press_brief_fields_instead_of_promo_fields():
    assert "const isPressRelease = selected.contentType === \"press_release\"" in APP
    assert "!isPressRelease && <MultiSelect label=\"Tone\"" in APP
    assert "!isPressRelease && <label className=\"field\"><span>Call to action</span>" in APP
    assert "!isPressRelease && <SingleSelect label=\"Spoiler level\"" in APP
    assert "Auto-generate news angle / hook" in APP
    assert "Auto-generate boilerplate" in APP
    assert "Byline city" in APP
    assert "Media contact" in APP
    assert "media_contact_email" in APP
    assert "media_contact_phone" in APP
    assert "media_contact_website" in APP


def test_chip_multiselect_replaces_native_multiple_select():
    assert "chip-multiselect" in FIELDS
    assert "aria-multiselectable" in FIELDS
    assert "<select multiple" not in FIELDS


def test_missing_required_fields_have_visible_aria_live_errors():
    assert "aria-live=\"polite\"" in APP
    assert "field-error" in APP
    assert "Add a topic or announcement before generating." in APP
