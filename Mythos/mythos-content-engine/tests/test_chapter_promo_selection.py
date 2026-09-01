from src.frontend_adapters import normalize_multi_select
from src import chapter_reader


def test_normalize_multi_select_handles_lists_and_singletons():
    assert normalize_multi_select(["spoiler_safe_teaser", "prestige_tv_preview"]) == [
        "spoiler_safe_teaser",
        "prestige_tv_preview",
    ]
    assert normalize_multi_select("Ominous") == ["Ominous"]
    assert normalize_multi_select(None, fallback=["A Character in Danger", "A Hidden Betrayal"]) == [
        "A Character in Danger",
        "A Hidden Betrayal",
    ]


def test_grim_tale_chapter_titles_are_exposed():
    options = chapter_reader.chapter_options("Mortal Vengeance: A Grim Tale")
    assert options["chapter-1"] == "Chapter 1: The Devil is in the Details"
    assert options["chapter-2"] == "Chapter 2: The Devil's Advocate"
    assert options["chapter-24"] == "Chapter 24: The Devil's Day Job"


def test_grim_tale_sections_are_not_dropped_from_chapter_promos():
    labels = chapter_reader.chapter_options("Mortal Vengeance: A Grim Tale")
    assert "prologue" in labels
    assert "b-roll-the-excelsior-archive-a-dossier-of-recovered-media" in labels
    assert "epilogue" in labels
    assert len(labels) >= 29
