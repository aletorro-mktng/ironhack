from backend.models import ContentGenerationRequest
from backend.services.content_generator import content_brief


def test_press_release_request_accepts_media_and_release_fields():
    request = ContentGenerationRequest(
        content_type="press_release",
        topic="Mortal Vengeance wins a major award",
        related_book="Mortal Vengeance",
        objectives=["Announce news"],
        press_timing="Publish after specific date",
        publication_date="2026-09-08",
        byline_city="New York, NY",
        auto_news_angle=True,
        auto_boilerplate=True,
        media_contact_name="Alejandro Torres De la Rocha",
        media_contact_email="press@example.com",
        media_contact_phone="+1 555 000 0000",
        media_contact_website="https://telltales.ink",
        companion_assets=["media pitch email"],
    )

    assert request.publication_date == "2026-09-08"
    assert request.byline_city == "New York, NY"
    assert request.auto_news_angle is True
    assert request.auto_boilerplate is True
    assert request.media_contact_email == "press@example.com"


def test_press_release_brief_uses_press_fields_not_social_promo_fields():
    brief = content_brief(
        ContentGenerationRequest(
            content_type="press_release",
            topic="Mortal Vengeance wins a major award",
            related_book="Mortal Vengeance",
            platform="Website Newsroom",
            objectives=["Announce news"],
            cta="Read Mortal Vengeance",
            press_timing="FOR IMMEDIATE RELEASE",
            byline_city="New York, NY",
            news_angle="Award recognition gives horror readers a timely reason to discover the book.",
            boilerplate="Alejandro Torres De la Rocha writes cinematic YA horror.",
            media_contact_email="press@example.com",
            media_contact_website="https://telltales.ink",
        )
    )

    assert "Format: professional press release." in brief
    assert "Byline city: New York, NY" in brief
    assert "News angle / hook: Award recognition" in brief
    assert "Boilerplate: Alejandro Torres" in brief
    assert "press@example.com" in brief
    assert "https://telltales.ink" in brief
    assert "Do not frame this as a social promo" in brief
    assert "CTA: Read Mortal Vengeance" not in brief
