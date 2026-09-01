from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend" / "src" / "App.tsx").read_text()
CSS = (ROOT / "frontend" / "src" / "styles.css").read_text()


def test_campaign_mode_has_premium_header_actions_and_stepper():
    campaign = APP.split("function CampaignScreen", 1)[1].split("function CampaignAssetControls", 1)[0]
    assert "campaign-page-head" in campaign
    assert "Plan the story.<br />Generate the campaign." in campaign
    assert "Save draft" in campaign
    assert "Continue to Content Mix" in campaign
    for step in ["Strategy", "Content mix", "Schedule", "Review & generate"]:
        assert step in campaign


def test_campaign_strategy_card_uses_clean_hierarchy_and_live_state():
    campaign = APP.split("function CampaignScreen", 1)[1].split("function CampaignAssetControls", 1)[0]
    assert "campaign-panel-head" in campaign
    assert "Set the foundation for your campaign strategy." in campaign
    assert "campaign_goal" in campaign
    assert "compact-date" in campaign
    assert "compact-cta" in campaign
    assert "PODCAST_RECOMMENDED_AUDIENCES" in campaign


def test_campaign_sidebar_is_light_and_split_into_cards():
    campaign = APP.split("function CampaignScreen", 1)[1].split("function CampaignAssetControls", 1)[0]
    assert "campaign-sidebar" in campaign
    assert "Campaign plan" in campaign
    assert "At a glance" in campaign
    assert "Tips & inspiration" in campaign
    assert "Explore Campaign Ideas" in campaign
    campaign_css = CSS.split(".campaign-sidebar", 1)[1].split(".chapter-builder", 1)[0]
    assert "background: #111116" not in campaign_css
    assert ".campaign-glance-card" in CSS


def test_campaign_visualization_prevents_dark_sidebar_and_phase_overflow():
    campaign_css = CSS.split(".campaign-layout", 1)[1].split(".chapter-builder", 1)[0]
    assert "background: rgba(255, 253, 249, 0.98) !important" in campaign_css
    assert "color: var(--ink) !important" in campaign_css
    assert "grid-template-columns: 54px minmax(0, 1fr) 18px" in campaign_css
    assert ".phase-grid strong" in campaign_css
    assert "text-overflow: ellipsis" in campaign_css
    assert ".phase-grid small em" in campaign_css
    assert "height: 44px" in campaign_css


def test_campaign_save_draft_uses_existing_draft_api():
    assert "async function saveCampaignDraft" in APP
    assert "api.createDraft" in APP
    assert "onSave={saveCampaignDraft}" in APP
    assert "Save campaign draft" in APP
