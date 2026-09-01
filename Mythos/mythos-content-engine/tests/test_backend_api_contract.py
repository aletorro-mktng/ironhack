from fastapi.testclient import TestClient

from backend.main import app


def test_backend_exposes_core_api_contract():
    paths = app.openapi()["paths"]
    expected = {
        "/api/content/generate",
        "/api/podcasts/scripts",
        "/api/podcasts/{draft_id}",
        "/api/podcasts/{draft_id}/preview",
        "/api/podcasts/{draft_id}/render",
        "/api/jobs/{job_id}",
        "/api/jobs/{job_id}/events",
        "/api/artifacts/{artifact_id}/download",
        "/api/exports",
        "/api/voices",
        "/api/voices/{voice_id}/preview",
    }
    assert expected.issubset(paths)


def test_openapi_contains_normalized_podcast_contract():
    schemas = app.openapi()["components"]["schemas"]
    assert {"PerformanceCue", "PodcastDraft", "PodcastSegment", "PodcastSpeaker", "PodcastTurn"}.issubset(schemas)
    podcast_draft = schemas["PodcastDraft"]["properties"]
    assert {"show_title", "episode_title", "speakers", "segments", "raw_script"}.issubset(podcast_draft)
    assert schemas["PodcastTurn"]["properties"]["cues"]["items"]["$ref"].endswith("/PerformanceCue")


def test_backend_serves_options_and_library_data_without_ui_dependencies():
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        options = client.get("/api/options")
        assert options.status_code == 200
        assert "contentTypes" in options.json()
        assert client.get("/api/drafts?limit=1").status_code == 200
        assert client.get("/api/gallery/images").status_code == 200
        assert client.get("/api/library/audio").status_code == 200


def test_podcast_render_endpoints_return_jobs_immediately():
    with TestClient(app) as client:
        response = client.post("/api/podcasts/example-draft/preview")
        assert response.status_code == 200
        payload = response.json()
        assert payload["id"].startswith("job_")
        assert payload["status"] in {"queued", "running"}
        assert "progress" in payload

        status = client.get(f"/api/jobs/{payload['id']}")
        assert status.status_code == 200
        assert status.json()["id"] == payload["id"]
