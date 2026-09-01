import type { Artifact, AudioEpisode, DraftRecord, GalleryImage, Job, OptionsResponse, PodcastDraft, Voice } from "../types";

type GenerationResponse = {
  content: string;
  draft: DraftRecord;
  result?: { generated_content: string };
  artifacts?: Record<string, string>;
};

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<T>;
}

export const api = {
  options: () => request<OptionsResponse>("/api/options"),
  drafts: () => request<DraftRecord[]>("/api/drafts?limit=100"),
  draft: (id: string) => request<DraftRecord>(`/api/drafts/${id}`),
  createDraft: (body: unknown) => request<DraftRecord>("/api/drafts", json(body)),
  gallery: () => request<GalleryImage[]>("/api/gallery/images"),
  audioLibrary: () => request<AudioEpisode[]>("/api/library/audio"),
  generateContent: (body: unknown) => request<GenerationResponse>("/api/content/generate", json(body)),
  generateCampaign: (body: unknown) => request<GenerationResponse>("/api/campaigns/generate", json(body)),
  generateChapterPromos: (body: unknown) => request<GenerationResponse>("/api/chapter-promos/generate", json(body)),
  generatePodcast: (body: unknown) => request<PodcastDraft>("/api/podcasts/scripts", json(body)),
  podcast: (id: string) => request<PodcastDraft>(`/api/podcasts/${id}`),
  updatePodcast: (id: string, body: unknown) => request<PodcastDraft>(`/api/podcasts/${id}`, { ...json(body), method: "PATCH" }),
  voices: () => request<Voice[]>("/api/voices"),
  podcastPreview: (draftId: string, body: unknown) => request<Job>(`/api/podcasts/${draftId}/preview`, json(body)),
  podcastRender: (draftId: string, body: unknown) => request<Job>(`/api/podcasts/${draftId}/render`, json(body)),
  exportContent: (body: unknown) => request<Artifact>("/api/exports", json(body)),
  artifactDownloadUrl: (id: string) => `/api/artifacts/${id}/download`,
  job: (id: string) => request<Job>(`/api/jobs/${id}`)
};

function json(body: unknown): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  };
}
