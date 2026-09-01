import type { Deliverable, GeneratedDraft } from "../../types";

export const labelize = (value: string) => value.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
export const optionize = (items: string[]) => items.map((value) => ({ value, label: labelize(value) }));
export const stripMarkdown = (value: string) => value.replace(/```[\s\S]*?```/g, "").replace(/[#>*_`~\[\]()]/g, "").replace(/\s+/g, " ").trim();

const stableId = (value: string, fallback: string) => value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "") || fallback;
const escapeHtml = (value: string) => value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const inlineMarkdownToHtml = (value: string) => escapeHtml(value)
  .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
  .replace(/__(.+?)__/g, "<strong>$1</strong>")
  .replace(/\*(.+?)\*/g, "<em>$1</em>")
  .replace(/_(.+?)_/g, "<em>$1</em>")
  .replace(/`(.+?)`/g, "<code>$1</code>");

export function normalizeDraft(raw: string, title = "Generated Draft", contentType = "content"): GeneratedDraft {
  const source = raw.trim();
  if (!source) return { title, contentType, deliverables: [], rawSource: "" };
  const matches = [...source.matchAll(/^(#{2,3})\s+(.+?)\s*$/gm)];
  const deliverables: Deliverable[] = matches.flatMap((match, index) => {
    const start = (match.index || 0) + match[0].length;
    const end = index + 1 < matches.length ? matches[index + 1].index || source.length : source.length;
    const heading = match[2].trim();
    const content = source.slice(start, end).trim();
    if (!content) return [];
    return [{
      id: `${stableId(heading, "deliverable")}-${index}`,
      label: heading,
      type: match[1].length === 2 ? "section" : "deliverable",
      platform: match[1].length === 2 ? "" : heading.replace(/promo/i, "").trim(),
      title: heading,
      content,
      characterCount: stripMarkdown(content).length,
      status: "generated" as const
    }];
  });
  if (!deliverables.length) {
    deliverables.push({ id: "generated-draft", label: labelize(contentType), type: contentType, title, content: source, characterCount: stripMarkdown(source).length, status: "generated" });
  }
  if (contentType === "press_release" && deliverables[0]?.id !== "full-press-release") {
    deliverables.unshift({
      id: "full-press-release",
      label: "Full Press Release",
      type: "press_release",
      title,
      content: source,
      characterCount: stripMarkdown(source).length,
      status: "generated" as const
    });
  }
  return { title, contentType, deliverables, rawSource: source };
}

export function markdownToHtml(value: string) {
  return value.split("\n").map((line) => {
    const trimmed = line.trim();
    if (trimmed.startsWith("### ")) return `<h3>${inlineMarkdownToHtml(trimmed.slice(4))}</h3>`;
    if (trimmed.startsWith("## ")) return `<h2>${inlineMarkdownToHtml(trimmed.slice(3))}</h2>`;
    if (trimmed.startsWith("- ")) return `<li>${inlineMarkdownToHtml(trimmed.slice(2))}</li>`;
    return trimmed ? `<p>${inlineMarkdownToHtml(trimmed)}</p>` : "";
  }).join("");
}
