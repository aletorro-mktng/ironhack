from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from docx import Document


@dataclass
class PodcastInput:
    source_type: str
    title: str
    raw_text: str


def clean_text(text: str) -> str:
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    return cleaned


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)

    return clean_text("\n".join(pages))


def extract_text_from_docx(file_path: str) -> str:
    document = Document(file_path)
    paragraphs = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            paragraphs.append(paragraph.text)

    return clean_text("\n".join(paragraphs))


def extract_text_from_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        raise ValueError("Please enter a valid URL starting with http:// or https://")

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()

    text = soup.get_text(separator="\n")
    return clean_text(text)


def process_input(
    pasted_text: Optional[str] = None,
    uploaded_file: Optional[str] = None,
    url: Optional[str] = None,
    title: str = "Podcast Recap"
) -> PodcastInput:
    if uploaded_file:
        file_path = Path(uploaded_file)
        extension = file_path.suffix.lower()

        if extension == ".pdf":
            raw_text = extract_text_from_pdf(str(file_path))
            source_type = "pdf"

        elif extension == ".docx":
            raw_text = extract_text_from_docx(str(file_path))
            source_type = "docx"

        else:
            raise ValueError("Unsupported file type. Please upload a PDF or DOCX file.")

    elif url and url.strip():
        raw_text = extract_text_from_url(url.strip())
        source_type = "website"

    elif pasted_text and pasted_text.strip():
        raw_text = clean_text(pasted_text)
        source_type = "text"

    else:
        raise ValueError("Please paste text, upload a PDF/DOCX, or enter a website URL.")

    if len(raw_text) < 100:
        raise ValueError("The extracted content is too short. Please provide a longer input.")

    return PodcastInput(
        source_type=source_type,
        title=title or "Podcast Recap",
        raw_text=raw_text
    )