"""Lightweight RAG over Domain_documents + Wikipedia context."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import requests
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

DOMAIN_DIR = Path(__file__).resolve().parents[1] / "Domain_documents"

# Wikipedia requires a descriptive User-Agent; missing UA often yields empty/HTML bodies
# which the wikipedia package then fails to parse as JSON.
_WIKI_HEADERS = {
    "User-Agent": "SDLCAutomationApp/1.0 (educational; contact=local-dev)",
    "Accept": "application/json",
}
_WIKI_API = "https://en.wikipedia.org/w/api.php"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_domain_chunks(chunk_size: int = 500) -> list[dict]:
    chunks: list[dict] = []
    if not DOMAIN_DIR.exists():
        return chunks
    for pdf in sorted(DOMAIN_DIR.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:
            continue
        text = re.sub(r"\s+", " ", text).strip()
        for i in range(0, len(text), chunk_size):
            piece = text[i : i + chunk_size]
            if piece:
                chunks.append({"source": pdf.name, "text": piece})
    return chunks


def retrieve_domain(query: str, top_k: int = 3) -> str:
    chunks = load_domain_chunks()
    if not chunks:
        return "No domain documents found in Domain_documents/."
    corpus = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_k]
    parts = []
    for i in ranked:
        if scores[i] <= 0:
            continue
        c = chunks[i]
        parts.append(f"[Source: {c['source']}]\n{c['text']}")
    return "\n\n".join(parts) if parts else chunks[0]["text"]


def fetch_wikipedia(query: str, sentences: int = 4) -> str:
    """Fetch a short Wikipedia extract via MediaWiki API (reliable User-Agent)."""
    topic = (query or "").strip() or "electronic health record"
    try:
        search = requests.get(
            _WIKI_API,
            params={
                "action": "opensearch",
                "search": topic,
                "limit": 3,
                "namespace": 0,
                "format": "json",
            },
            headers=_WIKI_HEADERS,
            timeout=15,
        )
        search.raise_for_status()
        data = search.json()
        titles = data[1] if isinstance(data, list) and len(data) > 1 else []
        if not titles:
            return f"No Wikipedia results for: {topic}"

        title = titles[0]
        extract_resp = requests.get(
            _WIKI_API,
            params={
                "action": "query",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "titles": title,
                "format": "json",
            },
            headers=_WIKI_HEADERS,
            timeout=15,
        )
        extract_resp.raise_for_status()
        pages = extract_resp.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        extract = (page.get("extract") or "").strip()
        if not extract:
            return f"Wikipedia page found ({title}) but no extract available."

        # Keep summary short (approx N sentences)
        parts = re.split(r"(?<=[.!?])\s+", extract)
        summary = " ".join(parts[: max(1, sentences)])
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        return f"[Wikipedia: {title} | {url}]\n{summary}"
    except Exception as exc:  # pragma: no cover
        # Never block the SDLC pipeline on wiki failures
        return (
            f"Wikipedia temporarily unavailable ({exc}). "
            "Continuing with domain RAG context only."
        )


def build_context(requirement: str, wiki_topic: Optional[str] = None) -> str:
    domain = retrieve_domain(requirement)
    wiki = fetch_wikipedia(wiki_topic or "electronic health record")
    return (
        "=== HEALTHCARE DOMAIN RAG CONTEXT ===\n"
        f"{domain}\n\n"
        "=== WIKIPEDIA CONTEXT ===\n"
        f"{wiki}\n"
    )
