"""Small internal MCP server for Healthcare SDLC Automation.

Transports:
  - stdio (default) — Cursor launches this via .cursor/mcp.json
  - http  — optional local server: set MCP_TRANSPORT=http

Tools expose domain RAG, feedback, traces, and lightweight project status.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv(ROOT / ".env", override=False)

_mcp_host = os.getenv("MCP_HOST", "127.0.0.1")
_mcp_port = int(os.getenv("MCP_PORT", "8765"))
mcp = FastMCP("sdlc-internal", host=_mcp_host, port=_mcp_port)


@mcp.tool()
def project_status() -> dict:
    """Return project health: domain PDFs, env key presence, output files."""
    domain = ROOT / "Domain_documents"
    pdfs = sorted(p.name for p in domain.glob("*.pdf")) if domain.exists() else []
    traces = ROOT / "outputs" / "traces.jsonl"
    feedback = ROOT / "outputs" / "feedback.jsonl"
    return {
        "root": str(ROOT),
        "groq_api_key_set": bool(os.getenv("GROQ_API_KEY")),
        "groq_model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        "domain_pdfs": pdfs,
        "traces_exists": traces.exists(),
        "feedback_exists": feedback.exists(),
        "app_entry": "app.py",
    }


@mcp.tool()
def search_domain(query: str, top_k: int = 3) -> str:
    """BM25 search over Domain_documents PDFs (healthcare RAG corpus)."""
    from rag.knowledge import retrieve_domain

    k = max(1, min(int(top_k), 8))
    return retrieve_domain(query.strip() or "healthcare EHR HIPAA", top_k=k)


@mcp.tool()
def wikipedia_context(topic: str = "electronic health record", sentences: int = 4) -> str:
    """Fetch a short Wikipedia extract for a healthcare topic."""
    from rag.knowledge import fetch_wikipedia

    return fetch_wikipedia(topic.strip() or "electronic health record", sentences=max(1, min(int(sentences), 8)))


@mcp.tool()
def recent_feedback(limit: int = 10) -> list:
    """Load recent agent feedback ratings from outputs/feedback.jsonl."""
    from feedback.store import load_feedback

    return load_feedback(max(1, min(int(limit), 50)))


@mcp.tool()
def recent_traces(limit: int = 5) -> list:
    """Load recent observability run/agent traces from outputs/traces.jsonl."""
    path = ROOT / "outputs" / "traces.jsonl"
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-max(1, min(int(limit), 30)) :]


@mcp.tool()
def list_agents() -> list:
    """List the 5 SDLC agents and their roles."""
    from agents.crew_pipeline import AGENTS

    return [{"key": a["key"], "name": a["name"], "role": a["role"]} for a in AGENTS]


def main() -> None:
    transport = (os.getenv("MCP_TRANSPORT") or "stdio").strip().lower()
    if transport in {"http", "streamable-http"}:
        # Local-only internal HTTP MCP endpoint (host/port set on FastMCP above)
        mcp.run(transport="streamable-http")
    elif transport == "sse":
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
