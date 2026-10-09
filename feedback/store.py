"""Simple feedback store so agents can learn from prior user ratings."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FEEDBACK_PATH = Path(__file__).resolve().parents[1] / "outputs" / "feedback.jsonl"
FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)


def save_feedback(
    run_id: str,
    agent: str,
    rating: int,
    comment: str,
    response_excerpt: str = "",
) -> None:
    record = {
        "run_id": run_id,
        "agent": agent,
        "rating": int(rating),
        "comment": comment.strip(),
        "response_excerpt": response_excerpt[:500],
    }
    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_feedback(limit: int = 50) -> list[dict[str, Any]]:
    if not FEEDBACK_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    with FEEDBACK_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[-limit:]


def feedback_prompt_for_agent(agent: str) -> str:
    """Build instruction text from prior feedback for this agent."""
    rows = [r for r in load_feedback(100) if r.get("agent") == agent]
    if not rows:
        return "No prior user feedback for this agent."
    low = [r for r in rows if int(r.get("rating", 3)) <= 2]
    high = [r for r in rows if int(r.get("rating", 3)) >= 4]
    parts = ["Prior user feedback to improve your next response:"]
    if high:
        parts.append("What users liked:")
        for r in high[-3:]:
            if r.get("comment"):
                parts.append(f"- {r['comment']}")
    if low:
        parts.append("What users disliked (avoid repeating):")
        for r in low[-3:]:
            if r.get("comment"):
                parts.append(f"- {r['comment']}")
    return "\n".join(parts)
