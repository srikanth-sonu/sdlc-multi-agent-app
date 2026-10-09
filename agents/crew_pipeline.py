"""SDLC multi-agent pipeline (CrewAI when available, Groq sequential fallback)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from feedback.store import feedback_prompt_for_agent
from observability.tracker import ObservabilityTracker
from rag.knowledge import build_context
from utils.docx_writer import write_release_notes

ROOT = Path(__file__).resolve().parents[1]
# Do not override env vars already set by Streamlit Cloud / HF secrets.
load_dotenv(ROOT / ".env", override=False)

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_groq_api_key() -> str:
    """Prefer process env, then Streamlit secrets (Community Cloud)."""
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if key:
        return key
    try:
        import streamlit as st

        value = st.secrets.get("GROQ_API_KEY")  # type: ignore[attr-defined]
        if value:
            key = str(value).strip()
            os.environ["GROQ_API_KEY"] = key
            return key
    except Exception:
        pass
    raise RuntimeError(
        "GROQ_API_KEY missing. Set it in Streamlit Cloud Secrets, or in local .env / .streamlit/secrets.toml"
    )

AGENTS = [
    {
        "key": "ba",
        "name": "Business Analyst",
        "role": "Business Analyst",
        "system": (
            "You are an expert healthcare Business Analyst. Clarify business, "
            "functional, and non-functional requirements. Respect HIPAA/PHI. "
            "Produce user stories, acceptance criteria, and a mermaid flowchart "
            "of the main process flow."
        ),
    },
    {
        "key": "architect",
        "name": "Solution Architect",
        "role": "Solution Architect",
        "system": (
            "You are a healthcare Solution Architect. Design pragmatic architecture: "
            "components, data model, APIs, security/HIPAA, deployment."
        ),
    },
    {
        "key": "developer",
        "name": "Full-Stack Developer",
        "role": "Full-Stack Developer",
        "system": (
            "You are an expert Python developer. Produce Python backend design and "
            "Streamlit frontend structure with clear code snippets."
        ),
    },
    {
        "key": "tester",
        "name": "QA Engineer",
        "role": "QA Engineer",
        "system": (
            "You are a QA engineer. Create pytest unit tests and automation test "
            "cases covering happy path and edge cases for healthcare workflows."
        ),
    },
    {
        "key": "docs",
        "name": "Technical Writer",
        "role": "Technical Writer",
        "system": (
            "You are a technical writer skilled at Word-ready documentation. "
            "Produce SDLC process summary and release notes."
        ),
    },
]


def _groq_client() -> Groq:
    return Groq(api_key=_resolve_groq_api_key())


def _chat(client: Groq, system: str, user: str) -> str:
    import time

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if "rate_limit" in msg or "429" in msg:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    raise last_exc  # type: ignore[misc]


def _clean_messages(messages: Any) -> list[dict[str, str]]:
    """Normalize CrewAI messages and strip unsupported Groq fields."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    cleaned: list[dict[str, str]] = []
    for msg in messages or []:
        if isinstance(msg, dict):
            role = str(msg.get("role") or "user")
            content = msg.get("content")
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        parts.append(str(block["text"]))
                    else:
                        parts.append(str(block))
                content = "\n".join(parts)
            cleaned.append({"role": role, "content": str(content or "")})
        else:
            cleaned.append({"role": "user", "content": str(msg)})
    return cleaned


def _make_crewai_llm():
    """CrewAI BaseLLM backed by Groq SDK (avoids LiteLLM cache_breakpoint bug)."""
    from crewai.llms.base_llm import BaseLLM

    api_key = _resolve_groq_api_key()
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").replace("groq/", "")
    try:
        import streamlit as st

        if not os.getenv("GROQ_MODEL") and st.secrets.get("GROQ_MODEL"):
            model = str(st.secrets["GROQ_MODEL"]).replace("groq/", "")
            os.environ["GROQ_MODEL"] = model
    except Exception:
        pass

    class GroqCrewLLM(BaseLLM):
        def __init__(self):
            super().__init__(model=model, temperature=0.3, provider="groq")
            self._client = Groq(api_key=api_key)

        def call(self, messages, tools=None, callbacks=None, available_functions=None, from_task=None, from_agent=None, response_model=None):
            import time

            cleaned = _clean_messages(messages)
            # Keep free-tier Groq TPM under ~6k tokens (~4 chars/token rough)
            total = 0
            capped: list[dict[str, str]] = []
            for msg in cleaned:
                content = _truncate(msg["content"], 3500)
                total += len(content)
                if total > 12000:
                    content = _truncate(content, max(200, 12000 - (total - len(content))))
                capped.append({"role": msg["role"], "content": content})
            cleaned = capped

            last_exc: Exception | None = None
            for attempt in range(4):
                try:
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature or 0.3,
                        messages=cleaned,
                        max_tokens=1200,
                    )
                    return (resp.choices[0].message.content or "").strip()
                except Exception as exc:
                    last_exc = exc
                    msg = str(exc).lower()
                    if "rate_limit" in msg or "429" in msg or "413" in msg:
                        time.sleep(8 * (attempt + 1))
                        # Shrink further on retry
                        cleaned = [
                            {"role": m["role"], "content": _truncate(m["content"], max(400, 2500 - attempt * 500))}
                            for m in cleaned
                        ]
                        continue
                    raise
            raise last_exc  # type: ignore[misc]

        def supports_function_calling(self) -> bool:
            return False

    return GroqCrewLLM()


def _run_with_crewai(requirement: str, context: str) -> dict[str, str]:
    """Preferred path: native CrewAI sequential crew."""
    from crewai import Agent, Crew, Process, Task

    llm = _make_crewai_llm()
    context = _truncate(context, 2500)
    requirement = _truncate(requirement, 1200)

    agents_map: dict[str, Agent] = {}
    for spec in AGENTS:
        agents_map[spec["key"]] = Agent(
            role=spec["role"],
            goal=spec["system"],
            backstory=spec["system"] + " Apply healthcare domain RAG and prior feedback.",
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

    tasks: list[Task] = []
    for i, spec in enumerate(AGENTS):
        fb = _truncate(feedback_prompt_for_agent(spec["role"]), 600)
        prior = ""
        if tasks:
            prior = "Build on previous agent outputs in context."
        task_kwargs = {
            "description": (
                f"Requirement:\n{requirement}\n\nContext:\n{context}\n\n"
                f"{fb}\n\n{prior}\n\nYour job: {spec['system']}"
            ),
            "expected_output": f"Deliverable from {spec['role']}",
            "agent": agents_map[spec["key"]],
        }
        if tasks:
            # Avoid feeding full prior task bodies (blows Groq free-tier TPM).
            # Only pass the immediately previous task as compact context.
            task_kwargs["context"] = [tasks[-1]]
        task = Task(**task_kwargs)
        tasks.append(task)

    crew = Crew(
        agents=list(agents_map.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff()
    artifacts: dict[str, str] = {}
    outputs = getattr(result, "tasks_output", None) or []
    if outputs and len(outputs) >= len(AGENTS):
        for spec, out in zip(AGENTS, outputs):
            artifacts[spec["name"]] = str(getattr(out, "raw", None) or out)
    else:
        text = str(result)
        for spec in AGENTS:
            artifacts[spec["name"]] = text
    return artifacts


def _truncate(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n...[truncated]..."


def _run_with_groq(requirement: str, context: str) -> dict[str, str]:
    """Fallback sequential multi-agent flow using Groq directly."""
    client = _groq_client()
    artifacts: dict[str, str] = {}
    history = ""
    context = _truncate(context, 2500)
    requirement = _truncate(requirement, 1200)
    for spec in AGENTS:
        fb = feedback_prompt_for_agent(spec["role"])
        user = (
            f"Healthcare software requirement:\n{requirement}\n\n"
            f"Domain + Wikipedia context:\n{context}\n\n"
            f"{_truncate(fb, 600)}\n\n"
            f"Prior agent outputs:\n{_truncate(history, 3500) or '(none yet)'}\n\n"
            f"Produce your deliverable as {spec['role']}."
        )
        output = _chat(client, spec["system"], user)
        artifacts[spec["name"]] = output
        history += f"\n\n### {spec['name']}\n{_truncate(output, 1200)}"
    return artifacts


def run_sdlc_pipeline(requirement: str, wiki_topic: str = "electronic health record") -> dict[str, Any]:
    tracker = ObservabilityTracker(requirement)
    context = build_context(requirement, wiki_topic=wiki_topic)
    engine = "groq-sequential"

    try:
        try:
            import crewai  # noqa: F401

            artifacts = _run_with_crewai(requirement, context)
            engine = "crewai"
        except Exception as crew_exc:
            # Prefer CrewAI; fall back to Groq sequential if CrewAI fails at runtime
            artifacts = _run_with_groq(requirement, context)
            engine = f"groq-sequential (crewai_error={type(crew_exc).__name__}: {crew_exc})"

        for spec in AGENTS:
            output = artifacts.get(spec["name"], "")
            tracker.record_agent(
                agent=spec["name"],
                role=spec["role"],
                input_text=requirement,
                output=output,
                notes=f"engine={engine}",
            )

        docx_path = OUTPUT_DIR / f"release_notes_{tracker.run.run_id}.docx"
        write_release_notes(
            docx_path,
            title=f"SDLC Release Notes — {tracker.run.run_id}",
            requirement=requirement,
            sections=artifacts,
        )
        tracker.finish("completed")
        return {
            "ok": True,
            "run_id": tracker.run.run_id,
            "engine": engine,
            "context_preview": context[:1200],
            "artifacts": artifacts,
            "docx_path": str(docx_path),
            "metrics": tracker.summary(),
            "final": artifacts.get("Technical Writer", ""),
        }
    except Exception as exc:
        tracker.finish("failed")
        return {
            "ok": False,
            "run_id": tracker.run.run_id,
            "error": str(exc),
            "metrics": tracker.summary(),
        }
