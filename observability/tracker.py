"""Simple observability, traceability, and agent quality metrics."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STORE = Path(__file__).resolve().parents[1] / "outputs" / "traces.jsonl"
STORE.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class AgentTrace:
    run_id: str
    agent: str
    role: str
    input_preview: str
    output: str
    started_at: float
    ended_at: float
    latency_sec: float
    relevancy: float
    hallucination_risk: float
    confidence: float
    notes: str = ""


@dataclass
class RunTrace:
    run_id: str
    requirement: str
    started_at: float
    ended_at: float | None = None
    agents: list[AgentTrace] = field(default_factory=list)
    status: str = "running"


def new_run_id() -> str:
    return str(uuid.uuid4())[:8]


def _score_metrics(requirement: str, output: str) -> tuple[float, float, float]:
    """Heuristic metrics for relevancy / hallucination risk / confidence."""
    req_tokens = set(requirement.lower().split())
    out_tokens = set(output.lower().split())
    if not out_tokens:
        return 0.1, 0.9, 0.1
    overlap = len(req_tokens & out_tokens) / max(len(req_tokens), 1)
    length_penalty = 0.0 if 80 <= len(output.split()) <= 1200 else 0.15
    healthcare_cues = {
        "patient",
        "hipaa",
        "ehr",
        "clinical",
        "fhir",
        "phi",
        "healthcare",
        "medical",
        "hospital",
        "care",
    }
    domain_hits = len(healthcare_cues & out_tokens)
    relevancy = min(0.95, 0.35 + overlap * 0.4 + min(domain_hits, 5) * 0.06)
    hallucination_risk = max(0.05, 0.55 - overlap * 0.3 - min(domain_hits, 5) * 0.05 + length_penalty)
    confidence = max(0.05, min(0.95, relevancy * (1 - hallucination_risk * 0.5)))
    return round(relevancy, 2), round(hallucination_risk, 2), round(confidence, 2)


class ObservabilityTracker:
    def __init__(self, requirement: str):
        self.run = RunTrace(run_id=new_run_id(), requirement=requirement, started_at=time.time())

    def record_agent(self, agent: str, role: str, input_text: str, output: str, notes: str = "") -> AgentTrace:
        started = time.time() - 0.01
        ended = time.time()
        relevancy, hallucination_risk, confidence = _score_metrics(self.run.requirement, output)
        trace = AgentTrace(
            run_id=self.run.run_id,
            agent=agent,
            role=role,
            input_preview=input_text[:300],
            output=output,
            started_at=started,
            ended_at=ended,
            latency_sec=round(ended - started, 3),
            relevancy=relevancy,
            hallucination_risk=hallucination_risk,
            confidence=confidence,
            notes=notes,
        )
        self.run.agents.append(trace)
        self._append(trace)
        return trace

    def finish(self, status: str = "completed") -> RunTrace:
        self.run.ended_at = time.time()
        self.run.status = status
        with STORE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "run", **asdict(self.run)}, default=str) + "\n")
        return self.run

    def _append(self, trace: AgentTrace) -> None:
        with STORE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "agent", **asdict(trace)}, default=str) + "\n")

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run.run_id,
            "status": self.run.status,
            "agent_count": len(self.run.agents),
            "avg_confidence": round(
                sum(a.confidence for a in self.run.agents) / max(len(self.run.agents), 1), 2
            ),
            "avg_relevancy": round(
                sum(a.relevancy for a in self.run.agents) / max(len(self.run.agents), 1), 2
            ),
            "avg_hallucination_risk": round(
                sum(a.hallucination_risk for a in self.run.agents) / max(len(self.run.agents), 1), 2
            ),
            "agents": [
                {
                    "agent": a.agent,
                    "confidence": a.confidence,
                    "relevancy": a.relevancy,
                    "hallucination_risk": a.hallucination_risk,
                }
                for a in self.run.agents
            ],
        }
