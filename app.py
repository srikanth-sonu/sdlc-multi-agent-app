"""Streamlit UI for healthcare SDLC multi-agent automation."""
from __future__ import annotations

import hmac
import os
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _apply_cloud_secrets() -> None:
    """Map Streamlit / HF secrets into env vars used by the pipeline."""
    try:
        secrets = st.secrets
    except Exception:
        return
    for key in ("GROQ_API_KEY", "GROQ_MODEL", "APP_PASSWORD", "MAX_RUNS_PER_HOUR"):
        try:
            value = secrets.get(key) if hasattr(secrets, "get") else secrets[key]
        except Exception:
            value = None
        if value:
            os.environ[key] = str(value)


st.set_page_config(page_title="SDLC Automation", page_icon="🏥", layout="wide")
_apply_cloud_secrets()

from agents.crew_pipeline import run_sdlc_pipeline
from feedback.store import MAX_COMMENT_CHARS, load_feedback, save_feedback

MAX_RUNS_PER_HOUR = int(os.getenv("MAX_RUNS_PER_HOUR", "20"))


@st.cache_resource
def _run_log() -> list[float]:
    """Run timestamps shared by all sessions, to cap spend on the Groq key."""
    return []


def _claim_run() -> bool:
    log = _run_log()
    cutoff = time.time() - 3600
    log[:] = [t for t in log if t > cutoff]
    if len(log) >= MAX_RUNS_PER_HOUR:
        return False
    log.append(time.time())
    return True


st.title("Healthcare SDLC Automation")
st.caption("6-agent SDLC flow · Domain RAG · Wikipedia · Observability · Feedback")

with st.sidebar:
    st.header("Settings")
    app_password = os.getenv("APP_PASSWORD", "")
    if app_password:
        entered = st.text_input("App password", type="password")
        if not hmac.compare_digest(entered.encode(), app_password.encode()):
            st.info("Enter the app password to continue.")
            st.stop()
    wiki_topic = st.text_input("Wikipedia topic", value="electronic health record")
    st.markdown(
        """
**Agents**
1. Business Analyst  
2. Solution Architect  
3. Full-Stack Developer (Python + Streamlit)  
4. QA Engineer  
5. DevOps Engineer (CI/CD + deployment)  
6. Technical Writer (Word release notes)
"""
    )
    prior = load_feedback(10)
    st.subheader("Recent feedback")
    if prior:
        for row in reversed(prior[-5:]):
            st.write(f"{row.get('agent')}: ⭐{row.get('rating')} — {row.get('comment') or '(no comment)'}")
    else:
        st.write("No feedback yet.")

requirement = st.text_area(
    "Business / functional / non-functional requirement",
    height=160,
    placeholder="Example: Build a patient appointment booking module for a clinic EHR with HIPAA-compliant audit logs.",
    value=(
        "Build a patient appointment booking module for a small clinic EHR. "
        "Functional: schedule/reschedule/cancel visits, notify patients, clinician calendar. "
        "Non-functional: HIPAA-aware access control, audit logging, 99.5% availability, "
        "Streamlit UI and Python backend."
    ),
)

run_clicked = st.button("Generate full SDLC", type="primary", use_container_width=True)

if run_clicked:
    _apply_cloud_secrets()
    if not os.getenv("GROQ_API_KEY"):
        st.error(
            "GROQ_API_KEY is missing. On Streamlit Cloud: App menu → Settings → Secrets, then paste:\n\n"
            'GROQ_API_KEY = "your_key"\n'
            'GROQ_MODEL = "openai/gpt-oss-20b"'
        )
    elif not requirement.strip():
        st.error("Please enter a requirement.")
    elif not _claim_run():
        st.error(f"Hourly limit of {MAX_RUNS_PER_HOUR} runs reached for this app. Please try again later.")
    else:
        with st.spinner("Running 6-agent SDLC crew (this may take a few minutes)..."):
            result = run_sdlc_pipeline(requirement.strip(), wiki_topic=wiki_topic.strip() or "electronic health record")
        st.session_state["last_result"] = result

result = st.session_state.get("last_result")

if result:
    if not result.get("ok"):
        st.error(f"Pipeline failed: {result.get('error')}")
        st.json(result.get("metrics", {}))
    else:
        st.success(f"Run completed · ID `{result['run_id']}` · engine `{result.get('engine', 'n/a')}`")
        st.download_button(
            "Download Word release notes",
            data=Path(result["docx_path"]).read_bytes(),
            file_name=Path(result["docx_path"]).name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

        tab_art, tab_obs, tab_ctx, tab_fb = st.tabs(
            ["Artifacts", "Observability & Metrics", "RAG / Wikipedia Context", "Agent Feedback"]
        )

        with tab_art:
            for name, content in result.get("artifacts", {}).items():
                with st.expander(name, expanded=(name == "Business Analyst")):
                    st.markdown(content)

        with tab_obs:
            metrics = result.get("metrics", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg confidence", metrics.get("avg_confidence", 0))
            c2.metric("Avg relevancy", metrics.get("avg_relevancy", 0))
            c3.metric("Avg hallucination risk", metrics.get("avg_hallucination_risk", 0))
            st.subheader("Per-agent metrics")
            for row in metrics.get("agents", []):
                st.write(
                    f"**{row['agent']}** — confidence `{row['confidence']}` · "
                    f"relevancy `{row['relevancy']}` · hallucination risk `{row['hallucination_risk']}`"
                )
            with st.expander("Metric formulas (how values are generated)", expanded=True):
                st.markdown(
                    """
**Token overlap**  
`overlap = |req_tokens ∩ out_tokens| / |req_tokens|`

**Domain hits**  
Count of healthcare cue words in the output  
`(patient, hipaa, ehr, clinical, fhir, phi, healthcare, medical, hospital, care)`  
capped at 5.

**Length penalty**  
`0.0` if output words ∈ [80, 1200], else `0.15`

**Relevancy**  
`min(0.95, 0.35 + overlap×0.4 + min(domain_hits,5)×0.06)`

**Hallucination risk**  
`max(0.05, 0.55 − overlap×0.3 − min(domain_hits,5)×0.05 + length_penalty)`

**Confidence**  
`clamp(relevancy × (1 − hallucination_risk × 0.5), 0.05, 0.95)`

These are heuristic audit scores for traceability (not model-native logits).
"""
                )
            st.caption("Traces are appended to outputs/traces.jsonl for auditability.")

        with tab_ctx:
            st.code(result.get("context_preview", ""), language="markdown")

        with tab_fb:
            st.write("Rate an agent response. Feedback is reused on the next run.")
            agents = list(result.get("artifacts", {}).keys()) or [
                "Business Analyst",
                "Solution Architect",
                "Full-Stack Developer",
                "QA Engineer",
                "DevOps Engineer",
                "Technical Writer",
            ]
            agent = st.selectbox("Agent", agents)
            rating = st.slider("Rating", 1, 5, 4)
            comment = st.text_input(
                "Feedback comment",
                placeholder="Be more specific about HIPAA controls",
                max_chars=MAX_COMMENT_CHARS,
            )
            if st.button("Submit feedback"):
                excerpt = result.get("artifacts", {}).get(agent, "")[:500]
                save_feedback(result["run_id"], agent, rating, comment, excerpt)
                st.success("Feedback saved — will influence the next run for this agent.")
