# Healthcare SDLC Automation — Project Agenda

Streamlit app that turns a healthcare software requirement into a full SDLC pack using **5 sequential AI agents**, domain RAG, Wikipedia context, observability metrics, and user feedback.

---

## 1. Goal

Given a business / functional / non-functional requirement (e.g. clinic EHR appointment booking), produce:

1. Requirements analysis (user stories, acceptance criteria, process flow)
2. Solution architecture (components, data, APIs, HIPAA/security)
3. Implementation design (Python + Streamlit)
4. QA plan (pytest + automation cases)
5. Release notes (Word `.docx`)

---

## 2. How to Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

**Env (`.env`):**

| Variable       | Purpose                          |
|----------------|----------------------------------|
| `GROQ_API_KEY` | Required for LLM calls           |
| `GROQ_MODEL`   | Default: `openai/gpt-oss-20b`  |

---

## 3. Agent Pipeline (sequential)

| # | Agent                 | Key         | Deliverable                                      |
|---|-----------------------|-------------|--------------------------------------------------|
| 1 | Business Analyst      | `ba`        | User stories, acceptance criteria, mermaid flow  |
| 2 | Solution Architect    | `architect` | Components, data model, APIs, HIPAA, deployment  |
| 3 | Full-Stack Developer  | `developer` | Python backend + Streamlit UI design/snippets    |
| 4 | QA Engineer           | `tester`    | pytest units + automation (happy + edge paths)   |
| 5 | Technical Writer      | `docs`      | SDLC summary + release notes                     |

**Execution engines** (`agents/crew_pipeline.py`):

1. **Preferred:** CrewAI sequential `Crew` with custom Groq `BaseLLM`
2. **Fallback:** Direct Groq chat loop if CrewAI fails at runtime

Each agent receives: requirement + RAG/Wikipedia context + prior-agent context + past user feedback for that role.

---

## 4. End-to-End Flow

```
User enters requirement (+ optional Wikipedia topic)
        │
        ▼
app.py  →  run_sdlc_pipeline()
        │
        ├─ build_context()          # Domain PDFs (BM25) + Wikipedia
        ├─ CrewAI or Groq agents    # 5 sequential roles
        ├─ ObservabilityTracker     # scores + traces.jsonl
        └─ write_release_notes()    # outputs/release_notes_<run_id>.docx
        │
        ▼
UI tabs: Artifacts | Metrics | RAG Context | Feedback
```

---

## 5. Modules

| Path                         | Role |
|------------------------------|------|
| `app.py`                     | Streamlit UI — run button, tabs, download, feedback |
| `agents/crew_pipeline.py`    | Agent defs, CrewAI/Groq runners, pipeline entry |
| `rag/knowledge.py`           | PDF chunking, BM25 retrieval, Wikipedia extract |
| `observability/tracker.py`   | Run IDs, heuristic metrics, `outputs/traces.jsonl` |
| `feedback/store.py`          | Ratings → `outputs/feedback.jsonl`, reused in prompts |
| `utils/docx_writer.py`       | Word release-notes generator |
| `Domain_documents/*.pdf`     | HIPAA / EHR / clinical domain corpus |
| `scripts/generate_domain_pdfs.py` | Helper to (re)build domain PDFs |

---

## 6. RAG Context

- **Domain:** PDFs in `Domain_documents/` → chunks (~500 chars) → BM25 top-k vs requirement
- **Wikipedia:** MediaWiki API extract for sidebar topic (default: `electronic health record`)
- Combined into one context string injected into every agent prompt (truncated for Groq limits)

---

## 7. Observability Metrics (heuristic)

Not model logits — audit scores from token overlap + healthcare cue words:

- **Relevancy** — requirement overlap + domain terms  
- **Hallucination risk** — inverse of overlap/domain hits + length penalty  
- **Confidence** — `relevancy × (1 − hallucination_risk × 0.5)`  

Persisted per agent/run in `outputs/traces.jsonl`.

---

## 8. Feedback Loop

- UI: rate any agent (1–5) + comment after a run  
- Stored in `outputs/feedback.jsonl`  
- Next run: `feedback_prompt_for_agent()` injects liked / disliked comments into that agent’s prompt  

---

## 9. UI Agenda (user journey)

1. Set Wikipedia topic in sidebar (optional)  
2. Enter / edit healthcare requirement  
3. Click **Generate full SDLC**  
4. Review five agent artifacts  
5. Check observability metrics  
6. Inspect RAG / Wikipedia context  
7. Submit agent feedback for the next run  
8. Download Word release notes  

---

## 10. Outputs

| Artifact                         | Location |
|----------------------------------|----------|
| Agent traces                     | `outputs/traces.jsonl` |
| User feedback                    | `outputs/feedback.jsonl` |
| Release notes Word doc           | `outputs/release_notes_<run_id>.docx` |

(`outputs/` is gitignored.)

Set `GROQ_API_KEY` only in `.env` (local) or Streamlit Cloud Secrets — never commit the real key.

