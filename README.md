---
title: Healthcare SDLC Automation
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.40.1
app_file: app.py
pinned: false
license: mit
short_description: 5-agent healthcare SDLC pack with RAG, metrics, and Word export
---

# Healthcare SDLC Automation

Streamlit app that turns a healthcare software requirement into a full SDLC pack using 5 sequential AI agents, domain RAG, Wikipedia context, observability metrics, and user feedback.

## Live deploy (free)

### A) Streamlit Community Cloud

1. Push this repo to GitHub.
2. Open [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select the repo, branch `main`, main file `app.py`.
4. Under **Advanced settings → Secrets**, paste:

```toml
GROQ_API_KEY = "your_groq_key"
GROQ_MODEL = "openai/gpt-oss-20b"
```

5. Deploy.

### B) Hugging Face Spaces

1. Create a Space with SDK **Streamlit**.
2. Push this repo to the Space git remote (or upload files).
3. In **Settings → Variables and secrets**, add:
   - `GROQ_API_KEY` (secret)
   - `GROQ_MODEL` = `openai/gpt-oss-20b` (optional)
4. Wait for the build; open the Space URL.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional: copy `.env.example` to `.env` and set `GROQ_API_KEY`.

## Notes

- Free hosts use the **Groq sequential** agent path (lightweight). CrewAI is optional locally.
- Domain PDFs live in `Domain_documents/`.
- Generated Word docs / traces write under `outputs/` (ephemeral on free hosts).
