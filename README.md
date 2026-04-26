# Talent Taiwan — Guarded RAG Demo

Production-grade reference architecture for a government regulatory chatbot.
Built as a skill-test deliverable for [Talent Taiwan](https://talent.nat.gov.tw/).

**Live demo:** [talent-taiwan-rag.streamlit.app](https://talent-taiwan-rag.streamlit.app/)

## What this is

A working Streamlit demo comparing two architectures for answering regulatory
questions about Taiwan's Employment Gold Card:

1. **Naive setup** — single system prompt, no retrieval, no guards. The
   Chatbase-style baseline.
2. **Guarded RAG** — input guard, retrieval over scraped Talent Taiwan
   content, citations, output guard.

Both run on Gemini 3 Flash. The architecture is the only variable.

## Three demos

- **Side-by-side test** — same question, both pipelines, instant comparison.
- **How it works** — every answer with its full forensic trace (input check,
  retrieval scores, generation, output check, correlation ID).
- **Content sync** — live incremental sync that re-fetches a page from
  `goldcard.nat.gov.tw`, hash-compares, re-embeds only on diff. Cost
  projection at production scale.

## Local setup

```bash
git clone https://github.com/hmtcelik/futureward-chatbot.git
cd futureward-chatbot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env — paste your GEMINI_API_KEY
streamlit run app.py
```

The repo includes pre-scraped content, a built ChromaDB index, and the
manifest, so the demo runs immediately. To re-crawl from scratch:

```bash
python -m scripts.initial_crawl
```

## Architecture

- **LLM** — Gemini 3 Flash (`gemini-3-flash-preview`) via the new
  `google-genai` SDK.
- **Embeddings** — `gemini-embedding-001`, 3,072 dimensions, asymmetric task
  types (`RETRIEVAL_DOCUMENT` for indexing, `RETRIEVAL_QUERY` for search).
- **Vector store** — ChromaDB persistent client, file-based, committed to
  the repo so Streamlit Cloud has a hot index on first boot.
- **Frontend** — Streamlit 1.56 with custom editorial CSS (Fraunces serif +
  Inter Tight + JetBrains Mono).
- **Guards** — two-layer LLM-judge architecture. Input guard fails closed
  (refuses on parse error); output guard fails open (escalates rather than
  blocking). Pipeline parallelizes input guard + embedding via
  `asyncio.gather`.
- **Observability** — `structlog` JSON logging, correlation IDs threaded
  through every pipeline stage, copyable from the UI for log lookup.

## Repository layout

```
app.py                  Streamlit entry point + navigation router
views/                  Page implementations (home, comparison, inside, sync)
src/
  config.py             Pydantic settings (env-driven)
  logger.py             structlog setup
  models.py             Pydantic data contracts
  scraper/              async crawler + change detection
  rag/                  chunking, embedding, vector store, retrieval
  guards/               input + output LLM judges + prompts
  llm/                  Gemini client + naive/guarded chatbots
  pipeline/             incremental sync logic
  ui/                   shared theme + components + runtime
scripts/
  initial_crawl.py      crawl + chunk + embed (one-shot pipeline)
  snapshot_originals.py back up extracted content for the sync demo
data/
  scraped/extracted/    cleaned per-doc JSON (committed)
  scraped/originals/    pristine snapshots (committed)
  chroma_db/            persistent vector store (committed)
  manifest.json         URL → content_hash registry
  decisions.md          design-decision log
  crawl_notes.md        scope notes for the PDF write-up
tests/                  guard + chatbot live-API smoke tests
```

## Testing

```bash
pip install pytest pytest-asyncio
SKIP_LIVE_TESTS=1 pytest               # imports only
pytest tests/test_guards.py            # 6 live guard cases
pytest tests/test_chatbots.py          # 5 live chatbot cases (~$0.005)
```

## Deploy

The repo is structured so a one-click deploy on
[Streamlit Community Cloud](https://share.streamlit.io) works:

1. Push to a public GitHub repo.
2. Connect the repo on Streamlit Cloud, point to `app.py`.
3. Paste `GEMINI_API_KEY = "…"` into the Cloud dashboard's Secrets editor.
4. Deploy.

Pre-built ChromaDB and scraped content travel with the repo, so first boot
serves real answers instantly — no crawl on cold start.

## Author

Hamit Çelik · Skill test submission for the AI Technical Consultant role at
Talent Taiwan, April 2026.
